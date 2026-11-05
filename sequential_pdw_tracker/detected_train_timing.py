"""Shadow timing model learned only from a track's detected TOA sequence.

This module models received inter-arrival gaps. It deliberately does not call
an adjacent gap a transmitter PRI and never contributes to active association.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


LOG_2PI = math.log(2.0 * math.pi)


def _normal_logpdf(residual: float, variance: float) -> float:
    variance = max(float(variance), 1e-12)
    return -0.5 * (LOG_2PI + math.log(variance) + residual * residual / variance)


@dataclass(frozen=True)
class DetectedTrainTimingConfig:
    min_gap_s: float = 0.25e-6
    max_gap_s: float = 50e-3
    histogram_bins: int = 96
    histogram_pseudocount: float = 0.5
    visibility_warmup_gaps: int = 32
    active_to_coast_probability: float = 0.02
    coast_to_active_probability: float = 0.80

    def __post_init__(self) -> None:
        if self.min_gap_s <= 0.0 or self.max_gap_s <= self.min_gap_s:
            raise ValueError("invalid detected-gap range")
        if self.histogram_bins < 8 or self.histogram_pseudocount <= 0.0:
            raise ValueError("invalid detected-gap histogram configuration")
        if self.visibility_warmup_gaps < 4:
            raise ValueError("visibility_warmup_gaps must be at least four")
        probabilities = (
            self.active_to_coast_probability,
            self.coast_to_active_probability,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("visibility transition probabilities must be in (0, 1)")


@dataclass(frozen=True)
class DetectedTrainTimingUpdate:
    gap_s: float | None
    log_predictive_density: float
    active_posterior: float
    coast_posterior: float
    ready: bool
    feedback_weight: float = 0.0


class DetectedTrainTimingEstimator:
    """Online empirical timing shadow for already accepted track pulses."""

    def __init__(self, config: DetectedTrainTimingConfig | None = None):
        self.config = config or DetectedTrainTimingConfig()
        self._edges = np.geomspace(
            self.config.min_gap_s,
            self.config.max_gap_s,
            self.config.histogram_bins + 1,
        )
        self._counts = np.zeros(self.config.histogram_bins, dtype=float)
        self._gaps: list[float] = []
        self._last_toa_s: float | None = None
        self._visibility_mean = np.empty(0, dtype=float)
        self._visibility_variance = np.empty(0, dtype=float)
        self._visibility_count = np.empty(0, dtype=float)
        self._visibility_posterior = np.asarray([0.5, 0.5], dtype=float)

    def _bin_index(self, gap_s: float) -> int:
        return int(np.clip(np.searchsorted(self._edges, gap_s, side="right") - 1,
                           0, len(self._counts) - 1))

    def _predictive_log_density(self, gap_s: float) -> float:
        index = self._bin_index(gap_s)
        alpha = self.config.histogram_pseudocount
        probability = (self._counts[index] + alpha) / (
            self._counts.sum() + alpha * len(self._counts)
        )
        width = self._edges[index + 1] - self._edges[index]
        return math.log(float(probability)) - math.log(float(width))

    def _initialize_visibility(self) -> None:
        log_gaps = np.log(np.asarray(self._gaps, dtype=float))
        median = float(np.median(log_gaps))
        lower = log_gaps[log_gaps <= median]
        upper = log_gaps[log_gaps > median]
        self._visibility_mean = np.asarray([
            float(np.mean(lower)), float(np.mean(upper)),
        ])
        self._visibility_variance = np.asarray([
            max(float(np.var(lower)), 0.05**2),
            max(float(np.var(upper)), 0.05**2),
        ])
        self._visibility_count = np.asarray([
            max(float(len(lower)), 1.0), max(float(len(upper)), 1.0),
        ])
        self._visibility_posterior = np.asarray([0.5, 0.5], dtype=float)

    def _update_visibility(self, gap_s: float) -> tuple[float, float]:
        initialized_now = False
        if not len(self._visibility_mean):
            if len(self._gaps) >= self.config.visibility_warmup_gaps:
                self._initialize_visibility()
                initialized_now = True
            else:
                return 0.5, 0.5

        transition = np.asarray([
            [1.0 - self.config.active_to_coast_probability,
             self.config.active_to_coast_probability],
            [self.config.coast_to_active_probability,
             1.0 - self.config.coast_to_active_probability],
        ])
        predicted = self._visibility_posterior @ transition
        value = math.log(gap_s)
        log_weight = np.log(np.maximum(predicted, 1e-300)) + np.asarray([
            _normal_logpdf(value - self._visibility_mean[index],
                           self._visibility_variance[index])
            for index in range(2)
        ])
        maximum = float(np.max(log_weight))
        posterior = np.exp(log_weight - maximum)
        posterior /= posterior.sum()

        # The warm-up gaps, including the current one, seeded the components.
        # Do not count the current gap a second time on the initialization step.
        if not initialized_now:
            for index, responsibility in enumerate(posterior):
                residual = value - self._visibility_mean[index]
                rate = float(
                    responsibility / (self._visibility_count[index] + responsibility)
                )
                self._visibility_mean[index] += rate * residual
                self._visibility_variance[index] = max(
                    (1.0 - rate) * self._visibility_variance[index]
                    + rate * residual * residual,
                    0.05**2,
                )
                self._visibility_count[index] += responsibility

        # State zero must remain the shorter-gap (active) component.
        if self._visibility_mean[0] > self._visibility_mean[1]:
            self._visibility_mean = self._visibility_mean[::-1].copy()
            self._visibility_variance = self._visibility_variance[::-1].copy()
            self._visibility_count = self._visibility_count[::-1].copy()
            posterior = posterior[::-1].copy()
        self._visibility_posterior = posterior
        return float(posterior[0]), float(posterior[1])

    def process(self, toa_s: float) -> DetectedTrainTimingUpdate:
        if self._last_toa_s is None:
            self._last_toa_s = toa_s
            return DetectedTrainTimingUpdate(None, 0.0, 0.5, 0.5, False)
        gap_s = toa_s - self._last_toa_s
        if gap_s < 0.0:
            raise ValueError("detected TOAs must be non-decreasing")
        self._last_toa_s = toa_s
        if gap_s == 0.0:
            return DetectedTrainTimingUpdate(0.0, 0.0, 0.5, 0.5, False)

        log_density = self._predictive_log_density(gap_s)
        self._gaps.append(gap_s)
        active, coast = self._update_visibility(gap_s)
        self._counts[self._bin_index(gap_s)] += 1.0
        ready = len(self._gaps) >= self.config.visibility_warmup_gaps
        return DetectedTrainTimingUpdate(
            gap_s, log_density, active, coast, ready,
        )

    def snapshot(self) -> dict:
        gaps = np.asarray(self._gaps, dtype=float)
        quantiles = None
        if len(gaps):
            quantiles = {
                "p05_s": float(np.quantile(gaps, 0.05)),
                "p25_s": float(np.quantile(gaps, 0.25)),
                "median_s": float(np.median(gaps)),
                "p75_s": float(np.quantile(gaps, 0.75)),
                "p95_s": float(np.quantile(gaps, 0.95)),
            }
        visibility = None
        if len(self._visibility_mean):
            visibility = {
                "active_log_gap_mean": float(self._visibility_mean[0]),
                "coast_log_gap_mean": float(self._visibility_mean[1]),
                "active_posterior": float(self._visibility_posterior[0]),
                "coast_posterior": float(self._visibility_posterior[1]),
            }
        return {
            "role": "shadow detected-pulse timing; never active association evidence",
            "feedback_weight": 0.0,
            "transmitter_pri_estimate": None,
            "observed_gap_count": len(self._gaps),
            "observed_gap_quantiles": quantiles,
            "histogram_edges_s": self._edges.tolist(),
            "histogram_counts": self._counts.tolist(),
            "visibility": visibility,
            "config": asdict(self.config),
        }
