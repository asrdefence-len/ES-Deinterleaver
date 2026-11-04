"""Rao-Blackwellised single-emitter HMM used by incremental Model 1.

The discrete RF chain is filtered with the HMM forward recursion and online
expected transition counts. AOA is a circular linear-Gaussian Markov state.
TOA is handled by a bank of semi-Markov PRI hypotheses with a latent integer
number of elapsed pulse intervals. Every input PDW updates the filter.

Model 1 deliberately does not read pulse width or amplitude.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .model import PDW


LOG_2PI = math.log(2.0 * math.pi)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _normal_logpdf(residual: float, variance: float) -> float:
    variance = max(float(variance), 1e-30)
    return -0.5 * (LOG_2PI + math.log(variance) + residual * residual / variance)


def _angular_error_deg(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class HMMModel1Config:
    max_rf_states: int = 16
    rf_new_state_gate_hz: float = 50e6
    rf_sigma_floor_hz: float = 2e6
    transition_pseudocount: float = 1.0
    new_state_prior: float = 0.05
    aoa_measurement_sigma_deg: float = 1.0
    aoa_process_variance_deg2_per_s: float = 0.25
    timing_min_pri_s: float = 0.5e-6
    timing_max_pri_s: float = 20e-3
    timing_max_hypotheses: int = 64
    timing_initial_divisors: int = 128
    timing_jitter_fraction: float = 0.10
    timing_toa_sigma_s: float = 25e-9
    timing_process_variance_fraction2_per_s: float = 1e-4

    def __post_init__(self) -> None:
        positive = (
            self.max_rf_states,
            self.rf_new_state_gate_hz,
            self.rf_sigma_floor_hz,
            self.transition_pseudocount,
            self.new_state_prior,
            self.aoa_measurement_sigma_deg,
            self.timing_min_pri_s,
            self.timing_max_pri_s,
            self.timing_max_hypotheses,
            self.timing_initial_divisors,
            self.timing_jitter_fraction,
            self.timing_toa_sigma_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("HMM configuration values must be positive")
        if self.timing_min_pri_s >= self.timing_max_pri_s:
            raise ValueError("timing_min_pri_s must be below timing_max_pri_s")


@dataclass
class _RFState:
    mean_hz: float
    variance_hz2: float
    effective_count: float = 1.0


@dataclass
class _TimingHypothesis:
    pri_s: float
    variance_s2: float
    log_weight: float


@dataclass(frozen=True)
class HMMModel1Update:
    pdw_id: int
    toa_s: float
    rf_posterior: tuple[float, ...]
    rf_log_posterior: tuple[float, ...]
    rf_map_state: int
    rf_log_evidence: float
    aoa_mean_deg: float
    aoa_sigma_deg: float
    aoa_innovation_deg: float
    aoa_log_evidence: float
    timing_log_evidence: float
    timing_map_pri_s: float | None
    timing_map_intervals: int | None
    timing_hypothesis_count: int
    total_log_evidence: float


class HMMEmitterModel1:
    """Proper forward-filtering state model for one already selected emitter."""

    enabled_features = ("Frequency", "ToA", "AoA")
    excluded_features = ("PulseWidth", "Amplitude")

    def __init__(self, config: HMMModel1Config | None = None):
        self.config = config or HMMModel1Config()
        self._rf_states: list[_RFState] = []
        self._rf_posterior = np.empty(0, dtype=float)
        self._transition_counts = np.empty((0, 0), dtype=float)
        self._aoa_mean_deg: float | None = None
        self._aoa_variance_deg2: float | None = None
        self._timing: list[_TimingHypothesis] = []
        self._last_toa_s: float | None = None
        self._updates = 0

    @property
    def update_count(self) -> int:
        return self._updates

    def _add_rf_state(self, frequency_hz: float) -> None:
        state_count = len(self._rf_states)
        self._rf_states.append(_RFState(
            mean_hz=frequency_hz,
            variance_hz2=self.config.rf_sigma_floor_hz**2,
        ))
        expanded = np.full(
            (state_count + 1, state_count + 1),
            self.config.transition_pseudocount,
            dtype=float,
        )
        if state_count:
            expanded[:state_count, :state_count] = self._transition_counts
            self._rf_posterior *= 1.0 - self.config.new_state_prior
            self._rf_posterior = np.r_[
                self._rf_posterior,
                self.config.new_state_prior,
            ]
        else:
            self._rf_posterior = np.ones(1, dtype=float)
        self._rf_posterior /= self._rf_posterior.sum()
        self._transition_counts = expanded

    def _rf_forward(
        self,
        frequency_hz: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if not self._rf_states:
            self._add_rf_state(frequency_hz)
            return 0.0, self._rf_posterior.copy(), np.zeros(1, dtype=float)
        nearest = min(abs(frequency_hz - state.mean_hz) for state in self._rf_states)
        if (nearest > self.config.rf_new_state_gate_hz
                and len(self._rf_states) < self.config.max_rf_states):
            self._add_rf_state(frequency_hz)

        previous = self._rf_posterior.copy()
        transition = self._transition_counts / self._transition_counts.sum(axis=1, keepdims=True)
        predicted = previous @ transition
        log_emission = np.asarray([
            _normal_logpdf(
                frequency_hz - state.mean_hz,
                max(state.variance_hz2, self.config.rf_sigma_floor_hz**2),
            )
            for state in self._rf_states
        ])
        log_forward = np.log(np.maximum(predicted, 1e-300)) + log_emission
        log_evidence = _logsumexp(log_forward)
        log_posterior = log_forward - log_evidence
        posterior = np.exp(log_posterior)

        # Online expected transition counts, not a hard state transition.
        log_xi = (
            np.log(np.maximum(previous[:, None], 1e-300))
            + np.log(np.maximum(transition, 1e-300))
            + log_emission[None, :]
        )
        xi_normalizer = _logsumexp(log_xi.reshape(-1))
        self._transition_counts += np.exp(log_xi - xi_normalizer)
        self._rf_posterior = posterior / posterior.sum()

        for weight, state in zip(self._rf_posterior, self._rf_states):
            if weight <= 1e-12:
                continue
            residual = frequency_hz - state.mean_hz
            rate = float(weight / (state.effective_count + weight))
            state.mean_hz += rate * residual
            state.variance_hz2 = max(
                (1.0 - rate) * state.variance_hz2 + rate * residual * residual,
                self.config.rf_sigma_floor_hz**2,
            )
            state.effective_count += float(weight)
        return log_evidence, self._rf_posterior.copy(), log_posterior

    def _aoa_update(self, aoa_deg: float, elapsed_s: float) -> tuple[float, float]:
        measurement_variance = self.config.aoa_measurement_sigma_deg**2
        if self._aoa_mean_deg is None:
            self._aoa_mean_deg = aoa_deg
            self._aoa_variance_deg2 = measurement_variance
            return 0.0, 0.0
        predicted_variance = (
            (self._aoa_variance_deg2 or 0.0)
            + self.config.aoa_process_variance_deg2_per_s * max(elapsed_s, 0.0)
        )
        innovation = _angular_error_deg(aoa_deg, self._aoa_mean_deg)
        innovation_variance = predicted_variance + measurement_variance
        log_evidence = _normal_logpdf(innovation, innovation_variance)
        gain = predicted_variance / innovation_variance
        self._aoa_mean_deg = (
            self._aoa_mean_deg + gain * innovation + 180.0
        ) % 360.0 - 180.0
        self._aoa_variance_deg2 = (1.0 - gain) * predicted_variance
        return innovation, log_evidence

    def _seed_timing(self, gap_s: float) -> None:
        candidates = []
        for intervals in range(1, self.config.timing_initial_divisors + 1):
            pri_s = gap_s / intervals
            if self.config.timing_min_pri_s <= pri_s <= self.config.timing_max_pri_s:
                candidates.append(pri_s)
        if not candidates:
            candidates = [min(
                self.config.timing_max_pri_s,
                max(self.config.timing_min_pri_s, gap_s),
            )]
        # Retain a deterministic, evenly spread set when a gap yields many divisors.
        if len(candidates) > self.config.timing_max_hypotheses:
            indices = np.linspace(
                0, len(candidates) - 1,
                self.config.timing_max_hypotheses,
                dtype=int,
            )
            candidates = [candidates[index] for index in indices]
        log_weight = -math.log(len(candidates))
        self._timing = [
            _TimingHypothesis(
                pri_s=value,
                variance_s2=max(
                    (self.config.timing_jitter_fraction * value) ** 2,
                    self.config.timing_toa_sigma_s**2,
                ),
                log_weight=log_weight,
            )
            for value in candidates
        ]

    def _timing_update(self, gap_s: float) -> tuple[float, float | None, int | None]:
        if gap_s <= 0.0:
            return 0.0, None, None
        if not self._timing:
            self._seed_timing(gap_s)
            best = max(self._timing, key=lambda item: item.log_weight)
            return 0.0, best.pri_s, max(1, int(round(gap_s / best.pri_s)))

        updated = []
        log_unnormalized = []
        intervals_by_hypothesis = []
        for hypothesis in self._timing:
            predicted_variance = hypothesis.variance_s2 + (
                self.config.timing_process_variance_fraction2_per_s
                * hypothesis.pri_s**2 * gap_s
            )
            intervals = max(1, int(round(gap_s / hypothesis.pri_s)))
            residual = gap_s - intervals * hypothesis.pri_s
            observation_variance = (
                intervals * (self.config.timing_jitter_fraction * hypothesis.pri_s) ** 2
                + self.config.timing_toa_sigma_s**2
            )
            innovation_variance = intervals * intervals * predicted_variance + observation_variance
            log_likelihood = _normal_logpdf(residual, innovation_variance)
            gain = predicted_variance * intervals / innovation_variance
            pri_s = min(
                self.config.timing_max_pri_s,
                max(
                    self.config.timing_min_pri_s,
                    hypothesis.pri_s + gain * residual,
                ),
            )
            variance_s2 = max(
                (1.0 - gain * intervals) * predicted_variance,
                self.config.timing_toa_sigma_s**2,
            )
            updated.append(_TimingHypothesis(pri_s, variance_s2, 0.0))
            log_unnormalized.append(hypothesis.log_weight + log_likelihood)
            intervals_by_hypothesis.append(intervals)

        log_values = np.asarray(log_unnormalized, dtype=float)
        log_evidence = _logsumexp(log_values)
        normalized = log_values - log_evidence
        for hypothesis, log_weight in zip(updated, normalized):
            hypothesis.log_weight = float(log_weight)
        self._timing = updated
        best_index = int(np.argmax(normalized))
        return (
            log_evidence,
            self._timing[best_index].pri_s,
            intervals_by_hypothesis[best_index],
        )

    def process(self, pdw: PDW) -> HMMModel1Update:
        if pdw.frequency_hz <= 0.0:
            raise ValueError("PDW frequency must be positive")
        if self._last_toa_s is not None and pdw.toa_s < self._last_toa_s:
            raise ValueError("PDWs must arrive in non-decreasing TOA order")
        elapsed_s = 0.0 if self._last_toa_s is None else pdw.toa_s - self._last_toa_s
        rf_log_evidence, rf_posterior, rf_log_posterior = self._rf_forward(pdw.frequency_hz)
        aoa_innovation, aoa_log_evidence = self._aoa_update(pdw.aoa_deg, elapsed_s)
        timing_log_evidence, timing_pri, timing_intervals = self._timing_update(elapsed_s)
        self._last_toa_s = pdw.toa_s
        self._updates += 1
        total = rf_log_evidence + aoa_log_evidence + timing_log_evidence
        return HMMModel1Update(
            pdw_id=pdw.pdw_id,
            toa_s=pdw.toa_s,
            rf_posterior=tuple(float(value) for value in rf_posterior),
            rf_log_posterior=tuple(float(value) for value in rf_log_posterior),
            rf_map_state=int(np.argmax(rf_posterior)),
            rf_log_evidence=rf_log_evidence,
            aoa_mean_deg=float(self._aoa_mean_deg),
            aoa_sigma_deg=math.sqrt(float(self._aoa_variance_deg2)),
            aoa_innovation_deg=aoa_innovation,
            aoa_log_evidence=aoa_log_evidence,
            timing_log_evidence=timing_log_evidence,
            timing_map_pri_s=timing_pri,
            timing_map_intervals=timing_intervals,
            timing_hypothesis_count=len(self._timing),
            total_log_evidence=total,
        )

    def snapshot(self) -> dict:
        transition = (
            self._transition_counts / self._transition_counts.sum(axis=1, keepdims=True)
            if len(self._transition_counts) else np.empty((0, 0))
        )
        timing_weights = (
            np.exp([item.log_weight for item in self._timing])
            if self._timing else np.empty(0)
        )
        timing_order = np.argsort(-timing_weights) if len(timing_weights) else []
        return {
            "model": "Rao-Blackwellised emitter HMM Model 1",
            "enabled_features": list(self.enabled_features),
            "excluded_features": list(self.excluded_features),
            "updates": self._updates,
            "rf_states": [
                {
                    "mean_hz": state.mean_hz,
                    "sigma_hz": math.sqrt(state.variance_hz2),
                    "effective_count": state.effective_count,
                    "posterior": float(self._rf_posterior[index]),
                }
                for index, state in enumerate(self._rf_states)
            ],
            "rf_transition_matrix": transition.tolist(),
            "aoa": {
                "mean_deg": self._aoa_mean_deg,
                "sigma_deg": (None if self._aoa_variance_deg2 is None else
                              math.sqrt(self._aoa_variance_deg2)),
            },
            "timing_hypotheses": [
                {
                    "pri_s": self._timing[index].pri_s,
                    "sigma_s": math.sqrt(self._timing[index].variance_s2),
                    "posterior": float(timing_weights[index]),
                }
                for index in timing_order
            ],
            "config": asdict(self.config),
        }
