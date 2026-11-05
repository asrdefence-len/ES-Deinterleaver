"""Single-emitter HMM Model 1 with timing isolated in shadow mode.

RF uses a proper discrete HMM forward recursion. AOA uses a circular Gaussian
Markov filter. TOA orders the sequence and controls elapsed-time prediction;
the detected pulse train is also observed by an explicitly non-influential
shadow timing model. Pulse width and amplitude are excluded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .detected_train_timing import (
    DetectedTrainTimingConfig,
    DetectedTrainTimingEstimator,
)
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
    shadow_timing: DetectedTrainTimingConfig = field(
        default_factory=DetectedTrainTimingConfig,
    )

    def __post_init__(self) -> None:
        positive = (
            self.max_rf_states,
            self.rf_new_state_gate_hz,
            self.rf_sigma_floor_hz,
            self.transition_pseudocount,
            self.new_state_prior,
            self.aoa_measurement_sigma_deg,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("HMM configuration values must be positive")


@dataclass
class _RFState:
    mean_hz: float
    variance_hz2: float
    effective_count: float = 1.0


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
    shadow_gap_s: float | None
    shadow_timing_log_predictive_density: float
    shadow_active_posterior: float
    shadow_coast_posterior: float
    shadow_timing_ready: bool
    timing_feedback_weight: float
    total_log_evidence: float


class HMMEmitterModel1:
    """Forward state filter for one already selected emitter."""

    enabled_features = ("Frequency", "ToA", "AoA")
    active_evidence_features = ("Frequency", "AoA")
    shadow_features = ("ToA",)
    excluded_features = ("PulseWidth", "Amplitude")

    def __init__(self, config: HMMModel1Config | None = None):
        self.config = config or HMMModel1Config()
        self._rf_states: list[_RFState] = []
        self._rf_posterior = np.empty(0, dtype=float)
        self._transition_counts = np.empty((0, 0), dtype=float)
        self._aoa_mean_deg: float | None = None
        self._aoa_variance_deg2: float | None = None
        self._shadow_timing = DetectedTrainTimingEstimator(self.config.shadow_timing)
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

    def process(self, pdw: PDW) -> HMMModel1Update:
        if pdw.frequency_hz <= 0.0:
            raise ValueError("PDW frequency must be positive")
        if self._last_toa_s is not None and pdw.toa_s < self._last_toa_s:
            raise ValueError("PDWs must arrive in non-decreasing TOA order")
        elapsed_s = 0.0 if self._last_toa_s is None else pdw.toa_s - self._last_toa_s
        rf_log_evidence, rf_posterior, rf_log_posterior = self._rf_forward(pdw.frequency_hz)
        aoa_innovation, aoa_log_evidence = self._aoa_update(pdw.aoa_deg, elapsed_s)
        shadow = self._shadow_timing.process(pdw.toa_s)
        self._last_toa_s = pdw.toa_s
        self._updates += 1

        total = rf_log_evidence + aoa_log_evidence
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
            shadow_gap_s=shadow.gap_s,
            shadow_timing_log_predictive_density=shadow.log_predictive_density,
            shadow_active_posterior=shadow.active_posterior,
            shadow_coast_posterior=shadow.coast_posterior,
            shadow_timing_ready=shadow.ready,
            timing_feedback_weight=shadow.feedback_weight,
            total_log_evidence=total,
        )

    def snapshot(self) -> dict:
        transition = (
            self._transition_counts / self._transition_counts.sum(axis=1, keepdims=True)
            if len(self._transition_counts) else np.empty((0, 0))
        )
        return {
            "model": "Emitter HMM Model 1 with detected-train timing shadow",
            "enabled_features": list(self.enabled_features),
            "active_evidence_features": list(self.active_evidence_features),
            "shadow_features": list(self.shadow_features),
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
            "detected_train_timing_shadow": self._shadow_timing.snapshot(),
            "config": asdict(self.config),
        }
