"""Single-emitter HMM Model 2: Model 1 plus pulse-width hidden states.

The implementation subclasses Model 1 so RF, AOA and the zero-feedback timing
shadow execute through the identical code path.  Pulse width is represented by
a separate discrete HMM with Gaussian emissions in log-width space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .hmm_emitter import (
    HMMEmitterModel1,
    HMMModel1Config,
    HMMModel1Update,
    _logsumexp,
    _normal_logpdf,
)
from .model import PDW


@dataclass(frozen=True)
class HMMModel2Config:
    model1: HMMModel1Config = field(default_factory=HMMModel1Config)
    max_pulse_width_states: int = 8
    pulse_width_new_state_gate_log: float = 0.25
    pulse_width_log_sigma_floor: float = 0.02
    transition_pseudocount: float = 1.0
    new_state_prior: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.max_pulse_width_states,
            self.pulse_width_new_state_gate_log,
            self.pulse_width_log_sigma_floor,
            self.transition_pseudocount,
            self.new_state_prior,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Model 2 pulse-width configuration must be positive")
        if self.new_state_prior >= 1.0:
            raise ValueError("new_state_prior must be below one")


@dataclass
class _PulseWidthState:
    log_mean: float
    log_variance: float
    effective_count: float = 1.0


@dataclass(frozen=True)
class HMMModel2Update(HMMModel1Update):
    pulse_width_s: float
    pulse_width_posterior: tuple[float, ...]
    pulse_width_log_posterior: tuple[float, ...]
    pulse_width_map_state: int
    pulse_width_map_mean_s: float
    pulse_width_log_evidence: float
    pulse_width_state_created: bool
    model1_log_evidence: float


@dataclass(frozen=True)
class Model2PredictiveEvidence:
    rf_log_evidence: float
    aoa_log_evidence: float
    pulse_width_log_evidence: float
    total_log_evidence: float


@dataclass(frozen=True)
class Model2FeatureNovelty:
    rf_distance_hz: float
    rf_normalized_distance: float
    rf_known: bool
    pulse_width_log_distance: float
    pulse_width_normalized_distance: float
    pulse_width_known: bool


class HMMEmitterModel2(HMMEmitterModel1):
    """Model 1 state filter extended by a pulse-width HMM."""

    enabled_features = ("Frequency", "ToA", "AoA", "PulseWidth")
    active_evidence_features = ("Frequency", "AoA", "PulseWidth")
    shadow_features = ("ToA",)
    excluded_features = ("Amplitude",)

    def __init__(self, config: HMMModel2Config | None = None):
        self.model2_config = config or HMMModel2Config()
        super().__init__(self.model2_config.model1)
        self._pulse_width_states: list[_PulseWidthState] = []
        self._pulse_width_posterior = np.empty(0, dtype=float)
        self._pulse_width_transition_counts = np.empty((0, 0), dtype=float)

    def _add_pulse_width_state(self, log_width: float) -> None:
        state_count = len(self._pulse_width_states)
        floor = self.model2_config.pulse_width_log_sigma_floor
        self._pulse_width_states.append(_PulseWidthState(
            log_mean=log_width,
            log_variance=floor**2,
        ))
        expanded = np.full(
            (state_count + 1, state_count + 1),
            self.model2_config.transition_pseudocount,
            dtype=float,
        )
        if state_count:
            expanded[:state_count, :state_count] = self._pulse_width_transition_counts
            self._pulse_width_posterior *= 1.0 - self.model2_config.new_state_prior
            self._pulse_width_posterior = np.r_[
                self._pulse_width_posterior,
                self.model2_config.new_state_prior,
            ]
        else:
            self._pulse_width_posterior = np.ones(1, dtype=float)
        self._pulse_width_posterior /= self._pulse_width_posterior.sum()
        self._pulse_width_transition_counts = expanded

    def _pulse_width_forward(
        self,
        pulse_width_s: float,
    ) -> tuple[float, np.ndarray, np.ndarray, bool]:
        log_width = math.log(pulse_width_s)
        if not self._pulse_width_states:
            self._add_pulse_width_state(log_width)
            return (
                0.0,
                self._pulse_width_posterior.copy(),
                np.zeros(1, dtype=float),
                True,
            )

        nearest = min(
            abs(log_width - state.log_mean) for state in self._pulse_width_states
        )
        state_created = False
        if (
            nearest > self.model2_config.pulse_width_new_state_gate_log
            and len(self._pulse_width_states)
            < self.model2_config.max_pulse_width_states
        ):
            self._add_pulse_width_state(log_width)
            state_created = True

        previous = self._pulse_width_posterior.copy()
        transition = (
            self._pulse_width_transition_counts
            / self._pulse_width_transition_counts.sum(axis=1, keepdims=True)
        )
        predicted = previous @ transition
        floor_variance = self.model2_config.pulse_width_log_sigma_floor**2
        log_emission = np.asarray([
            _normal_logpdf(
                log_width - state.log_mean,
                max(state.log_variance, floor_variance),
            )
            for state in self._pulse_width_states
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
        self._pulse_width_transition_counts += np.exp(log_xi - xi_normalizer)
        self._pulse_width_posterior = posterior / posterior.sum()

        for weight, state in zip(
            self._pulse_width_posterior, self._pulse_width_states,
        ):
            if weight <= 1e-12:
                continue
            residual = log_width - state.log_mean
            rate = float(weight / (state.effective_count + weight))
            state.log_mean += rate * residual
            state.log_variance = max(
                (1.0 - rate) * state.log_variance + rate * residual * residual,
                floor_variance,
            )
            state.effective_count += float(weight)
        return (
            log_evidence,
            self._pulse_width_posterior.copy(),
            log_posterior,
            state_created,
        )

    def process(self, pdw: PDW) -> HMMModel2Update:
        if pdw.pulse_width_s <= 0.0:
            raise ValueError("PDW pulse width must be positive")
        model1_update = super().process(pdw)
        (
            pulse_width_log_evidence,
            pulse_width_posterior,
            pulse_width_log_posterior,
            pulse_width_state_created,
        ) = self._pulse_width_forward(pdw.pulse_width_s)
        map_state = int(np.argmax(pulse_width_posterior))

        values = asdict(model1_update)
        values["total_log_evidence"] = (
            model1_update.total_log_evidence + pulse_width_log_evidence
        )
        return HMMModel2Update(
            **values,
            pulse_width_s=pdw.pulse_width_s,
            pulse_width_posterior=tuple(
                float(value) for value in pulse_width_posterior
            ),
            pulse_width_log_posterior=tuple(
                float(value) for value in pulse_width_log_posterior
            ),
            pulse_width_map_state=map_state,
            pulse_width_map_mean_s=math.exp(
                self._pulse_width_states[map_state].log_mean
            ),
            pulse_width_log_evidence=pulse_width_log_evidence,
            pulse_width_state_created=pulse_width_state_created,
            model1_log_evidence=model1_update.total_log_evidence,
        )

    def predictive_evidence(self, pdw: PDW) -> Model2PredictiveEvidence:
        """Score a candidate without changing state or creating new modes."""
        if pdw.frequency_hz <= 0.0 or pdw.pulse_width_s <= 0.0:
            raise ValueError("PDW frequency and pulse width must be positive")
        if not self._rf_states or not self._pulse_width_states:
            raise RuntimeError("Model 2 must be trained before predictive scoring")
        if self._last_toa_s is not None and pdw.toa_s < self._last_toa_s:
            raise ValueError("candidate TOA precedes the last accepted pulse")

        rf_transition = (
            self._transition_counts
            / self._transition_counts.sum(axis=1, keepdims=True)
        )
        rf_predicted = self._rf_posterior @ rf_transition
        rf_log_emission = np.asarray([
            _normal_logpdf(
                pdw.frequency_hz - state.mean_hz,
                max(state.variance_hz2, self.config.rf_sigma_floor_hz**2),
            )
            for state in self._rf_states
        ])
        rf_log_evidence = _logsumexp(
            np.log(np.maximum(rf_predicted, 1e-300)) + rf_log_emission
        )

        aoa_log_evidence = self.predictive_aoa_log_evidence(pdw)

        pw_transition = (
            self._pulse_width_transition_counts
            / self._pulse_width_transition_counts.sum(axis=1, keepdims=True)
        )
        pw_predicted = self._pulse_width_posterior @ pw_transition
        log_width = math.log(pdw.pulse_width_s)
        floor_variance = self.model2_config.pulse_width_log_sigma_floor**2
        pw_log_emission = np.asarray([
            _normal_logpdf(
                log_width - state.log_mean,
                max(state.log_variance, floor_variance),
            )
            for state in self._pulse_width_states
        ])
        pulse_width_log_evidence = _logsumexp(
            np.log(np.maximum(pw_predicted, 1e-300)) + pw_log_emission
        )
        total = rf_log_evidence + aoa_log_evidence + pulse_width_log_evidence
        return Model2PredictiveEvidence(
            rf_log_evidence=rf_log_evidence,
            aoa_log_evidence=aoa_log_evidence,
            pulse_width_log_evidence=pulse_width_log_evidence,
            total_log_evidence=total,
        )

    def predictive_aoa_log_evidence(self, pdw: PDW) -> float:
        """Return the non-mutating continuous AOA predictive log density."""
        if self._aoa_mean_deg is None:
            raise RuntimeError("Model 2 must be trained before predictive scoring")
        if self._last_toa_s is not None and pdw.toa_s < self._last_toa_s:
            raise ValueError("candidate TOA precedes the last accepted pulse")
        elapsed_s = (
            0.0 if self._last_toa_s is None else pdw.toa_s - self._last_toa_s
        )
        predicted_aoa_variance = (
            float(self._aoa_variance_deg2 or 0.0)
            + self.config.aoa_process_variance_deg2_per_s * elapsed_s
        )
        aoa_innovation = (
            pdw.aoa_deg - float(self._aoa_mean_deg) + 180.0
        ) % 360.0 - 180.0
        return _normal_logpdf(
            aoa_innovation,
            predicted_aoa_variance
            + self.config.aoa_measurement_sigma_deg**2,
        )

    def predictive_rf_pw_log_evidence_upper_bound(self) -> float:
        """Bound the best possible RF+PW density under the current states."""
        if not self._rf_states or not self._pulse_width_states:
            raise RuntimeError("Model 2 must be trained before predictive scoring")
        rf_floor = self.config.rf_sigma_floor_hz**2
        pw_floor = self.model2_config.pulse_width_log_sigma_floor**2
        rf_upper = max(
            _normal_logpdf(0.0, max(state.variance_hz2, rf_floor))
            for state in self._rf_states
        )
        pw_upper = max(
            _normal_logpdf(0.0, max(state.log_variance, pw_floor))
            for state in self._pulse_width_states
        )
        return rf_upper + pw_upper

    def feature_novelty(self, pdw: PDW) -> Model2FeatureNovelty:
        """Measure RF/PW distance to learned states without changing them."""
        if not self._rf_states or not self._pulse_width_states:
            raise RuntimeError("Model 2 must be trained before novelty scoring")
        rf_distance = min(
            abs(pdw.frequency_hz - state.mean_hz) for state in self._rf_states
        )
        log_width = math.log(pdw.pulse_width_s)
        pw_distance = min(
            abs(log_width - state.log_mean)
            for state in self._pulse_width_states
        )
        rf_gate = self.config.rf_new_state_gate_hz
        pw_gate = self.model2_config.pulse_width_new_state_gate_log
        return Model2FeatureNovelty(
            rf_distance_hz=rf_distance,
            rf_normalized_distance=rf_distance / rf_gate,
            rf_known=rf_distance <= rf_gate,
            pulse_width_log_distance=pw_distance,
            pulse_width_normalized_distance=pw_distance / pw_gate,
            pulse_width_known=pw_distance <= pw_gate,
        )

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        transition = (
            self._pulse_width_transition_counts
            / self._pulse_width_transition_counts.sum(axis=1, keepdims=True)
            if len(self._pulse_width_transition_counts) else np.empty((0, 0))
        )
        snapshot.update({
            "model": "Emitter HMM Model 2 with pulse-width hidden states",
            "pulse_width_emission_space": "natural logarithm of seconds",
            "pulse_width_states": [
                {
                    "mean_s": math.exp(state.log_mean),
                    "log_sigma": math.sqrt(state.log_variance),
                    "effective_count": state.effective_count,
                    "posterior": float(self._pulse_width_posterior[index]),
                }
                for index, state in enumerate(self._pulse_width_states)
            ],
            "pulse_width_transition_matrix": transition.tolist(),
            "model2_config": asdict(self.model2_config),
        })
        return snapshot
