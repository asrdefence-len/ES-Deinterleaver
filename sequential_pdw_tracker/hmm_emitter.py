"""Rao-Blackwellised single-emitter HMM used by incremental Model 1.

The discrete RF chain is filtered with the HMM forward recursion and online
expected transition counts. AOA is a circular linear-Gaussian Markov state.
TOA is handled by a visibility-aware bank of semi-Markov PRI hypotheses. Each
observed gap retains a posterior over an active pulse interval and a coast gap;
the active branch marginalizes the latent integer interval count. Every input
PDW updates the filter.

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
    timing_max_hypotheses: int = 16
    timing_initial_divisors: int = 128
    timing_jitter_fraction: float = 0.05
    timing_toa_sigma_s: float = 25e-9
    timing_process_variance_fraction2_per_s: float = 1e-4
    timing_transition_pseudocount: float = 0.01
    timing_transition_stay_pseudocount: float = 1.0
    timing_max_active_intervals: int = 16
    timing_detection_alpha_prior: float = 9.0
    timing_detection_beta_prior: float = 1.0
    timing_active_detection_floor: float = 0.50
    timing_active_to_coast_probability: float = 0.02
    timing_coast_to_active_probability: float = 0.80
    timing_coast_mean_gap_s: float = 1e-3
    timing_mode_report_min_support: float = 4.0

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
            self.timing_transition_pseudocount,
            self.timing_transition_stay_pseudocount,
            self.timing_max_active_intervals,
            self.timing_detection_alpha_prior,
            self.timing_detection_beta_prior,
            self.timing_coast_mean_gap_s,
            self.timing_mode_report_min_support,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("HMM configuration values must be positive")
        if self.timing_min_pri_s >= self.timing_max_pri_s:
            raise ValueError("timing_min_pri_s must be below timing_max_pri_s")
        probabilities = (
            self.timing_active_to_coast_probability,
            self.timing_coast_to_active_probability,
            self.timing_active_detection_floor,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("timing visibility transition probabilities must be in (0, 1)")


@dataclass
class _RFState:
    mean_hz: float
    variance_hz2: float
    effective_count: float = 1.0


@dataclass
class _TimingMode:
    pri_s: float
    variance_s2: float
    log_active_weight: float
    log_coast_weight: float
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
    timing_log_evidence: float
    timing_map_pri_s: float | None
    timing_map_intervals: int | None
    timing_expected_intervals: float | None
    timing_active_posterior: float
    timing_coast_posterior: float
    timing_detection_probability: float
    timing_posterior_mean_pri_s: float | None
    timing_posterior_sigma_s: float | None
    timing_mode_posterior: tuple[float, ...]
    timing_map_mode: int | None
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
        self._timing: list[_TimingMode] = []
        self._timing_transition_counts = np.empty((0, 0), dtype=float)
        self._timing_detection_alpha = self.config.timing_detection_alpha_prior
        self._timing_detection_beta = self.config.timing_detection_beta_prior
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
        # Retain a deterministic, logarithmically spread set when a gap yields
        # many divisors.
        if len(candidates) > self.config.timing_max_hypotheses:
            # Logarithmic divisor spacing retains the important low-order
            # candidates (gap/1, gap/2, gap/3, ...) while still representing
            # smaller possible intervals. Linear indices previously skipped
            # gap/3 through gap/8 on a large first gap.
            indices = np.unique(np.rint(np.geomspace(
                1,
                len(candidates),
                self.config.timing_max_hypotheses,
            )).astype(int) - 1)
            candidates = [candidates[index] for index in indices]
        log_weight = -math.log(len(candidates))
        initial_active_probability = 0.9
        self._timing = [
            _TimingMode(
                pri_s=value,
                variance_s2=self.config.timing_toa_sigma_s**2,
                log_active_weight=log_weight + math.log(initial_active_probability),
                log_coast_weight=log_weight + math.log(1.0 - initial_active_probability),
            )
            for value in candidates
        ]
        mode_count = len(self._timing)
        self._timing_transition_counts = np.full(
            (mode_count, mode_count),
            self.config.timing_transition_pseudocount,
            dtype=float,
        )
        self._timing_transition_counts += np.eye(mode_count) * (
            self.config.timing_transition_stay_pseudocount
        )

    @property
    def _timing_detection_probability(self) -> float:
        return self._timing_detection_alpha / (
            self._timing_detection_alpha + self._timing_detection_beta
        )

    def _timing_moments(self) -> tuple[float | None, float | None]:
        if not self._timing:
            return None, None
        weights = np.asarray([
            math.exp(np.logaddexp(item.log_active_weight, item.log_coast_weight))
            for item in self._timing
        ])
        weights /= weights.sum()
        mean = float(sum(weight * item.pri_s for weight, item in zip(weights, self._timing)))
        variance = float(sum(
            weight * (item.variance_s2 + (item.pri_s - mean) ** 2)
            for weight, item in zip(weights, self._timing)
        ))
        return mean, math.sqrt(max(variance, 0.0))

    def _timing_visibility_posterior(self) -> tuple[float, float]:
        if not self._timing:
            return 0.0, 0.0
        active = sum(math.exp(item.log_active_weight) for item in self._timing)
        coast = sum(math.exp(item.log_coast_weight) for item in self._timing)
        total = active + coast
        return active / total, coast / total

    def _timing_active_components(
        self,
        gap_s: float,
        active_source: np.ndarray,
        transition: np.ndarray,
        detection_probability: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward a Markov-additive PRI path for every elapsed interval count.

        A multi-interval gap is a sum of possibly different PRI states. The
        Gaussian mixture is moment-matched separately for each endpoint state;
        it is never approximated as ``n * one PRI``.
        """
        mode_means = np.asarray([item.pri_s for item in self._timing])
        predicted_variances = np.asarray([
            item.variance_s2
            + self.config.timing_process_variance_fraction2_per_s
            * item.pri_s**2 * gap_s
            for item in self._timing
        ])
        duration_variances = (
            predicted_variances
            + (self.config.timing_jitter_fraction * mode_means) ** 2
        )

        path_mass = active_source @ transition
        accumulated_mean = mode_means.copy()
        accumulated_variance = duration_variances.copy()
        components = []
        for intervals in range(1, self.config.timing_max_active_intervals + 1):
            log_interval_prior = (
                math.log(detection_probability)
                + (intervals - 1) * math.log1p(-detection_probability)
            )
            components.append(np.asarray([
                math.log(max(float(path_mass[index]), 1e-300))
                + log_interval_prior
                + _normal_logpdf(
                    gap_s - accumulated_mean[index],
                    accumulated_variance[index] + self.config.timing_toa_sigma_s**2,
                )
                for index in range(len(self._timing))
            ]))
            if intervals == self.config.timing_max_active_intervals:
                break

            incoming = path_mass[:, None] * transition
            next_mass = incoming.sum(axis=0)
            next_mean = np.empty_like(accumulated_mean)
            next_variance = np.empty_like(accumulated_variance)
            for endpoint in range(len(self._timing)):
                weights = incoming[:, endpoint] / max(float(next_mass[endpoint]), 1e-300)
                candidate_means = accumulated_mean + mode_means[endpoint]
                candidate_variances = accumulated_variance + duration_variances[endpoint]
                mean = float(np.dot(weights, candidate_means))
                next_mean[endpoint] = mean
                next_variance[endpoint] = float(np.dot(
                    weights,
                    candidate_variances + (candidate_means - mean) ** 2,
                ))
            path_mass = next_mass
            accumulated_mean = next_mean
            accumulated_variance = next_variance
        return np.asarray(components), duration_variances

    def _timing_update(
        self,
        gap_s: float,
    ) -> tuple[float, float | None, int | None, float | None, int | None]:
        if gap_s <= 0.0:
            return 0.0, None, None, None, None
        if not self._timing:
            self._seed_timing(gap_s)
            best = max(
                self._timing,
                key=lambda item: float(np.logaddexp(
                    item.log_active_weight, item.log_coast_weight,
                )),
            )
            intervals = max(1, int(round(gap_s / best.pri_s)))
            best_index = self._timing.index(best)
            return 0.0, best.pri_s, intervals, float(intervals), best_index

        active_to_coast = self.config.timing_active_to_coast_probability
        coast_to_active = self.config.timing_coast_to_active_probability
        detection_probability = max(
            self._timing_detection_probability,
            self.config.timing_active_detection_floor,
        )
        coast_log_likelihood = (
            -math.log(self.config.timing_coast_mean_gap_s)
            - gap_s / self.config.timing_coast_mean_gap_s
        )
        previous_active = np.exp([
            item.log_active_weight for item in self._timing
        ])
        previous_coast = np.exp([
            item.log_coast_weight for item in self._timing
        ])
        transition = (
            self._timing_transition_counts
            / self._timing_transition_counts.sum(axis=1, keepdims=True)
        )
        active_source = (
            previous_active * (1.0 - active_to_coast)
            + previous_coast * coast_to_active
        )
        # Coast preserves the last timing mode; no unseen mode transition is
        # invented while the emitter is not visible.
        predicted_coast = (
            previous_active * active_to_coast
            + previous_coast * (1.0 - coast_to_active)
        )

        active_components, duration_variances = self._timing_active_components(
            gap_s, active_source, transition, detection_probability,
        )
        log_active_weights = np.asarray([
            _logsumexp(active_components[:, index])
            for index in range(len(self._timing))
        ])
        log_coast_weights = (
            np.log(np.maximum(predicted_coast, 1e-300))
            + coast_log_likelihood
        )
        joint_values = np.r_[log_active_weights, log_coast_weights]
        log_evidence = _logsumexp(joint_values)
        log_active_weights -= log_evidence
        log_coast_weights -= log_evidence
        posterior_active = np.exp(log_active_weights)
        posterior_coast = np.exp(log_coast_weights)

        component_posterior = np.exp(active_components - log_evidence)
        interval_posterior = component_posterior.sum(axis=1)
        active_probability = float(interval_posterior.sum())

        # Direct one-interval gaps provide unambiguous state-transition and
        # mode-centre evidence. Longer gaps contribute to filtering through the
        # additive path likelihood but do not drag a mode toward gap / n.
        mode_means = np.asarray([item.pri_s for item in self._timing])
        direct_log_emission = np.asarray([
            _normal_logpdf(
                gap_s - mode_means[index],
                duration_variances[index] + self.config.timing_toa_sigma_s**2,
            )
            for index in range(len(self._timing))
        ])
        direct_log_xi = (
            np.log(np.maximum(active_source[:, None], 1e-300))
            + np.log(np.maximum(transition, 1e-300))
            + math.log(detection_probability)
            + direct_log_emission[None, :]
        )
        self._timing_transition_counts += np.exp(direct_log_xi - log_evidence)

        # Each timing state is an HMM emission state, not a competing global
        # model. Low-posterior states remain available through the transition
        # prior and therefore can recover when the waveform changes mode.
        for index, hypothesis in enumerate(self._timing):
            responsibility = float(component_posterior[0, index])
            if responsibility > 1e-12:
                residual = gap_s - hypothesis.pri_s
                rate = responsibility / (hypothesis.effective_count + responsibility)
                hypothesis.pri_s += rate * residual
                # Intrinsic jitter belongs to the emission variance above. Do
                # not let an occasional direct-gap ambiguity inflate a state
                # into a broad catch-all timing mode.
                hypothesis.variance_s2 = max(
                    hypothesis.variance_s2,
                    self.config.timing_toa_sigma_s**2,
                )
            hypothesis.log_active_weight = float(log_active_weights[index])
            hypothesis.log_coast_weight = float(log_coast_weights[index])
            hypothesis.effective_count += responsibility

        coast_probability = float(posterior_coast.sum())
        if active_probability > 0.0:
            interval_values = np.arange(1, len(interval_posterior) + 1)
            expected_count = float(np.dot(
                interval_posterior, interval_values,
            ) / active_probability)
            self._timing_detection_alpha += active_probability
            self._timing_detection_beta += active_probability * max(expected_count - 1.0, 0.0)
        else:
            expected_count = None

        best_active_index = int(np.argmax(posterior_active))
        map_intervals = int(np.argmax(interval_posterior)) + 1
        selected_intervals = (
            map_intervals
            if active_probability >= coast_probability else None
        )
        mode_posterior = posterior_active + posterior_coast
        best_index = int(np.argmax(mode_posterior))
        return (
            log_evidence,
            self._timing[best_index].pri_s,
            selected_intervals,
            expected_count,
            best_index,
        )

    def process(self, pdw: PDW) -> HMMModel1Update:
        if pdw.frequency_hz <= 0.0:
            raise ValueError("PDW frequency must be positive")
        if self._last_toa_s is not None and pdw.toa_s < self._last_toa_s:
            raise ValueError("PDWs must arrive in non-decreasing TOA order")
        elapsed_s = 0.0 if self._last_toa_s is None else pdw.toa_s - self._last_toa_s
        rf_log_evidence, rf_posterior, rf_log_posterior = self._rf_forward(pdw.frequency_hz)
        aoa_innovation, aoa_log_evidence = self._aoa_update(pdw.aoa_deg, elapsed_s)
        (
            timing_log_evidence,
            timing_pri,
            timing_intervals,
            timing_expected_intervals,
            timing_map_mode,
        ) = self._timing_update(elapsed_s)
        timing_active, timing_coast = self._timing_visibility_posterior()
        timing_mean, timing_sigma = self._timing_moments()
        timing_mode_posterior = tuple(
            math.exp(np.logaddexp(item.log_active_weight, item.log_coast_weight))
            for item in self._timing
        )
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
            timing_expected_intervals=timing_expected_intervals,
            timing_active_posterior=timing_active,
            timing_coast_posterior=timing_coast,
            timing_detection_probability=self._timing_detection_probability,
            timing_posterior_mean_pri_s=timing_mean,
            timing_posterior_sigma_s=timing_sigma,
            timing_mode_posterior=timing_mode_posterior,
            timing_map_mode=timing_map_mode,
            timing_hypothesis_count=len(self._timing),
            total_log_evidence=total,
        )

    def snapshot(self) -> dict:
        transition = (
            self._transition_counts / self._transition_counts.sum(axis=1, keepdims=True)
            if len(self._transition_counts) else np.empty((0, 0))
        )
        timing_weights = np.asarray([
            math.exp(np.logaddexp(item.log_active_weight, item.log_coast_weight))
            for item in self._timing
        ])
        timing_order = np.argsort(-timing_weights) if len(timing_weights) else []
        timing_transition = (
            self._timing_transition_counts
            / self._timing_transition_counts.sum(axis=1, keepdims=True)
            if len(self._timing_transition_counts) else np.empty((0, 0))
        )
        timing_active, timing_coast = self._timing_visibility_posterior()
        timing_mean, timing_sigma = self._timing_moments()
        learned_timing_indices = [
            index for index, item in enumerate(self._timing)
            if item.effective_count - 1.0 >= self.config.timing_mode_report_min_support
        ]
        learned_timing_indices.sort(key=lambda index: self._timing[index].pri_s)
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
                    "active_posterior": math.exp(
                        self._timing[index].log_active_weight
                    ),
                    "coast_posterior": math.exp(
                        self._timing[index].log_coast_weight
                    ),
                    "effective_count": self._timing[index].effective_count,
                    "state_index": int(index),
                }
                for index in timing_order
            ],
            "timing_transition_matrix": timing_transition.tolist(),
            "timing_learned_modes": [
                {
                    "state_index": int(index),
                    "pri_s": self._timing[index].pri_s,
                    "sigma_s": math.sqrt(self._timing[index].variance_s2),
                    "effective_support": self._timing[index].effective_count - 1.0,
                    "posterior": float(timing_weights[index]),
                }
                for index in learned_timing_indices
            ],
            "timing_visibility": {
                "active_posterior": timing_active,
                "coast_posterior": timing_coast,
            },
            "timing_detection": {
                "probability": self._timing_detection_probability,
                "alpha": self._timing_detection_alpha,
                "beta": self._timing_detection_beta,
            },
            "timing_posterior": {
                "mean_pri_s": timing_mean,
                "sigma_s": timing_sigma,
            },
            "config": asdict(self.config),
        }
