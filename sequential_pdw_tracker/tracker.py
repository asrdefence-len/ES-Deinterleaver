"""Small sequential probabilistic multi-emitter baseline.

The hidden assignment for each incoming PDW is one existing emitter, a new
emitter, or clutter. Each emitter has a Markov chain over learned RF modes and
online probabilistic models for AOA, pulse width and PRI. This is deliberately
a filtering baseline: it makes one causal MAP update and does not yet retain a
multi-hypothesis history or smooth earlier assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .model import AssociationDecision, PDW, TrackSnapshot


def _angular_error_deg(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def _gaussian_compatibility(residual: float, sigma: float) -> float:
    return math.exp(-0.5 * (residual / max(sigma, 1e-15)) ** 2)


@dataclass(frozen=True)
class TrackerConfig:
    max_tracks: int = 64
    max_rf_modes: int = 12
    max_pulse_width_modes: int = 8
    rf_mode_gate_hz: float = 20e6
    rf_sigma_floor_hz: float = 2e6
    rf_hop_probability: float = 0.08
    aoa_sigma_floor_deg: float = 2.0
    pulse_width_log_sigma_floor: float = 0.08
    pulse_width_mode_gate_log: float = 0.25
    pulse_width_hop_probability: float = 0.08
    pri_relative_sigma_floor: float = 0.08
    pri_min_support: int = 3
    max_missed_pulses: int = 20
    missed_pulse_probability: float = 0.85
    # A compatible RF hop must be able to outrank a new-emitter hypothesis.
    # With the previous 0.08 default, an unseen-mode score was at most 0.08
    # before the other likelihoods were applied, so birth always won or tied.
    birth_prior: float = 0.02
    clutter_prior: float = 0.01
    min_assignment_posterior: float = 0.45
    learning_rate_floor: float = 0.02

    def __post_init__(self) -> None:
        if (self.max_tracks < 1 or self.max_rf_modes < 1 or self.max_pulse_width_modes < 1
                or self.pri_min_support < 1 or self.max_missed_pulses < 1):
            raise ValueError("track, mode, PRI-support and missed-pulse limits must be positive")
        probabilities = (
            self.rf_hop_probability,
            self.pulse_width_hop_probability,
            self.missed_pulse_probability,
            self.birth_prior,
            self.clutter_prior,
            self.min_assignment_posterior,
            self.learning_rate_floor,
        )
        if any(not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("probability and learning-rate settings must be in (0, 1]")


@dataclass
class _RFMode:
    mean_hz: float
    variance_hz2: float
    count: int = 1


@dataclass
class _PulseWidthMode:
    log_mean: float
    log_variance: float
    count: int = 1


@dataclass
class _Track:
    track_id: str
    first_toa_s: float
    last_toa_s: float
    aoa_deg: float
    aoa_variance_deg2: float
    pulse_width_modes: list[_PulseWidthMode]
    pulse_width_transition_counts: np.ndarray
    rf_modes: list[_RFMode]
    transition_counts: np.ndarray
    last_pulse_width_mode: int = 0
    last_rf_mode: int = 0
    pulse_count: int = 1
    pri_s: float | None = None
    pri_variance_s2: float | None = None
    pri_support: int = 0


class SequentialPDWTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_track_number = 1

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def _rf_likelihood(self, track: _Track, frequency_hz: float) -> tuple[float, int | None]:
        likelihoods = []
        transition_row = track.transition_counts[track.last_rf_mode]
        transition_probability = transition_row / transition_row.sum()
        for index, mode in enumerate(track.rf_modes):
            sigma = math.sqrt(max(mode.variance_hz2, self.config.rf_sigma_floor_hz**2))
            likelihoods.append(
                float(transition_probability[index])
                * _gaussian_compatibility(frequency_hz - mode.mean_hz, sigma)
            )
        best = int(np.argmax(likelihoods))
        if abs(frequency_hz - track.rf_modes[best].mean_hz) <= self.config.rf_mode_gate_hz:
            return max(likelihoods[best], 1e-12), best
        # An unseen frequency can be a hop from the same emitter. This explicit
        # probability prevents RF alone from forcing a new identity.
        # This branch represents the composite event "some previously unseen
        # RF mode", not one particular member of an enumerated mode set. Its
        # probability mass must not shrink as known modes are added; doing so
        # made a fifth E46 mode unable to beat the fixed birth prior.
        return self.config.rf_hop_probability, None

    def _timing_likelihood(self, track: _Track, toa_s: float) -> float:
        gap = toa_s - track.last_toa_s
        if gap < 0.0:
            return 0.0
        # One inter-arrival gap is not a reliable PRI estimate in an interleaved
        # stream: it may span missed pulses or a temporary association. Keep
        # timing neutral until several assigned gaps support the estimate.
        if (track.pri_s is None or track.pri_support < self.config.pri_min_support
                or gap == 0.0):
            return 1.0
        intervals = min(
            self.config.max_missed_pulses,
            max(1, int(round(gap / track.pri_s))),
        )
        residual = gap - intervals * track.pri_s
        sigma = max(
            math.sqrt(max(track.pri_variance_s2 or 0.0, 0.0)) * math.sqrt(intervals),
            self.config.pri_relative_sigma_floor * track.pri_s * math.sqrt(intervals),
        )
        return _gaussian_compatibility(residual, sigma) * self.config.missed_pulse_probability ** (intervals - 1)

    def _pulse_width_likelihood(
        self,
        track: _Track,
        pulse_width_s: float,
    ) -> tuple[float, int | None]:
        log_width = math.log(pulse_width_s)
        likelihoods = []
        for index, mode in enumerate(track.pulse_width_modes):
            sigma = math.sqrt(max(
                mode.log_variance,
                self.config.pulse_width_log_sigma_floor**2,
            ))
            likelihoods.append(
                _gaussian_compatibility(log_width - mode.log_mean, sigma)
            )
        best = int(np.argmax(likelihoods))
        if abs(log_width - track.pulse_width_modes[best].log_mean) <= self.config.pulse_width_mode_gate_log:
            return max(likelihoods[best], 1e-12), best
        return self.config.pulse_width_hop_probability, None

    def _track_likelihood(
        self,
        track: _Track,
        pdw: PDW,
    ) -> tuple[float, dict[str, float], tuple[int | None, int | None]]:
        aoa_sigma = math.sqrt(max(track.aoa_variance_deg2, self.config.aoa_sigma_floor_deg**2))
        aoa = _gaussian_compatibility(_angular_error_deg(pdw.aoa_deg, track.aoa_deg), aoa_sigma)
        pulse_width, pulse_width_mode = self._pulse_width_likelihood(track, pdw.pulse_width_s)
        frequency, rf_mode = self._rf_likelihood(track, pdw.frequency_hz)
        timing = self._timing_likelihood(track, pdw.toa_s)
        components = {"aoa": aoa, "pulse_width": pulse_width, "frequency": frequency, "timing": timing}
        return math.prod(components.values()), components, (rf_mode, pulse_width_mode)

    def process(self, pdw: PDW) -> AssociationDecision:
        if pdw.pulse_width_s <= 0.0 or pdw.frequency_hz <= 0.0:
            raise ValueError("PDW frequency and pulse width must be positive")
        scores: dict[str, float] = {"new": self.config.birth_prior, "clutter": self.config.clutter_prior}
        components: dict[str, dict[str, float]] = {}
        modes: dict[str, tuple[int | None, int | None]] = {}
        for track in self._tracks:
            score, detail, mode = self._track_likelihood(track, pdw)
            scores[track.track_id] = max(score, 1e-300)
            components[track.track_id] = detail
            modes[track.track_id] = mode
        total = sum(scores.values())
        posterior = {name: score / total for name, score in scores.items()}
        winner = max(posterior, key=posterior.get)
        outcome = winner
        track_id: str | None = None
        if winner not in ("new", "clutter") and posterior[winner] >= self.config.min_assignment_posterior:
            track = next(track for track in self._tracks if track.track_id == winner)
            rf_mode, pulse_width_mode = modes[winner]
            self._update(track, pdw, rf_mode, pulse_width_mode)
            outcome, track_id = "assigned", winner
        elif winner == "new" and len(self._tracks) < self.config.max_tracks:
            track_id = self._create_track(pdw)
            outcome = "birth"
        else:
            outcome = "clutter"
        return AssociationDecision(
            pdw_id=pdw.pdw_id,
            toa_s=pdw.toa_s,
            outcome=outcome,
            track_id=track_id,
            posterior=posterior,
            component_likelihoods=components,
        )

    def _create_track(self, pdw: PDW) -> str:
        track_id = f"E{self._next_track_number}"
        self._next_track_number += 1
        self._tracks.append(_Track(
            track_id=track_id,
            first_toa_s=pdw.toa_s,
            last_toa_s=pdw.toa_s,
            aoa_deg=pdw.aoa_deg,
            aoa_variance_deg2=self.config.aoa_sigma_floor_deg**2,
            pulse_width_modes=[_PulseWidthMode(
                math.log(pdw.pulse_width_s),
                self.config.pulse_width_log_sigma_floor**2,
            )],
            pulse_width_transition_counts=np.ones((1, 1), dtype=float),
            rf_modes=[_RFMode(pdw.frequency_hz, self.config.rf_sigma_floor_hz**2)],
            transition_counts=np.ones((1, 1), dtype=float),
        ))
        return track_id

    def _update(
        self,
        track: _Track,
        pdw: PDW,
        mode_index: int | None,
        pulse_width_mode_index: int | None,
    ) -> None:
        gap = pdw.toa_s - track.last_toa_s
        alpha = max(self.config.learning_rate_floor, 1.0 / (track.pulse_count + 1))
        aoa_residual = _angular_error_deg(pdw.aoa_deg, track.aoa_deg)
        track.aoa_deg = (track.aoa_deg + alpha * aoa_residual + 180.0) % 360.0 - 180.0
        track.aoa_variance_deg2 = (1.0 - alpha) * track.aoa_variance_deg2 + alpha * aoa_residual**2

        if (pulse_width_mode_index is None
                and len(track.pulse_width_modes) < self.config.max_pulse_width_modes):
            pulse_width_mode_index = len(track.pulse_width_modes)
            track.pulse_width_modes.append(_PulseWidthMode(
                math.log(pdw.pulse_width_s),
                self.config.pulse_width_log_sigma_floor**2,
            ))
            expanded = np.ones((pulse_width_mode_index + 1, pulse_width_mode_index + 1), dtype=float)
            expanded[:pulse_width_mode_index, :pulse_width_mode_index] = track.pulse_width_transition_counts
            track.pulse_width_transition_counts = expanded
        elif pulse_width_mode_index is None:
            log_width = math.log(pdw.pulse_width_s)
            pulse_width_mode_index = min(
                range(len(track.pulse_width_modes)),
                key=lambda i: abs(log_width - track.pulse_width_modes[i].log_mean),
            )
        pulse_width_mode = track.pulse_width_modes[pulse_width_mode_index]
        pw_alpha = max(
            self.config.learning_rate_floor,
            1.0 / (pulse_width_mode.count + 1),
        )
        pw_residual = math.log(pdw.pulse_width_s) - pulse_width_mode.log_mean
        pulse_width_mode.log_mean += pw_alpha * pw_residual
        pulse_width_mode.log_variance = (
            (1.0 - pw_alpha) * pulse_width_mode.log_variance
            + pw_alpha * pw_residual**2
        )
        pulse_width_mode.count += 1
        track.pulse_width_transition_counts[
            track.last_pulse_width_mode, pulse_width_mode_index
        ] += 1.0
        track.last_pulse_width_mode = pulse_width_mode_index

        if mode_index is None and len(track.rf_modes) < self.config.max_rf_modes:
            mode_index = len(track.rf_modes)
            track.rf_modes.append(_RFMode(pdw.frequency_hz, self.config.rf_sigma_floor_hz**2))
            expanded = np.ones((mode_index + 1, mode_index + 1), dtype=float)
            expanded[:mode_index, :mode_index] = track.transition_counts
            track.transition_counts = expanded
        elif mode_index is None:
            mode_index = min(range(len(track.rf_modes)), key=lambda i: abs(pdw.frequency_hz - track.rf_modes[i].mean_hz))
        mode = track.rf_modes[mode_index]
        rf_alpha = max(self.config.learning_rate_floor, 1.0 / (mode.count + 1))
        rf_residual = pdw.frequency_hz - mode.mean_hz
        mode.mean_hz += rf_alpha * rf_residual
        mode.variance_hz2 = (1.0 - rf_alpha) * mode.variance_hz2 + rf_alpha * rf_residual**2
        mode.count += 1
        track.transition_counts[track.last_rf_mode, mode_index] += 1.0
        track.last_rf_mode = mode_index

        if gap > 0.0:
            if track.pri_s is None:
                track.pri_s = gap
                track.pri_variance_s2 = (self.config.pri_relative_sigma_floor * gap) ** 2
                track.pri_support = 1
            else:
                intervals = min(self.config.max_missed_pulses, max(1, int(round(gap / track.pri_s))))
                observed_pri = gap / intervals
                pri_residual = observed_pri - track.pri_s
                track.pri_s += alpha * pri_residual
                track.pri_variance_s2 = (1.0 - alpha) * (track.pri_variance_s2 or 0.0) + alpha * pri_residual**2
                track.pri_support += 1
        track.last_toa_s = pdw.toa_s
        track.pulse_count += 1

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        snapshots = []
        for track in self._tracks:
            pulse_width_mode = track.pulse_width_modes[track.last_pulse_width_mode]
            snapshots.append(TrackSnapshot(
                track_id=track.track_id,
                pulse_count=track.pulse_count,
                first_toa_s=track.first_toa_s,
                last_toa_s=track.last_toa_s,
                aoa_deg=track.aoa_deg,
                aoa_sigma_deg=math.sqrt(max(track.aoa_variance_deg2, 0.0)),
                pulse_width_s=math.exp(pulse_width_mode.log_mean),
                pulse_width_log_sigma=math.sqrt(max(pulse_width_mode.log_variance, 0.0)),
                pulse_width_modes_s=tuple(
                    math.exp(mode.log_mean) for mode in track.pulse_width_modes
                ),
                last_pulse_width_mode=track.last_pulse_width_mode,
                pri_s=track.pri_s,
                pri_sigma_s=(None if track.pri_variance_s2 is None else
                             math.sqrt(max(track.pri_variance_s2, 0.0))),
                pri_support=track.pri_support,
                rf_modes_hz=tuple(mode.mean_hz for mode in track.rf_modes),
                last_rf_mode=track.last_rf_mode,
            ))
        return tuple(snapshots)
