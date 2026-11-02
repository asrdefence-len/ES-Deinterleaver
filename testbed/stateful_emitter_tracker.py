"""Stateful multi-emitter tracker with covariance-derived association gates.

This is the first operational tracker baseline.  It does not use truth labels.
Association confidence is a normalized model score against other tracks and a
clutter alternative; it is not calibrated until the noise/clutter models are.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np


def _logit(probability):
    return math.log(probability / (1. - probability))


def _logistic(value):
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1. + exp_value)


def _angle_difference(value, reference):
    return (value - reference + 180.) % 360. - 180.


@dataclass
class RFModeFilter:
    mean_hz: float
    variance_hz2: float
    updates: int = 0


@dataclass
class EmitterFilterState:
    emitter_id: str
    last_predict_s: float
    last_update_s: float
    aoa_deg: float
    aoa_variance_deg2: float
    pulse_width_s: float
    pulse_width_variance_s2: float
    rf_modes: list[RFModeFilter]
    existence_log_odds: float
    last_confidence_s: float
    pri_s: float | None = None
    pri_variance_s2: float | None = None
    phase_epoch_s: float | None = None
    timing_enabled: bool = False
    confirmed_updates: int = 0
    associated_pdws: list[int] = field(default_factory=list)
    last_innovation: dict | None = None

    @property
    def existence_confidence(self):
        return _logistic(self.existence_log_odds)

    @property
    def status(self):
        if self.existence_confidence >= .8:
            return "confirmed"
        if self.existence_confidence >= .3:
            return "coasting" if self.confirmed_updates else "tentative"
        return "lost"


@dataclass(frozen=True)
class TrackGate:
    emitter_id: str
    aoa_center_deg: float
    aoa_half_width_deg: float
    pulse_width_center_s: float
    pulse_width_half_width_s: float
    rf_modes_hz: tuple[float, ...]
    rf_half_widths_hz: tuple[float, ...]
    pri_s: float | None
    timing_half_width_s: float | None
    existence_confidence: float


@dataclass(frozen=True)
class TrackAssociation:
    emitter_id: str
    confidence: float
    normalized_innovation_squared: float
    aoa_innovation_deg: float
    pulse_width_innovation_s: float
    rf_innovation_hz: float
    timing_innovation_s: float | None


@dataclass(frozen=True)
class TrackerDecision:
    pdw_id: int
    status: str
    selected_emitter_id: str | None
    associations: tuple[TrackAssociation, ...]
    confirmed_pdw_ids: tuple[int, ...] = ()


class StatefulEmitterTracker:
    """Predict, gate, associate, confirm, update and coast emitter tracks."""

    def __init__(self, *, gate_sigma=3., min_confirm_pulses=5,
                 confirmation_span_s=.010, confirmation_confidence=.70,
                 ambiguity_margin=.10, clutter_weight=.02,
                 aoa_measurement_sigma_deg=.6, pw_measurement_sigma_s=.08e-6,
                 rf_measurement_sigma_hz=8e6, timing_measurement_sigma_s=5e-6,
                 aoa_process_variance_deg2_s=.04,
                 pw_process_variance_s2_s=(.01e-6) ** 2,
                 rf_process_variance_hz2_s=(1e6) ** 2,
                 pri_process_variance_s2_s=(.2e-6) ** 2,
                 hit_log_odds=.8, miss_log_odds_s=.7):
        if (gate_sigma <= 0 or min_confirm_pulses < 2
                or confirmation_span_s <= 0 or not 0 < confirmation_confidence < 1
                or ambiguity_margin < 0 or clutter_weight <= 0):
            raise ValueError("Invalid tracker configuration")
        self.gate_sigma = gate_sigma
        self.min_confirm_pulses = min_confirm_pulses
        self.confirmation_span_s = confirmation_span_s
        self.confirmation_confidence = confirmation_confidence
        self.ambiguity_margin = ambiguity_margin
        self.clutter_weight = clutter_weight
        self.aoa_r = aoa_measurement_sigma_deg ** 2
        self.pw_r = pw_measurement_sigma_s ** 2
        self.rf_r = rf_measurement_sigma_hz ** 2
        self.timing_r = timing_measurement_sigma_s ** 2
        self.aoa_q = aoa_process_variance_deg2_s
        self.pw_q = pw_process_variance_s2_s
        self.rf_q = rf_process_variance_hz2_s
        self.pri_q = pri_process_variance_s2_s
        self.hit_log_odds = hit_log_odds
        self.miss_log_odds_s = miss_log_odds_s
        self.tracks = {}
        self._provisional = {}
        self._seen = set()
        self._last_toa_s = float("-inf")

    def add_track(self, emitter_id, *, time_s, aoa_deg, pulse_width_s,
                  frequency_modes_hz, pri_s=None, phase_epoch_s=None,
                  aoa_sigma_deg=1.5, pulse_width_sigma_s=.15e-6,
                  rf_sigma_hz=15e6, pri_sigma_s=2e-6,
                  existence_confidence=.7, timing_enabled=False):
        if emitter_id in self.tracks or not frequency_modes_hz:
            raise ValueError("Duplicate track or missing RF modes")
        if timing_enabled and (pri_s is None or phase_epoch_s is None):
            raise ValueError("Timing requires PRI and phase epoch")
        self.tracks[emitter_id] = EmitterFilterState(
            emitter_id, time_s, time_s, aoa_deg, aoa_sigma_deg ** 2,
            pulse_width_s, pulse_width_sigma_s ** 2,
            [RFModeFilter(float(mode), rf_sigma_hz ** 2)
             for mode in frequency_modes_hz],
            _logit(existence_confidence), time_s, pri_s,
            None if pri_s is None else pri_sigma_s ** 2,
            phase_epoch_s, timing_enabled)
        self._provisional[emitter_id] = []
        return emitter_id

    def _predict_track(self, track, time_s):
        dt = time_s - track.last_predict_s
        if dt < 0:
            raise ValueError("Tracker time cannot move backwards")
        if dt == 0:
            return
        track.aoa_variance_deg2 += self.aoa_q * dt
        track.pulse_width_variance_s2 += self.pw_q * dt
        for mode in track.rf_modes:
            mode.variance_hz2 += self.rf_q * dt
        if track.pri_variance_s2 is not None:
            track.pri_variance_s2 += self.pri_q * dt
        track.last_predict_s = time_s

    def predict_to(self, time_s):
        for track in self.tracks.values():
            self._predict_track(track, time_s)

    def gate(self, emitter_id, time_s=None):
        track = self.tracks[emitter_id]
        if time_s is not None:
            self._predict_track(track, time_s)
        return TrackGate(
            emitter_id, track.aoa_deg,
            self.gate_sigma * math.sqrt(track.aoa_variance_deg2 + self.aoa_r),
            track.pulse_width_s,
            self.gate_sigma * math.sqrt(track.pulse_width_variance_s2 + self.pw_r),
            tuple(mode.mean_hz for mode in track.rf_modes),
            tuple(self.gate_sigma * math.sqrt(mode.variance_hz2 + self.rf_r)
                  for mode in track.rf_modes),
            track.pri_s,
            (None if not track.timing_enabled else self.gate_sigma * math.sqrt(
                self.timing_r + (track.pri_variance_s2 or 0.))),
            track.existence_confidence)

    def _candidate(self, pdw, track):
        aoa_innovation = _angle_difference(pdw.aoa_deg, track.aoa_deg)
        pw_innovation = pdw.pulse_width_s - track.pulse_width_s
        mode = min(track.rf_modes, key=lambda value: abs(pdw.frequency_hz - value.mean_hz))
        rf_innovation = pdw.frequency_hz - mode.mean_hz
        variances = (track.aoa_variance_deg2 + self.aoa_r,
                     track.pulse_width_variance_s2 + self.pw_r,
                     mode.variance_hz2 + self.rf_r)
        z = (aoa_innovation / math.sqrt(variances[0]),
             pw_innovation / math.sqrt(variances[1]),
             rf_innovation / math.sqrt(variances[2]))
        timing_innovation = None
        if track.timing_enabled:
            difference = pdw.toa_s - track.phase_epoch_s
            timing_innovation = ((difference + track.pri_s / 2.)
                                 % track.pri_s - track.pri_s / 2.)
            timing_variance = self.timing_r + (track.pri_variance_s2 or 0.)
            z += (timing_innovation / math.sqrt(timing_variance),)
        if any(abs(value) > self.gate_sigma for value in z):
            return None
        nis = sum(value * value for value in z)
        weight = track.existence_confidence * math.exp(-.5 * nis)
        return weight, nis, aoa_innovation, pw_innovation, rf_innovation, timing_innovation

    def assess(self, pdw):
        self.predict_to(pdw.toa_s)
        raw = []
        for track in self.tracks.values():
            candidate = self._candidate(pdw, track)
            if candidate is not None:
                raw.append((track, candidate))
        total = self.clutter_weight + sum(item[1][0] for item in raw)
        associations = [TrackAssociation(track.emitter_id,
            candidate[0] / total, candidate[1], candidate[2], candidate[3],
            candidate[4], candidate[5]) for track, candidate in raw]
        return tuple(sorted(associations, key=lambda item: item.confidence, reverse=True))

    @staticmethod
    def _kalman_scalar(mean, variance, measurement, measurement_variance):
        gain = variance / (variance + measurement_variance)
        return mean + gain * (measurement - mean), (1. - gain) * variance, gain

    def _update(self, track, batch, association):
        n = len(batch)
        angles = np.deg2rad([pdw.aoa_deg for pdw in batch])
        measured_aoa = float(np.rad2deg(np.angle(np.mean(np.exp(1j * angles)))))
        aoa_innovation = _angle_difference(measured_aoa, track.aoa_deg)
        aoa_measurement = track.aoa_deg + aoa_innovation
        track.aoa_deg, track.aoa_variance_deg2, aoa_gain = self._kalman_scalar(
            track.aoa_deg, track.aoa_variance_deg2, aoa_measurement,
            self.aoa_r / n)
        track.aoa_deg = (track.aoa_deg + 180.) % 360. - 180.
        measured_pw = float(np.mean([pdw.pulse_width_s for pdw in batch]))
        track.pulse_width_s, track.pulse_width_variance_s2, pw_gain = self._kalman_scalar(
            track.pulse_width_s, track.pulse_width_variance_s2,
            measured_pw, self.pw_r / n)
        for pdw in batch:
            mode = min(track.rf_modes,
                key=lambda value: abs(pdw.frequency_hz - value.mean_hz))
            mode.mean_hz, mode.variance_hz2, _ = self._kalman_scalar(
                mode.mean_hz, mode.variance_hz2, pdw.frequency_hz, self.rf_r)
            mode.updates += 1
        track.last_update_s = batch[-1].toa_s
        track.last_confidence_s = batch[-1].toa_s
        track.confirmed_updates += 1
        track.associated_pdws.extend(pdw.pdw_id for pdw in batch)
        track.existence_log_odds = min(8., track.existence_log_odds
            + self.hit_log_odds * max(.25, association.confidence))
        track.last_innovation = {
            "association_confidence": association.confidence,
            "normalized_innovation_squared": association.normalized_innovation_squared,
            "aoa_innovation_deg": association.aoa_innovation_deg,
            "pulse_width_innovation_s": association.pulse_width_innovation_s,
            "rf_innovation_hz": association.rf_innovation_hz,
            "timing_innovation_s": association.timing_innovation_s,
            "aoa_kalman_gain": aoa_gain, "pulse_width_kalman_gain": pw_gain,
        }

    def process_pdw(self, pdw):
        if pdw.pdw_id in self._seen:
            raise ValueError("Duplicate PDW ID")
        if pdw.toa_s < self._last_toa_s:
            raise ValueError("PDWs must arrive in TOA order")
        self._seen.add(pdw.pdw_id)
        self._last_toa_s = pdw.toa_s
        associations = self.assess(pdw)
        if not associations:
            return TrackerDecision(pdw.pdw_id, "unclaimed", None, associations)
        if (len(associations) > 1 and associations[0].confidence -
                associations[1].confidence < self.ambiguity_margin):
            return TrackerDecision(pdw.pdw_id, "ambiguous", None, associations)
        if associations[0].confidence < self.confirmation_confidence:
            return TrackerDecision(pdw.pdw_id, "unclaimed", None, associations)
        selected = associations[0]
        buffer = self._provisional[selected.emitter_id]
        buffer[:] = [(item, score) for item, score in buffer
                     if pdw.toa_s - item.toa_s <= self.confirmation_span_s]
        buffer.append((pdw, selected))
        if len(buffer) < self.min_confirm_pulses:
            return TrackerDecision(pdw.pdw_id, "provisional",
                                   selected.emitter_id, associations)
        batch = [item for item, _ in buffer]
        batch_score = buffer[-1][1]
        self._update(self.tracks[selected.emitter_id], batch, batch_score)
        confirmed = tuple(item.pdw_id for item in batch)
        buffer.clear()
        return TrackerDecision(pdw.pdw_id, "confirmed",
            selected.emitter_id, associations, confirmed)

    def coast_to(self, time_s):
        self.predict_to(time_s)
        for track in self.tracks.values():
            elapsed = max(0., time_s - track.last_confidence_s)
            if time_s > track.last_update_s:
                track.existence_log_odds = max(-8., track.existence_log_odds
                    - self.miss_log_odds_s * elapsed)
            track.last_confidence_s = time_s

    def snapshot(self, time_s=None):
        if time_s is not None:
            self.predict_to(time_s)
        rows = []
        for track in self.tracks.values():
            gate = self.gate(track.emitter_id)
            rows.append({
                "emitter_id": track.emitter_id, "status": track.status,
                "existence_confidence": track.existence_confidence,
                "last_update_s": track.last_update_s,
                "confirmed_updates": track.confirmed_updates,
                "associated_pdw_count": len(track.associated_pdws),
                "state": {"aoa_deg": track.aoa_deg,
                    "pulse_width_s": track.pulse_width_s,
                    "pri_s": track.pri_s,
                    "rf_modes_hz": [mode.mean_hz for mode in track.rf_modes]},
                "covariance": {"aoa_variance_deg2": track.aoa_variance_deg2,
                    "pulse_width_variance_s2": track.pulse_width_variance_s2,
                    "pri_variance_s2": track.pri_variance_s2,
                    "rf_mode_variances_hz2": [mode.variance_hz2
                                              for mode in track.rf_modes]},
                "gates": asdict(gate),
                "last_innovation": track.last_innovation,
            })
        return {"time_s": self._last_toa_s if time_s is None else time_s,
                "tracks": rows,
                "confidence_note": "relative model confidence; not yet calibrated"}
