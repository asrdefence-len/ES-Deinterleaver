"""Interpretable, uncalibrated evidence accumulation for emitter association.

The score produced here is deliberately called an association score rather
than a probability.  It combines independently inspectable evidence and
retains state through sparse or empty observation windows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AssociationEvidence:
    window_id: str
    time_s: float
    candidate_groups: int
    candidate_pulses: int
    associated_pulses: int
    timing_score: float | None
    rf_score: float | None
    signal_score: float | None
    continuity_score: float
    instantaneous_score: float | None
    track_score: float

    def to_dict(self):
        return asdict(self)


class AssociationConfidenceTracker:
    """Accumulate timing, RF, signal and continuity evidence over time."""

    def __init__(self, *, expected_rf_hz, expected_aoa_deg, expected_pw_s,
                 expected_pri_s, timing_gate_s=5e-6, rf_gate_hz=40e6,
                 aoa_gate_deg=3., pw_gate_s=.12e-6, hold_time_s=.30,
                 weights=None, memory=.65):
        self.expected_rf_hz = np.asarray(expected_rf_hz, float)
        values = (expected_aoa_deg, expected_pw_s, expected_pri_s,
                  timing_gate_s, rf_gate_hz, aoa_gate_deg, pw_gate_s,
                  hold_time_s, memory)
        if (self.expected_rf_hz.ndim != 1 or not len(self.expected_rf_hz)
                or not np.all(np.isfinite(self.expected_rf_hz))
                or np.any(self.expected_rf_hz <= 0)
                or any(not math.isfinite(v) for v in values)
                or expected_pw_s <= 0 or expected_pri_s <= 0
                or min(timing_gate_s, rf_gate_hz, aoa_gate_deg,
                       pw_gate_s, hold_time_s) <= 0
                or not 0 <= memory < 1):
            raise ValueError("Invalid association-confidence configuration")
        self.expected_aoa_deg = float(expected_aoa_deg)
        self.expected_pw_s = float(expected_pw_s)
        self.expected_pri_s = float(expected_pri_s)
        self.timing_gate_s = float(timing_gate_s)
        self.rf_gate_hz = float(rf_gate_hz)
        self.aoa_gate_deg = float(aoa_gate_deg)
        self.pw_gate_s = float(pw_gate_s)
        self.hold_time_s = float(hold_time_s)
        self.memory = float(memory)
        self.weights = weights or {
            "timing": .40, "rf": .25, "signal": .20, "continuity": .15}
        if set(self.weights) != {"timing", "rf", "signal", "continuity"}:
            raise ValueError("Require timing, RF, signal and continuity weights")
        if any(not math.isfinite(v) or v <= 0 for v in self.weights.values()):
            raise ValueError("Confidence weights must be finite and positive")
        self._last_seen_s = None
        self._last_time_s = None
        self._track_score = None

    @staticmethod
    def _angular_difference(values, reference):
        return np.abs((np.asarray(values) - reference + 180.) % 360. - 180.)

    @staticmethod
    def _bounded(value):
        return float(np.clip(value, 0., 1.))

    def _measurement_scores(self, pdws, rf_transition_score):
        if not pdws:
            return None, None
        rf = np.asarray([p.frequency_hz for p in pdws])
        aoa = np.asarray([p.aoa_deg for p in pdws])
        pw = np.asarray([p.pulse_width_s for p in pdws])
        rf_error = np.min(np.abs(rf[:, None] - self.expected_rf_hz[None, :]), axis=1)
        mode_match = float(np.mean(np.exp(-.5 * (rf_error / self.rf_gate_hz) ** 2)))
        transition = mode_match if rf_transition_score is None else rf_transition_score
        rf_score = .75 * mode_match + .25 * self._bounded(transition)
        aoa_error = self._angular_difference(aoa, self.expected_aoa_deg)
        signal = np.exp(-.5 * ((aoa_error / self.aoa_gate_deg) ** 2
                              + ((pw - self.expected_pw_s) / self.pw_gate_s) ** 2))
        return self._bounded(rf_score), self._bounded(np.mean(signal))

    def _timing_score(self, trains, candidate_count):
        if not trains:
            return None, 0
        counts = np.asarray([t.observation.pulse_count for t in trains], float)
        associated = int(np.sum(counts))
        residual = np.asarray([t.rms_residual_s for t in trains])
        residual_quality = float(np.average(
            np.exp(-.5 * (residual / self.timing_gate_s) ** 2), weights=counts))
        pri_quality = []
        for train in trains:
            pri = train.observation.pri_s
            harmonic = int(np.clip(round(pri / self.expected_pri_s), 1, 4))
            normalised = pri / harmonic
            pri_quality.append(math.exp(-.5 * (
                (normalised - self.expected_pri_s) / (.01 * self.expected_pri_s)) ** 2))
        pri_quality = float(np.average(pri_quality, weights=counts))
        coverage = min(1., associated / max(candidate_count, 1))
        score = (max(coverage, 1e-9) * max(residual_quality, 1e-9)
                 * max(pri_quality, 1e-9)) ** (1 / 3)
        return self._bounded(score), associated

    def update(self, *, window_id, time_s, window_end_s, pdws, trains,
               candidate_groups, rf_transition_score=None):
        pdws, trains = tuple(pdws), tuple(trains)
        if (not window_id or not math.isfinite(time_s) or not math.isfinite(window_end_s)
                or window_end_s < time_s or candidate_groups < 0
                or self._last_time_s is not None and time_s < self._last_time_s):
            raise ValueError("Confidence windows must be valid and chronological")
        if pdws:
            first, last = min(p.toa_s for p in pdws), max(p.toa_s for p in pdws)
            gap = 0. if self._last_seen_s is None else max(0., first - self._last_seen_s)
            continuity = math.exp(-gap / self.hold_time_s)
            self._last_seen_s = last
        else:
            elapsed = (window_end_s - self._last_seen_s
                       if self._last_seen_s is not None else self.hold_time_s)
            continuity = math.exp(-max(0., elapsed) / self.hold_time_s)
        timing, associated = self._timing_score(trains, len(pdws))
        rf, signal = self._measurement_scores(pdws, rf_transition_score)
        components = {"timing": timing, "rf": rf, "signal": signal,
                      "continuity": self._bounded(continuity)}
        present = {name: value for name, value in components.items() if value is not None}
        if present:
            total_weight = sum(self.weights[name] for name in present)
            instantaneous = sum(self.weights[name] * value
                                for name, value in present.items()) / total_weight
        else:
            instantaneous = None
        if self._track_score is None:
            track = instantaneous if instantaneous is not None else 0.
        elif pdws and instantaneous is not None:
            track = self.memory * self._track_score + (1. - self.memory) * instantaneous
        else:
            track = self._track_score * continuity
        self._track_score = self._bounded(track)
        self._last_time_s = window_end_s
        return AssociationEvidence(
            window_id, float(time_s), int(candidate_groups), len(pdws), associated,
            timing, rf, signal, self._bounded(continuity), instantaneous,
            self._track_score)
