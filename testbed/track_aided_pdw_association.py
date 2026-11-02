"""Prediction-assisted raw-PDW association for established emitter tracks.

This is the maintenance/reacquisition path, not the discovery path.  It uses
track predictions to evaluate sparse PDWs before clustering.  Assignments stay
provisional until several compatible pulses accumulate; ambiguous and rejected
PDWs remain available to the normal discovery pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrackPrediction:
    emitter_id: str
    generated_at_s: float
    last_toa_s: float
    frequency_modes_hz: tuple[float, ...]
    aoa_deg: float
    pulse_width_s: float
    pri_s: float | None = None
    phase_epoch_s: float | None = None
    timing_sigma_s: float = 5e-6
    rf_sigma_hz: float = 15e6
    aoa_sigma_deg: float = 1.5
    pulse_width_sigma_s: float = .15e-6
    timing_horizon_s: float = .3
    reacquisition_horizon_s: float = 10.
    track_score: float | None = None

    def __post_init__(self):
        positive = (self.timing_sigma_s, self.rf_sigma_hz,
                    self.aoa_sigma_deg, self.pulse_width_sigma_s,
                    self.timing_horizon_s, self.reacquisition_horizon_s)
        if (not self.emitter_id or not self.frequency_modes_hz
                or any(not math.isfinite(v) or v <= 0 for v in positive)
                or any(not math.isfinite(v) or v <= 0 for v in self.frequency_modes_hz)
                or not all(math.isfinite(v) for v in (
                    self.generated_at_s, self.last_toa_s, self.aoa_deg,
                    self.pulse_width_s))
                or self.pulse_width_s <= 0
                or (self.pri_s is None) != (self.phase_epoch_s is None)
                or self.pri_s is not None and self.pri_s <= 0
                or self.track_score is not None and not 0 <= self.track_score <= 1):
            raise ValueError("Invalid track prediction")


@dataclass(frozen=True)
class TrackCompatibility:
    emitter_id: str
    score: float
    timing_score: float | None
    rf_score: float
    aoa_score: float
    pulse_width_score: float


@dataclass(frozen=True)
class PDWGateDecision:
    pdw_id: int
    status: str
    selected_emitter_id: str | None
    candidates: tuple[TrackCompatibility, ...]
    confirmed_pdw_ids: tuple[int, ...] = ()

    def to_dict(self):
        row = asdict(self)
        row["candidates"] = [asdict(item) for item in self.candidates]
        return row


class TrackAidedPDWAssociator:
    """Gate PDWs against predictions and confirm only cumulative evidence."""

    def __init__(self, *, score_threshold=.8, ambiguity_margin=.08,
                 hard_gate_sigma=4., min_confirm_pulses=5,
                 confirmation_span_s=.010):
        if (not 0 < score_threshold <= 1 or ambiguity_margin < 0
                or hard_gate_sigma <= 0 or min_confirm_pulses < 2
                or confirmation_span_s <= 0):
            raise ValueError("Invalid track-aided association configuration")
        self.score_threshold = score_threshold
        self.ambiguity_margin = ambiguity_margin
        self.hard_gate_sigma = hard_gate_sigma
        self.min_confirm_pulses = min_confirm_pulses
        self.confirmation_span_s = confirmation_span_s
        self._provisional = {}
        self._last_toa_s = float("-inf")
        self._seen = set()

    @staticmethod
    def _angular_difference(a, b):
        return abs((a - b + 180.) % 360. - 180.)

    @staticmethod
    def _score(z):
        return math.exp(-.5 * z * z)

    def compare(self, pdw, prediction):
        gap = pdw.toa_s - prediction.last_toa_s
        if gap < 0 or gap > prediction.reacquisition_horizon_s:
            return None
        rf_z = min(abs(pdw.frequency_hz - mode)
                   for mode in prediction.frequency_modes_hz) / prediction.rf_sigma_hz
        aoa_z = self._angular_difference(
            pdw.aoa_deg, prediction.aoa_deg) / prediction.aoa_sigma_deg
        pw_z = abs(pdw.pulse_width_s - prediction.pulse_width_s) / prediction.pulse_width_sigma_s
        if max(rf_z, aoa_z, pw_z) > self.hard_gate_sigma:
            return None
        timing = None
        if (prediction.pri_s is not None and gap <= prediction.timing_horizon_s):
            difference = pdw.toa_s - prediction.phase_epoch_s
            residual = abs((difference + prediction.pri_s / 2.)
                           % prediction.pri_s - prediction.pri_s / 2.)
            timing_z = residual / prediction.timing_sigma_s
            if timing_z > self.hard_gate_sigma:
                return None
            timing = self._score(timing_z)
        components = {
            "timing": (timing, .40), "rf": (self._score(rf_z), .25),
            "aoa": (self._score(aoa_z), .20), "pw": (self._score(pw_z), .15)}
        available = [(value, weight) for value, weight in components.values()
                     if value is not None]
        score = sum(value * weight for value, weight in available) / sum(
            weight for _, weight in available)
        return TrackCompatibility(prediction.emitter_id, score, timing,
            components["rf"][0], components["aoa"][0], components["pw"][0])

    def process(self, pdw, predictions):
        if pdw.pdw_id in self._seen:
            raise ValueError("Duplicate PDW ID")
        if pdw.toa_s < self._last_toa_s:
            raise ValueError("PDWs must be processed in TOA order")
        self._seen.add(pdw.pdw_id)
        self._last_toa_s = pdw.toa_s
        candidates = tuple(sorted((candidate for prediction in predictions
            if (candidate := self.compare(pdw, prediction)) is not None),
            key=lambda item: item.score, reverse=True))
        if not candidates or candidates[0].score < self.score_threshold:
            return PDWGateDecision(pdw.pdw_id, "unclaimed", None, candidates)
        if (len(candidates) > 1 and
                candidates[0].score - candidates[1].score < self.ambiguity_margin):
            return PDWGateDecision(pdw.pdw_id, "ambiguous", None, candidates)
        emitter_id = candidates[0].emitter_id
        buffer = self._provisional.setdefault(emitter_id, [])
        buffer[:] = [item for item in buffer
                     if pdw.toa_s - item.toa_s <= self.confirmation_span_s]
        buffer.append(pdw)
        if len(buffer) < self.min_confirm_pulses:
            return PDWGateDecision(pdw.pdw_id, "provisional", emitter_id, candidates)
        confirmed = tuple(item.pdw_id for item in buffer)
        buffer.clear()
        return PDWGateDecision(pdw.pdw_id, "confirmed", emitter_id,
                               candidates, confirmed)

    def provisional_snapshot(self):
        return {emitter_id: tuple(item.pdw_id for item in pdws)
                for emitter_id, pdws in self._provisional.items() if pdws}
