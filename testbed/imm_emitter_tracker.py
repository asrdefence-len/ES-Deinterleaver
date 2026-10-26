"""Experimental IMM-style signal-model tracker with conservative emitter association.

Model probabilities use heuristic evidence, not calibrated generative
likelihoods. This is a research baseline, not a validated textbook IMM.
Observations are non-overlapping time windows; physical IDs remain hypotheses.
"""
from dataclasses import dataclass, field
import math
from testbed.signal_models import (
    PRI_MODELS, RF_MODELS, markov_update, pri_evidence, frequency_evidence,
)


@dataclass
class ProbabilisticEmitter:
    emitter_id: str
    last_end_s: float
    pri_s: float
    frequency_hz: float
    aoa_deg: float
    pulse_width_s: float
    pri_probabilities: tuple = (0.25, 0.25, 0.25, 0.25)
    frequency_probabilities: tuple = (1/3, 1/3, 1/3)
    observations: list = field(default_factory=list)

    def snapshot(self):
        return {
            "emitter_id": self.emitter_id,
            "train_ids": [o.train_id for o in self.observations],
            "pri_s": self.pri_s,
            "frequency_hz": self.frequency_hz,
            "pri_models": dict(zip(PRI_MODELS, self.pri_probabilities)),
            "frequency_models": dict(zip(RF_MODELS, self.frequency_probabilities)),
        }


class IMMEmitterTracker:
    """Separate association/birth hypotheses from signal-evolution models.

    Association uses spatial/pulse-width continuity and a permissive
    frequency likelihood so switching does not automatically break identity.
    Ambiguous observations are withheld, never assigned arbitrarily.
    """

    def __init__(self, max_gap_s=0.3, aoa_sigma_deg=1.5,
                 pw_sigma_s=0.25e-6, ambiguity_margin=1.5):
        self.max_gap_s = max_gap_s
        self.aoa_sigma_deg = aoa_sigma_deg
        self.pw_sigma_s = pw_sigma_s
        self.ambiguity_margin = ambiguity_margin
        self.tracks = {}
        self.pending = []
        self._next_id = 1
        self._seen = set()
        self._last_start = float("-inf")

    def process(self, obs):
        if obs.train_id in self._seen:
            raise ValueError("Duplicate observation ID")
        if obs.start_toa_s < self._last_start:
            raise ValueError("Observations must arrive in start-time order")
        self._seen.add(obs.train_id)
        self._last_start = obs.start_toa_s
        candidates = []
        for track in self.tracks.values():
            gap = obs.start_toa_s - track.last_end_s
            if gap < -1e-9 or gap > self.max_gap_s:
                continue
            aoa_scale = max(self.aoa_sigma_deg, obs.aoa_sigma_deg or 0)
            pw_scale = max(self.pw_sigma_s, obs.pulse_width_sigma_s or 0)
            aoa_z = (obs.aoa_deg - track.aoa_deg) / aoa_scale
            pw_z = (obs.pulse_width_s - track.pulse_width_s) / pw_scale
            if abs(aoa_z) > 4 or abs(pw_z) > 4:
                continue
            # Frequency switching is permitted, but incurs a soft cost.
            # Large frequency excursions alone must not be a hard rejection.
            rf_z = abs(obs.frequency_hz - track.frequency_hz) / 25e6
            score = aoa_z**2 + pw_z**2 + min(rf_z**2, 9)
            candidates.append((score, track))
        candidates.sort(key=lambda x: x[0])
        if candidates and (candidates[0][0] > 16 or
                           (len(candidates) > 1 and
                            candidates[1][0] - candidates[0][0] < self.ambiguity_margin)):
            self.pending.append(obs)
            return None
        if not candidates:
            eid = f"E{self._next_id}"
            self._next_id += 1
            track = ProbabilisticEmitter(
                eid, obs.end_toa_s, obs.pri_s, obs.frequency_hz,
                obs.aoa_deg, obs.pulse_width_s)
            self.tracks[eid] = track
            # A birth is provisional, not proof of a new physical source.
        else:
            track = candidates[0][1]
            track.pri_probabilities = markov_update(
                track.pri_probabilities, pri_evidence(obs, track.pri_s))
            track.frequency_probabilities = markov_update(
                track.frequency_probabilities,
                frequency_evidence(obs, track.frequency_hz))
        track.last_end_s = max(track.last_end_s, obs.end_toa_s)
        track.pri_s = obs.pri_s
        track.frequency_hz = obs.frequency_hz
        track.aoa_deg = obs.aoa_deg
        track.pulse_width_s = obs.pulse_width_s
        track.observations.append(obs)
        return track.emitter_id

    def snapshot(self):
        return {
            "tracks": [t.snapshot() for t in self.tracks.values()],
            "pending_train_ids": [o.train_id for o in self.pending],
            "limitations": "IMM-style heuristic model probabilities; not calibrated",
        }
