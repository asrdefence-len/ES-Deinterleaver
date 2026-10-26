"""Slow-rate emitter identity manager for pulse-train observations.

This is a conservative deterministic baseline, NOT MHT. It never sees PDW
truth labels. PRI changes are allowed; identity association uses RF/AOA/PW
and temporal continuity. Ambiguous matches are explicitly withheld.
"""
from dataclasses import dataclass, field
from testbed.pulse_train_observation import PulseTrainObservation


@dataclass
class EmitterTrack:
    emitter_id: str
    last_end_s: float
    frequency_hz: float
    aoa_deg: float
    pulse_width_s: float
    pri_s: float
    observations: list = field(default_factory=list)


class EmitterTrackManager:
    def __init__(self, frequency_gate_hz=500_000, aoa_gate_deg=3.0,
                 pw_gate_s=1e-6, max_gap_s=0.3, ambiguity_margin=0.25):
        if min(frequency_gate_hz, aoa_gate_deg, pw_gate_s, max_gap_s) <= 0:
            raise ValueError("Gates and maximum gap must be positive")
        self.frequency_gate_hz = frequency_gate_hz
        self.aoa_gate_deg = aoa_gate_deg
        self.pw_gate_s = pw_gate_s
        self.max_gap_s = max_gap_s
        self.ambiguity_margin = ambiguity_margin
        self.tracks = {}
        self.pending = []
        self._next_id = 1
        self._seen_train_ids = set()
        self._last_start_s = float("-inf")

    def process(self, observation: PulseTrainObservation):
        """Return emitter ID, or None if identity is ambiguous.

        Observations must arrive in nondecreasing start-time order. Pending
        observations are not automatically assigned to a new emitter.
        """
        if observation.train_id in self._seen_train_ids:
            raise ValueError("Duplicate pulse-train observation ID")
        if observation.start_toa_s < self._last_start_s:
            raise ValueError("Out-of-order pulse-train observation")
        self._seen_train_ids.add(observation.train_id)
        self._last_start_s = observation.start_toa_s

        candidates = []
        for track in self.tracks.values():
            gap = observation.start_toa_s - track.last_end_s
            # Separate pulse-train observations that overlap in time are
            # provisionally distinct simultaneous trains. Do not silently
            # merge them into one physical emitter in this simple baseline.
            # This assumes non-overlapping observation windows per emitter;
            # a future tracker must handle sliding-window updates explicitly.
            if gap < 0 or gap > self.max_gap_s:
                continue
            df = abs(observation.frequency_hz - track.frequency_hz) / self.frequency_gate_hz
            da = abs(observation.aoa_deg - track.aoa_deg) / self.aoa_gate_deg
            dp = abs(observation.pulse_width_s - track.pulse_width_s) / self.pw_gate_s
            if df <= 1 and da <= 1 and dp <= 1:
                candidates.append((df * df + da * da + dp * dp, track))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < self.ambiguity_margin:
            self.pending.append(observation)
            return None
        if candidates:
            track = candidates[0][1]
        else:
            # A new ID is a provisional physical-emitter hypothesis, not proof
            # that a genuinely new radar appeared.
            emitter_id = f"E{self._next_id}"
            self._next_id += 1
            track = EmitterTrack(emitter_id, observation.end_toa_s,
                                 observation.frequency_hz, observation.aoa_deg,
                                 observation.pulse_width_s, observation.pri_s)
            self.tracks[emitter_id] = track
        track.last_end_s = max(track.last_end_s, observation.end_toa_s)
        track.frequency_hz = observation.frequency_hz
        track.aoa_deg = observation.aoa_deg
        track.pulse_width_s = observation.pulse_width_s
        track.pri_s = observation.pri_s
        track.observations.append(observation)
        return track.emitter_id

    def snapshot(self):
        return {
            "tracks": [{
                "emitter_id": t.emitter_id,
                "last_end_s": t.last_end_s,
                "pri_s": t.pri_s,
                "train_ids": [o.train_id for o in t.observations],
            } for t in self.tracks.values()],
            "pending_train_ids": [o.train_id for o in self.pending],
        }
