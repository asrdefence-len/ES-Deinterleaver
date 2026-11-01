"""Bounded whole-blob association hypotheses; no PDW processing.

One track may receive overlapping blobs and several RF modes. A new RF
mode and a new emitter remain competing explanations. Relative costs are
heuristics, NOT calibrated probabilities or proof of physical identity.
"""
from dataclasses import dataclass, replace
import math

from testbed.blob_observation import BlobObservation


@dataclass(frozen=True)
class RFMode:
    low_hz: float
    median_hz: float
    high_hz: float


@dataclass(frozen=True)
class BlobTrack:
    track_id: str
    last_end_s: float
    aoa_deg: float
    aoa_spread_deg: float
    pw_median_s: float
    pw_spread_s: float
    rf_modes: tuple[RFMode, ...]
    recent_blob_ids: tuple[str, ...]


@dataclass(frozen=True)
class BlobHypothesis:
    tracks: tuple[BlobTrack, ...] = ()
    recent_assignments: tuple[tuple[str, str | None], ...] = ()
    cost: float = 0.


class BlobEmitterTracker:
    """Bounded beam over associations of whole blob observations.

    Existing-mode match, new-mode match, new emitter and capacity-unresolved
    explanations are considered without modifying input membership. Limits
    bound beam size, active tracks, modes, matching branches and history.
    """

    def __init__(self, beam_width=16, max_tracks=64, max_modes=8,
                 max_match_branches=4, history_limit=256, track_history_limit=16,
                 max_gap_s=.3, aoa_floor_deg=.5, pw_floor_s=.05e-6,
                 feature_gate=4., rf_gate_hz=5e6, birth_cost=3., new_mode_cost=3.,
                 ambiguity_margin=2.):
        limits = (beam_width, max_tracks, max_modes, max_match_branches,
                  history_limit, track_history_limit)
        scales = (max_gap_s, aoa_floor_deg, pw_floor_s, feature_gate, rf_gate_hz)
        costs = (birth_cost, new_mode_cost, ambiguity_margin)
        if (any(not isinstance(v, int) or v < 1 for v in limits)
                or any(not math.isfinite(v) or v <= 0 for v in scales)
                or any(not math.isfinite(v) or v < 0 for v in costs)):
            raise ValueError("Invalid blob tracker configuration")
        self.beam_width = beam_width
        self.max_tracks = max_tracks
        self.max_modes = max_modes
        self.max_match_branches = max_match_branches
        self.history_limit = history_limit
        self.track_history_limit = track_history_limit
        self.max_gap_s = max_gap_s
        self.aoa_floor_deg = aoa_floor_deg
        self.pw_floor_s = pw_floor_s
        self.feature_gate = feature_gate
        self.rf_gate_hz = rf_gate_hz
        self.birth_cost = birth_cost
        self.new_mode_cost = new_mode_cost
        self.ambiguity_margin = ambiguity_margin
        self.hypotheses = [BlobHypothesis()]
        self._last_start_s = float("-inf")
        self._recent_ids = ()
        self._sequence = 0

    @staticmethod
    def _pw_spread(obs):
        return (obs.pw_p95_s - obs.pw_p05_s) / 3.29

    def _compatibility(self, obs, track):
        if obs.start_s - track.last_end_s > self.max_gap_s:
            return None
        aoa_scale = max(self.aoa_floor_deg,
                        math.hypot(obs.aoa_spread_deg, track.aoa_spread_deg))
        pw_scale = max(self.pw_floor_s, math.hypot(self._pw_spread(obs), track.pw_spread_s))
        da = abs((obs.aoa_deg - track.aoa_deg + 180) % 360 - 180) / aoa_scale
        dp = abs(obs.pw_median_s - track.pw_median_s) / pw_scale
        if da > self.feature_gate or dp > self.feature_gate:
            return None
        gaps = [max(0., obs.frequency_p05_hz - mode.high_hz,
                        mode.low_hz - obs.frequency_p95_hz) for mode in track.rf_modes]
        mode_index = min(range(len(gaps)), key=gaps.__getitem__)
        known_mode = gaps[mode_index] <= self.rf_gate_hz
        cost = .5 * (da * da + dp * dp) + (0. if known_mode else self.new_mode_cost)
        return cost, mode_index if known_mode else None

    def _new_track(self, obs, track_id):
        return BlobTrack(track_id, obs.end_s, obs.aoa_deg, obs.aoa_spread_deg,
            obs.pw_median_s, self._pw_spread(obs),
            (RFMode(obs.frequency_p05_hz, obs.frequency_median_hz, obs.frequency_p95_hz),),
            (obs.observation_id,))

    def _updated_track(self, track, obs, mode_index):
        mode = RFMode(obs.frequency_p05_hz, obs.frequency_median_hz, obs.frequency_p95_hz)
        modes = list(track.rf_modes)
        if mode_index is None:
            modes.append(mode)
            modes = modes[-self.max_modes:]
        else:
            modes[mode_index] = mode
        return replace(track, last_end_s=max(track.last_end_s, obs.end_s),
            aoa_deg=obs.aoa_deg, aoa_spread_deg=obs.aoa_spread_deg,
            pw_median_s=obs.pw_median_s, pw_spread_s=self._pw_spread(obs),
            rf_modes=tuple(modes),
            recent_blob_ids=(track.recent_blob_ids + (obs.observation_id,))[-self.track_history_limit:])

    def process(self, obs):
        if not isinstance(obs, BlobObservation):
            raise TypeError("Tracker input must be a BlobObservation, not PDWs")
        if obs.observation_id in self._recent_ids:
            raise ValueError("Duplicate recent blob observation")
        if obs.start_s < self._last_start_s:
            raise ValueError("Blob observations must arrive in start-time order")
        self._sequence += 1
        new_track_id = f"E{self._sequence}"
        branches = []
        for hypothesis in self.hypotheses:
            # Expired identities disappear from active inference, not from the
            # caller's persisted output. IDs are never reused.
            tracks = tuple(t for t in hypothesis.tracks
                           if obs.start_s - t.last_end_s <= self.max_gap_s)
            def record(updated_tracks, track_id, extra_cost):
                assignments = (hypothesis.recent_assignments +
                               ((obs.observation_id, track_id),))[-self.history_limit:]
                branches.append(BlobHypothesis(tuple(updated_tracks), assignments,
                                               hypothesis.cost + extra_cost))
            matches = []
            for i, track in enumerate(tracks):
                compatible = self._compatibility(obs, track)
                if compatible is not None:
                    matches.append((compatible[0], track.track_id, i, compatible[1]))
            for cost, _, index, mode_index in sorted(matches)[:self.max_match_branches]:
                updated = list(tracks)
                updated[index] = self._updated_track(tracks[index], obs, mode_index)
                record(updated, tracks[index].track_id, cost)
            if len(tracks) < self.max_tracks:
                record(tracks + (self._new_track(obs, new_track_id),), new_track_id, self.birth_cost)
            else:
                # Do not force a blob into an incompatible track at capacity.
                record(tracks, None, self.birth_cost)
        # Deterministic deduplication; retain alternative blob associations.
        unique = {}
        for branch in branches:
            key = (branch.tracks, branch.recent_assignments)
            if key not in unique or branch.cost < unique[key].cost:
                unique[key] = branch
        ranked = sorted(unique.values(), key=lambda h: (h.cost,
            repr(h.recent_assignments)))[:self.beam_width]
        offset = ranked[0].cost
        self.hypotheses = [replace(h, cost=h.cost - offset) for h in ranked]
        self._last_start_s = obs.start_s
        self._recent_ids = (self._recent_ids + (obs.observation_id,))[-self.history_limit:]
        return self.association_options(obs.observation_id)

    def plausible_hypotheses(self):
        return [h for h in self.hypotheses if h.cost <= self.ambiguity_margin]

    def association_options(self, observation_id):
        options = {dict(h.recent_assignments).get(observation_id)
                   for h in self.plausible_hypotheses()
                   if observation_id in dict(h.recent_assignments)}
        if not options:
            return {"status": "OUTSIDE_RETAINED_HISTORY", "candidate_track_ids": []}
        real = sorted(o for o in options if o is not None)
        status = ("UNRESOLVED_CAPACITY" if options == {None} else
                  "PROVISIONAL" if len(options) == 1 and None not in options else "AMBIGUOUS")
        return {"status": status, "candidate_track_ids": real}

    def snapshot(self):
        return {
            "retained_hypotheses": len(self.hypotheses),
            "plausible_hypotheses": len(self.plausible_hypotheses()),
            "relative_costs_are_probabilities": False,
            "observations_processed": self._sequence,
            "associations": {oid: self.association_options(oid) for oid in self._recent_ids},
            "hypotheses": [{"relative_cost": h.cost,
                "active_tracks": [{"track_id": t.track_id,
                    "aoa_deg": t.aoa_deg, "pw_median_s": t.pw_median_s,
                    "recent_blob_ids": list(t.recent_blob_ids),
                    "rf_modes_hz": [[m.low_hz, m.median_hz, m.high_hz] for m in t.rf_modes]}
                    for t in h.tracks],
                "recent_assignments": dict(h.recent_assignments)} for h in self.hypotheses],
            "limits": {name: getattr(self, name) for name in (
                "beam_width", "max_tracks", "max_modes", "max_match_branches",
                "history_limit", "track_history_limit")},
        }
