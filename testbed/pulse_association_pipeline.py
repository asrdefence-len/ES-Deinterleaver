"""Pulse association as a component owned by the emitter-tracking pipeline.

This first integration reuses V0 fixed-PRI discovery and the existing
PulseTrainObservation / EmitterTrackManager boundary. No truth input,
OT, density splitting, or new emitter-association heuristic is introduced.
Window-local train membership and provisional emitter identity are distinct.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from testbed.emitter_track_manager import EmitterTrackManager
from testbed.pulse_train_observation import PulseTrainObservation


@dataclass(frozen=True)
class PulseMeasurement:
    pdw_id: int
    toa_s: float
    frequency_hz: float
    pulse_width_s: float
    aoa_deg: float


@dataclass(frozen=True)
class AssociatedTrain:
    observation: PulseTrainObservation
    blob_id: int
    phase_epoch_s: float
    rms_residual_s: float
    occupied_slots: int
    span_slots: int


@dataclass(frozen=True)
class AssociationWindow:
    window_id: str
    trains: tuple[AssociatedTrain, ...]
    unexplained_pdw_ids: tuple[int, ...]
    pulse_train_assignments: dict[int, Optional[str]]
    train_emitter_assignments: dict[str, Optional[str]]
    provisional_emitter_assignments: dict[int, Optional[str]]


class FixedPRIPulseAssociation:
    """Batch association inside a frozen blob, exported in SI units.

    V0's internal histogram constants are in microseconds. Convert SI TOAs
    to relative microseconds before calling it, and convert its model back.
    Greedy fixed-PRI discovery cannot resolve all competing memberships or
    fundamental/harmonic ambiguity. Unexplained pulses are retained.
    """

    def __init__(self, pri_min_s=20e-6, pri_max_s=200e-6,
                 tolerance_s=2e-6, bin_width_s=.1e-6,
                 min_pulses=20, max_trains_per_blob=4):
        if (not all(np.isfinite(v) for v in (pri_min_s, pri_max_s, tolerance_s, bin_width_s))
                or not 0 < pri_min_s < pri_max_s or not 0 < tolerance_s < pri_min_s / 4
                or bin_width_s <= 0 or min_pulses < 3 or max_trains_per_blob < 1):
            raise ValueError("Invalid fixed-PRI association parameters")
        self.pri_min_s = pri_min_s
        self.pri_max_s = pri_max_s
        self.tolerance_s = tolerance_s
        self.bin_width_s = bin_width_s
        self.min_pulses = min_pulses
        self.max_trains_per_blob = max_trains_per_blob

    def process(self, pdws, blob_labels, window_id):
        """Return trains + total PDW accounting; no emitter identities assigned."""
        from deinterleaver_v0 import discover

        pdws = tuple(pdws)
        labels = np.asarray(blob_labels)
        if labels.shape != (len(pdws),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Require one integer blob label per PDW")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("A nonempty window ID is required")
        ids = [p.pdw_id for p in pdws]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate PDW IDs")
        values = np.asarray([(p.toa_s, p.frequency_hz, p.pulse_width_s, p.aoa_deg)
                             for p in pdws], float).reshape(-1, 4)
        if (not np.all(np.isfinite(values))
                or (len(pdws) and (np.any(values[:, 1] <= 0) or np.any(values[:, 2] <= 0)))):
            raise ValueError("Require finite measurements and positive RF/PW")
        assignments = {i: None for i in ids}
        if not pdws:
            return (), (), assignments
        origin = float(values[:, 0].min())
        microsecond = 1e-6
        trains = []
        for blob in np.unique(labels[labels >= 0]):
            indices = np.flatnonzero(labels == blob)
            if len(indices) < self.min_pulses:
                continue
            times = (values[indices, 0] - origin) / microsecond
            models, _ = discover(times, self.pri_min_s / microsecond,
                self.pri_max_s / microsecond, self.tolerance_s / microsecond,
                self.min_pulses, self.max_trains_per_blob, self.bin_width_s / microsecond)
            for model in models:
                members = sorted((pdws[int(indices[i])] for i in model.pdw_indices),
                                 key=lambda p: (p.toa_s, p.pdw_id))
                train_id = f"{window_id}:B{int(blob)}:{model.id}"
                refs = tuple(p.pdw_id for p in members)
                if any(assignments[i] is not None for i in refs):
                    raise RuntimeError("A PDW was assigned to multiple pulse trains")
                angles = np.deg2rad([p.aoa_deg for p in members])
                bearing = float(np.rad2deg(np.angle(np.mean(np.exp(1j * angles)))))
                observation = PulseTrainObservation(
                    schema_version=1, train_id=train_id,
                    start_toa_s=members[0].toa_s, end_toa_s=members[-1].toa_s,
                    pri_s=model.period * microsecond, pri_sigma_s=None,
                    frequency_hz=float(np.median([p.frequency_hz for p in members])),
                    frequency_sigma_hz=None, aoa_deg=bearing, aoa_sigma_deg=None,
                    pulse_width_s=float(np.median([p.pulse_width_s for p in members])),
                    pulse_width_sigma_s=None, pulse_count=len(members), pdw_ids=refs,
                    association_confidence=None,
                    toa_sequence_s=tuple(p.toa_s for p in members),
                    frequency_sequence_hz=tuple(p.frequency_hz for p in members),
                    pri_intervals_s=tuple(b.toa_s - a.toa_s for a, b in zip(members, members[1:])),
                    # Empty slots may be missed detections, other hop frequencies,
                    # or scan gaps: do not claim a known number of missed pulses.
                    missed_pulse_count=None)
                trains.append(AssociatedTrain(observation, int(blob),
                    origin + model.phase * microsecond,
                    model.rms_residual * microsecond,
                    model.occupied_slots, model.span_slots))
                assignments.update((i, train_id) for i in refs)
        trains.sort(key=lambda tr: (tr.observation.start_toa_s, tr.observation.train_id))
        unexplained = tuple(i for i in ids if assignments[i] is None)
        return tuple(trains), unexplained, assignments


class PulseAssociationTrackingPipeline:
    """Own association and persistent identity state behind one interface.

    Input windows must be disjoint and chronological. Association is batch
    within a window; emitter identity state survives between windows. The
    existing identity baseline does not merge overlapping trains or widely
    separated RF modes into one emitter. Its IDs are provisional hypotheses.
    """

    def __init__(self, association=None, identity_manager=None):
        self.association = association if association is not None else FixedPRIPulseAssociation()
        self.identity_manager = identity_manager if identity_manager is not None else EmitterTrackManager()
        self._seen_windows = set()
        self._seen_pdws = set()
        self._last_end_s = float("-inf")

    def process_window(self, pdws, blob_labels, window_id):
        pdws = tuple(pdws)
        ids = {p.pdw_id for p in pdws}
        if window_id in self._seen_windows or ids & self._seen_pdws:
            raise ValueError("Window or PDW IDs were already processed")
        if pdws and min(p.toa_s for p in pdws) < self._last_end_s:
            raise ValueError("Windows must be disjoint and chronological")
        trains, unexplained, train_assignments = self.association.process(pdws, blob_labels, window_id)
        identities = {}
        emitter_assignments = {p.pdw_id: None for p in pdws}
        for train in trains:
            obs = train.observation
            emitter_id = self.identity_manager.process(obs)
            identities[obs.train_id] = emitter_id
            if emitter_id is not None:
                emitter_assignments.update((i, emitter_id) for i in obs.pdw_ids)
        self._seen_windows.add(window_id)
        self._seen_pdws.update(ids)
        if pdws:
            self._last_end_s = max(p.toa_s for p in pdws)
        return AssociationWindow(window_id, trains, unexplained, train_assignments,
                                 identities, emitter_assignments)
