"""Candidate-family generation for pulse trains split across RF groups.

This linker does not declare physical-emitter identity.  It identifies train
observations that deserve a joint hypothesis because their AOA, pulse width,
activity time and PRI family are compatible.  Simultaneous phase-separated
trains may be modes of one multifunction emitter or independent emitters, so
both merged and separate interpretations are retained explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from testbed.pulse_association_pipeline import AssociatedTrain


@dataclass(frozen=True)
class TrainLink:
    left_train_id: str
    right_train_id: str
    base_pri_s: float
    harmonic_left: int
    harmonic_right: int
    pri_error_fraction: float
    aoa_difference_deg: float
    pulse_width_difference_s: float
    phase_residual_s: float
    phase_aligned: bool


@dataclass(frozen=True)
class CrossRFTrainFamily:
    family_id: str
    train_ids: tuple[str, ...]
    frequency_modes_hz: tuple[float, ...]
    base_pri_s: float
    phase_cluster_count: int
    links: tuple[TrainLink, ...]
    retained_hypotheses: tuple[tuple[tuple[str, ...], ...], ...]


class CrossRFTrainLinker:
    """Build bounded candidate families without using truth labels."""

    def __init__(self, *, aoa_gate_deg=3., pulse_width_gate_s=.1e-6,
                 max_time_gap_s=.3, max_harmonic=4, pri_error_fraction=.01,
                 phase_tolerance_s=2e-6):
        values = (aoa_gate_deg, pulse_width_gate_s, max_time_gap_s,
                  pri_error_fraction, phase_tolerance_s)
        if (any(not math.isfinite(v) or v <= 0 for v in values)
                or not isinstance(max_harmonic, int) or max_harmonic < 1):
            raise ValueError("Invalid cross-RF linker configuration")
        self.aoa_gate_deg = aoa_gate_deg
        self.pulse_width_gate_s = pulse_width_gate_s
        self.max_time_gap_s = max_time_gap_s
        self.max_harmonic = max_harmonic
        self.pri_error_fraction = pri_error_fraction
        self.phase_tolerance_s = phase_tolerance_s

    @staticmethod
    def _angular_difference(a, b):
        return abs((a - b + 180.) % 360. - 180.)

    def _pri_family(self, left_pri, right_pri):
        best = None
        for left_harmonic in range(1, self.max_harmonic + 1):
            left_base = left_pri / left_harmonic
            for right_harmonic in range(1, self.max_harmonic + 1):
                right_base = right_pri / right_harmonic
                base = .5 * (left_base + right_base)
                error = abs(left_base - right_base) / base
                # Equivalent ratios can be represented by 1:1 or needlessly
                # by 2:2, 3:3, etc.  Round floating noise in the error and
                # prefer the simplest harmonic explanation.
                candidate = (round(error, 12), left_harmonic + right_harmonic,
                             -base, base, left_harmonic, right_harmonic, error)
                if best is None or candidate < best:
                    best = candidate
        return best[3], best[4], best[5], best[6]

    def compare(self, left: AssociatedTrain, right: AssociatedTrain):
        a, b = left.observation, right.observation
        gap = max(a.start_toa_s, b.start_toa_s) - min(a.end_toa_s, b.end_toa_s)
        if gap > self.max_time_gap_s:
            return None
        aoa_difference = self._angular_difference(a.aoa_deg, b.aoa_deg)
        pw_difference = abs(a.pulse_width_s - b.pulse_width_s)
        if aoa_difference > self.aoa_gate_deg or pw_difference > self.pulse_width_gate_s:
            return None
        base, left_harmonic, right_harmonic, pri_error = self._pri_family(
            a.pri_s, b.pri_s)
        if pri_error > self.pri_error_fraction:
            return None
        overlap_start = max(a.start_toa_s, b.start_toa_s)
        overlap_end = min(a.end_toa_s, b.end_toa_s)
        reference = (.5 * (overlap_start + overlap_end) if overlap_start <= overlap_end
                     else .5 * (a.end_toa_s + b.start_toa_s))
        left_epoch = (left.phase_epoch_s
            + round((reference - left.phase_epoch_s) / a.pri_s) * a.pri_s)
        right_epoch = (right.phase_epoch_s
            + round((reference - right.phase_epoch_s) / b.pri_s) * b.pri_s)
        phase_difference = left_epoch - right_epoch
        phase_residual = abs((phase_difference + base / 2.) % base - base / 2.)
        return TrainLink(a.train_id, b.train_id, base, left_harmonic,
            right_harmonic, pri_error, aoa_difference, pw_difference,
            phase_residual, phase_residual <= self.phase_tolerance_s)

    def link(self, trains) -> tuple[CrossRFTrainFamily, ...]:
        trains = tuple(trains)
        by_id = {train.observation.train_id: train for train in trains}
        if len(by_id) != len(trains):
            raise ValueError("Duplicate train IDs")
        adjacency = {train_id: set() for train_id in by_id}
        links = {}
        for index, left in enumerate(trains):
            for right in trains[index + 1:]:
                link = self.compare(left, right)
                if link is None:
                    continue
                key = tuple(sorted((link.left_train_id, link.right_train_id)))
                links[key] = link
                adjacency[key[0]].add(key[1])
                adjacency[key[1]].add(key[0])

        families, unseen = [], set(by_id)
        while unseen:
            seed = min(unseen)
            component, frontier = set(), [seed]
            while frontier:
                train_id = frontier.pop()
                if train_id in component:
                    continue
                component.add(train_id)
                frontier.extend(adjacency[train_id] - component)
            unseen -= component
            members = tuple(sorted(component))
            component_links = tuple(links[key] for key in sorted(links)
                                    if key[0] in component and key[1] in component)
            bases = sorted(link.base_pri_s for link in component_links)
            base_pri = bases[len(bases) // 2] if bases else by_id[members[0]].observation.pri_s
            phase_adjacency = {train_id: set() for train_id in members}
            for link in component_links:
                if link.phase_aligned:
                    phase_adjacency[link.left_train_id].add(link.right_train_id)
                    phase_adjacency[link.right_train_id].add(link.left_train_id)
            phase_clusters, phase_unseen = 0, set(members)
            while phase_unseen:
                phase_clusters += 1
                phase_frontier = [phase_unseen.pop()]
                while phase_frontier:
                    train_id = phase_frontier.pop()
                    neighbours = phase_adjacency[train_id] & phase_unseen
                    phase_unseen -= neighbours
                    phase_frontier.extend(neighbours)
            frequencies = tuple(sorted({by_id[train_id].observation.frequency_hz
                                        for train_id in members}))
            merged = (members,)
            separate = tuple((train_id,) for train_id in members)
            families.append(CrossRFTrainFamily(
                f"F{len(families) + 1}", members, frequencies, base_pri,
                phase_clusters, component_links, (merged, separate)))
        return tuple(families)
