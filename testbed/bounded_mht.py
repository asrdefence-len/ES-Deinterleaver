"""Experimental bounded multi-hypothesis reassociation of fixed-PRI pulse trains.

This is a batch proof of concept, NOT a streaming MHT or an overlap-resolving
receiver. Phase folding discovers tracks without access to simulator truth.
A top-K beam search revisits timing-ambiguous PDWs while preserving the
one-pulse-per-track-per-PRI-slot constraint.
"""
from dataclasses import dataclass
import numpy as np
from testbed.phase_folding import PhaseFoldingDeinterleaver


@dataclass(frozen=True)
class Candidate:
    track_id: str
    slot: int
    residual_s: float


class BoundedMHTDeinterleaver:
    def __init__(self, max_hypotheses=8, gate_s=0.5e-6,
                 ambiguity_margin_s=0.2e-6, unassigned_penalty=6.0):
        if max_hypotheses < 1 or gate_s <= 0:
            raise ValueError("Invalid hypothesis limit or timing gate")
        self.max_hypotheses = max_hypotheses
        self.gate_s = gate_s
        self.ambiguity_margin_s = ambiguity_margin_s
        self.unassigned_penalty = unassigned_penalty
        self.tracks = []
        self.ambiguous_pdw_ids = []
        self.hypotheses_retained = 0

    @staticmethod
    def _model(toas, pri):
        # Robust reference epoch. Circular phase offsets are estimated relative
        # to the first observed pulse to avoid wraparound at phase zero.
        origin = float(toas[0])
        cycles = np.rint((toas - origin) / pri)
        epoch = float(np.median(toas - cycles * pri))
        return epoch, pri

    def process(self, pdws):
        self.tracks = []
        self.ambiguous_pdw_ids = []
        self.hypotheses_retained = 0
        if not pdws:
            return {}
        pdws = sorted(pdws, key=lambda p: p.toa_s)
        by_id = {p.pdw_id: p for p in pdws}
        finder = PhaseFoldingDeinterleaver()
        initial = finder.process(pdws)
        self.tracks = finder.tracks
        models = {}
        for track in self.tracks:
            times = np.array([by_id[i].toa_s for i in track.associated_pdw_ids])
            models[track.track_id] = self._model(times, track.pri_model.mean_s)

        # All observed PDWs are re-evaluated; no simulator truth is used.
        candidates = {}
        for p in pdws:
            options = []
            for track_id, (epoch, pri) in models.items():
                slot = int(round((p.toa_s - epoch) / pri))
                residual = abs(p.toa_s - (epoch + slot * pri))
                if residual <= self.gate_s:
                    options.append(Candidate(track_id, slot, residual))
            options.sort(key=lambda x: x.residual_s)
            candidates[p.pdw_id] = options

        # Freeze uniquely compatible observations as reliable anchors, subject
        # to slot exclusivity. Conflict groups are handled by beam search.
        fixed = {}
        occupied = set()
        undecided = []
        for p in pdws:
            options = candidates[p.pdw_id]
            if len(options) == 1 and (options[0].track_id, options[0].slot) not in occupied:
                c = options[0]
                fixed[p.pdw_id] = c.track_id
                occupied.add((c.track_id, c.slot))
            else:
                undecided.append(p)
                if len(options) > 1:
                    self.ambiguous_pdw_ids.append(p.pdw_id)

        # Hypothesis: (negative-log score, assignments, occupied slots).
        beam = [(0.0, {}, frozenset(occupied))]
        for p in undecided:
            expanded = []
            for cost, assignment, slots in beam:
                options = [c for c in candidates[p.pdw_id]
                           if (c.track_id, c.slot) not in slots]
                for c in options:
                    new = dict(assignment)
                    new[p.pdw_id] = c.track_id
                    expanded.append((cost + (c.residual_s / self.gate_s) ** 2,
                                     new, slots | {(c.track_id, c.slot)}))
                # Allow a pulse to remain unassigned rather than invent a fit.
                expanded.append((cost + self.unassigned_penalty,
                                 assignment, slots))
            expanded.sort(key=lambda h: h[0])
            beam = expanded[:self.max_hypotheses]
        self.hypotheses_retained = len(beam)
        result = dict(fixed)
        if beam:
            result.update(beam[0][1])
        # Tracks describe final association, not the initial phase-folding fit.
        for track in self.tracks:
            track.associated_pdw_ids = [
                p.pdw_id for p in pdws if result.get(p.pdw_id) == track.track_id
            ]
            track.last_toa_s = (by_id[track.associated_pdw_ids[-1]].toa_s
                                if track.associated_pdw_ids else None)
        return result
