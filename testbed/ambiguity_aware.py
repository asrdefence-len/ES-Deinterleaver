"""Conservative ambiguity-aware fixed-PRI association.

Phase folding discovers tracks. This stage retains its assignments except
where a PDW is timing-compatible with multiple established pulse trains.
Those observations are withheld (not reassigned). No simulator truth is used.
This is a batch research baseline, not a streaming tracker.
"""
from dataclasses import dataclass
import numpy as np
from testbed.phase_folding import PhaseFoldingDeinterleaver


@dataclass(frozen=True)
class AmbiguousObservation:
    pdw_id: int
    candidate_tracks: tuple[str, ...]


class AmbiguityAwareDeinterleaver:
    def __init__(self, gate_s=0.5e-6):
        if gate_s <= 0:
            raise ValueError("gate_s must be positive")
        self.gate_s = gate_s
        self.tracks = []
        self.ambiguous = []
        self.unassigned_pdw_ids = []

    @staticmethod
    def _model(toas, pri):
        """Estimate phase relative to first TOA; robust to a few wrong pulses."""
        origin = float(toas[0])
        slots = np.rint((toas - origin) / pri)
        epoch = float(np.median(toas - slots * pri))
        return epoch, pri

    def process(self, pdws):
        self.tracks = []
        self.ambiguous = []
        self.unassigned_pdw_ids = []
        if not pdws:
            return {}
        pdws = sorted(pdws, key=lambda p: p.toa_s)
        by_id = {p.pdw_id: p for p in pdws}
        finder = PhaseFoldingDeinterleaver()
        baseline = finder.process(pdws)
        self.tracks = finder.tracks
        models = {}
        for track in self.tracks:
            times = np.array([by_id[i].toa_s for i in track.associated_pdw_ids])
            models[track.track_id] = self._model(times, track.pri_model.mean_s)

        result = {}
        for p in pdws:
            compatible = []
            for track_id, (epoch, pri) in models.items():
                slot = round((p.toa_s - epoch) / pri)
                residual = abs(p.toa_s - (epoch + slot * pri))
                if residual <= self.gate_s:
                    compatible.append(track_id)
            if len(compatible) > 1:
                self.ambiguous.append(AmbiguousObservation(
                    p.pdw_id, tuple(sorted(compatible))))
            elif p.pdw_id in baseline:
                # Preserve the proven baseline decision whenever the timing
                # models do not provide positive evidence of ambiguity.
                result[p.pdw_id] = baseline[p.pdw_id]
            else:
                self.unassigned_pdw_ids.append(p.pdw_id)

        self.unassigned_pdw_ids.extend(item.pdw_id for item in self.ambiguous)
        for track in self.tracks:
            track.associated_pdw_ids = [
                p.pdw_id for p in pdws if result.get(p.pdw_id) == track.track_id
            ]
            track.last_toa_s = (by_id[track.associated_pdw_ids[-1]].toa_s
                                if track.associated_pdw_ids else None)
        return result
