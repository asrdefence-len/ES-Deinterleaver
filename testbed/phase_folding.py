"""Experimental batch fixed-PRI phase-folding deinterleaver.

Grid search finds coherent phase concentrations, then globally assigns PDWs.
No emitter count, scenario parameters, or truth IDs are supplied.
"""
import numpy as np
from testbed.pri_discovery import difference_histogram, candidate_peaks
from testbed.tracks import EmitterTrack, PRIType


class PhaseFoldingDeinterleaver:
    def __init__(self, tolerance_s=4e-6, min_track_pdws=40,
                 pri_min_s=0.0005, pri_max_s=0.0015,
                 coarse_step_s=0.2e-6, fine_step_s=0.01e-6):
        self.tolerance_s = tolerance_s
        self.min_track_pdws = min_track_pdws
        self.pri_min_s = pri_min_s
        self.pri_max_s = pri_max_s
        self.coarse_step_s = coarse_step_s
        self.fine_step_s = fine_step_s
        self.tracks = []

    @staticmethod
    def _residual(toas, pri, phase):
        return np.abs((toas - phase + pri / 2) % pri - pri / 2)

    def _best_phase(self, toas, pri):
        # Circular histogram of phases. Adjacent bins combined to avoid
        # missing a peak that straddles a bin boundary.
        width = self.tolerance_s
        count = max(8, int(np.ceil(pri / width)))
        phase = np.mod(toas, pri)
        bins = np.floor(phase / pri * count).astype(int) % count
        hist = np.bincount(bins, minlength=count)
        votes = np.roll(hist, 1) + hist + np.roll(hist, -1)
        centre = (int(np.argmax(votes)) + 0.5) * pri / count
        # Select phase from an observed pulse, then score exact tolerance.
        near = self._residual(phase, pri, centre) <= 2 * width
        if not np.any(near):
            return centre, 0
        phases = phase[near]
        # Circular mean avoids wrap discontinuity.
        angle = 2 * np.pi * phases / pri
        complex_mean = np.exp(1j * angle).mean()
        refined = (np.angle(complex_mean) % (2 * np.pi)) * pri / (2 * np.pi)
        count_matches = int(np.count_nonzero(self._residual(toas, pri, refined) <= width))
        return refined, count_matches

    def _search(self, toas, centre, span, step):
        grid = centre + np.arange(-span, span + step / 2, step)
        best = None
        for pri in grid:
            if pri < self.pri_min_s or pri > self.pri_max_s:
                continue
            phase, count = self._best_phase(toas, pri)
            if best is None or count > best[0]:
                best = (count, float(pri), float(phase))
        return best

    def process(self, pdws):
        self.tracks = []
        if not pdws:
            return {}
        pdws = sorted(pdws, key=lambda p: p.toa_s)
        toas = np.array([p.toa_s for p in pdws], dtype=float)
        hist = difference_histogram(pdws, self.pri_min_s, self.pri_max_s, 2e-6)
        peaks = candidate_peaks(hist, top_n=12, min_spacing_s=12e-6)
        # Fundamental candidates only; no scenario truth used.
        pri_seeds = [p.interval_s for p in peaks]
        available = np.ones(len(pdws), dtype=bool)
        assignments = {}
        while np.count_nonzero(available) >= self.min_track_pdws:
            active = toas[available]
            proposals = []
            for seed in pri_seeds:
                coarse = self._search(active, seed, 3e-6, self.coarse_step_s)
                if coarse is None:
                    continue
                fine = self._search(active, coarse[1], 0.25e-6, self.fine_step_s)
                if fine is not None:
                    proposals.append(fine)
            if not proposals:
                break
            proposals.sort(reverse=True)
            count, pri, phase = proposals[0]
            indices = np.flatnonzero(available & (self._residual(toas, pri, phase) <= self.tolerance_s))
            if len(indices) < self.min_track_pdws:
                break
            # A physical fixed-PRI train has at most one detected pulse per
            # repetition slot. Prevent unrelated coincident pulses inflating score.
            slots = np.rint((toas[indices] - phase) / pri).astype(np.int64)
            selected = {}
            for idx, slot in zip(indices, slots):
                error = float(self._residual(toas[idx:idx+1], pri, phase)[0])
                if slot not in selected or error < selected[slot][0]:
                    selected[slot] = (error, int(idx))
            indices = sorted(v[1] for v in selected.values())
            if len(indices) < self.min_track_pdws:
                break
            track = EmitterTrack(f"T{len(self.tracks)+1}")
            for idx in indices:
                track.add_pdw(pdws[idx])
                assignments[pdws[idx].pdw_id] = track.track_id
                available[idx] = False
            track.pri_model.kind = PRIType.FIXED
            track.pri_model.mean_s = pri
            track.pri_model.confidence = min(0.9, len(indices) / 100)
            self.tracks.append(track)
        return assignments
