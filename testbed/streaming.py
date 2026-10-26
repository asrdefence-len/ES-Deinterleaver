"""Experimental incremental fixed-PRI emitter tracker.

Bootstrap from a bounded observation window using phase folding, then associate
new PDWs against persistent timing models. Ambiguous observations are withheld.
This is NOT yet a general frequency-agile, jittered-PRI or full-MHT tracker.
No simulator truth is consumed.
"""
from dataclasses import dataclass, field
import numpy as np
from testbed.phase_folding import PhaseFoldingDeinterleaver


@dataclass
class StreamTrack:
    track_id: str
    pri_s: float
    epoch_s: float
    frequency_hz: float
    aoa_deg: float
    pulse_width_s: float
    last_toa_s: float
    pdw_ids: list[int] = field(default_factory=list)
    last_slot: int = -1
    state: str = "CONFIRMED"


class StreamingDeinterleaver:
    def __init__(self, bootstrap_pdws=600, gate_s=0.5e-6,
                 frequency_gate_hz=500_000, aoa_gate_deg=8.0,
                 pw_gate_s=3e-6, max_buffer_pdws=2000):
        if bootstrap_pdws < 40 or gate_s <= 0 or max_buffer_pdws < bootstrap_pdws:
            raise ValueError("Invalid streaming configuration")
        self.bootstrap_pdws = bootstrap_pdws
        self.gate_s = gate_s
        self.frequency_gate_hz = frequency_gate_hz
        self.aoa_gate_deg = aoa_gate_deg
        self.pw_gate_s = pw_gate_s
        self.max_buffer_pdws = max_buffer_pdws
        self.tracks = {}
        self.assignments = {}
        self.ambiguous_pdw_ids = []
        self.unexplained_pdw_ids = []
        self._buffer = []
        self._last_input_toa = float("-inf")
        self._seen_ids = set()
        self.bootstrapped = False

    @staticmethod
    def _fit_timing(times, initial_pri):
        """Refine PRI and epoch from pulse indices; resist occasional outliers.

        A 30 ns PRI bias accumulates to 30 us after 1000 repetitions and
        breaks a 0.5 us association gate. Fit slope, not only phase.
        """
        times = np.asarray(times, dtype=float)
        origin = float(times[0])
        slots = np.rint((times - origin) / initial_pri)
        good = np.ones(len(times), dtype=bool)
        for _ in range(4):
            if np.count_nonzero(good) < 3:
                break
            slope, intercept = np.polyfit(slots[good], times[good], 1)
            residual = np.abs(times - (intercept + slots * slope))
            new_good = residual <= 0.5e-6
            if np.count_nonzero(new_good) < 3 or np.array_equal(new_good, good):
                break
            good = new_good
        slope, intercept = np.polyfit(slots[good], times[good], 1)
        return float(intercept), float(slope)

    def _bootstrap(self):
        finder = PhaseFoldingDeinterleaver(min_track_pdws=40)
        initial = finder.process(self._buffer)
        by_id = {p.pdw_id: p for p in self._buffer}
        for t in finder.tracks:
            observations = [by_id[i] for i in t.associated_pdw_ids]
            if not observations:
                continue
            epoch, pri = self._fit_timing(
                [p.toa_s for p in observations], t.pri_model.mean_s)
            self.tracks[t.track_id] = StreamTrack(
                t.track_id, pri, epoch,
                float(np.median([p.frequency_hz for p in observations])),
                float(np.median([p.aoa_deg for p in observations])),
                float(np.median([p.pulse_width_s for p in observations])),
                observations[-1].toa_s)
        # Seed with phase-folding assignments, preserving its strong batch
        # solution. New observations use conservative predictive association.
        for p in self._buffer:
            track_id = initial.get(p.pdw_id)
            if track_id not in self.tracks:
                self.unexplained_pdw_ids.append(p.pdw_id)
                continue
            t = self.tracks[track_id]
            slot = round((p.toa_s - t.epoch_s) / t.pri_s)
            if slot <= t.last_slot:
                self.unexplained_pdw_ids.append(p.pdw_id)
                continue
            self.assignments[p.pdw_id] = track_id
            t.pdw_ids.append(p.pdw_id)
            t.last_slot = slot
            t.last_toa_s = p.toa_s
        self._buffer.clear()
        self.bootstrapped = True

    def _associate(self, p):
        candidates = []
        for t in self.tracks.values():
            if abs(p.frequency_hz - t.frequency_hz) > self.frequency_gate_hz:
                continue
            if abs(p.aoa_deg - t.aoa_deg) > self.aoa_gate_deg:
                continue
            if abs(p.pulse_width_s - t.pulse_width_s) > self.pw_gate_s:
                continue
            slot = round((p.toa_s - t.epoch_s) / t.pri_s)
            if slot < 0 or slot <= t.last_slot:
                continue
            residual = abs(p.toa_s - (t.epoch_s + slot * t.pri_s))
            if residual <= self.gate_s:
                candidates.append((t, slot, residual))
        if len(candidates) == 1:
            t, slot, _ = candidates[0]
            t.pdw_ids.append(p.pdw_id)
            t.last_toa_s = p.toa_s
            t.last_slot = slot
            self.assignments[p.pdw_id] = t.track_id
            return t.track_id
        if len(candidates) > 1:
            self.ambiguous_pdw_ids.append(p.pdw_id)
        else:
            self.unexplained_pdw_ids.append(p.pdw_id)
        return None

    def process_chunk(self, pdws):
        """Accept ordered chunks; return NEW assignments made in this call.

        Initial calls return no assignments until the bootstrap window fills.
        The bootstrap call may return assignments for buffered earlier PDWs.
        """
        previous = set(self.assignments)
        for p in pdws:
            if p.toa_s < self._last_input_toa:
                raise ValueError("PDWs must arrive in nondecreasing TOA order")
            if p.pdw_id in self._seen_ids:
                raise ValueError("Duplicate PDW ID")
            self._seen_ids.add(p.pdw_id)
            self._last_input_toa = p.toa_s
            if not self.bootstrapped:
                self._buffer.append(p)
                if len(self._buffer) >= self.bootstrap_pdws:
                    self._bootstrap()
            else:
                self._associate(p)
        return {i: t for i, t in self.assignments.items() if i not in previous}

    def snapshot(self):
        return {
            "bootstrapped": self.bootstrapped,
            "buffered": len(self._buffer),
            "tracks": [{
                "track_id": t.track_id, "pri_s": t.pri_s,
                "frequency_hz": t.frequency_hz, "aoa_deg": t.aoa_deg,
                "pulse_width_s": t.pulse_width_s, "last_toa_s": t.last_toa_s,
                "associated_pdws": len(t.pdw_ids), "state": t.state
            } for t in self.tracks.values()],
            "ambiguous": len(self.ambiguous_pdw_ids),
            "unexplained": len(self.unexplained_pdw_ids),
        }
