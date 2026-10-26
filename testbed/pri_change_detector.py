"""Conservative half-PRI change-point extension for established pulse trains.

Research prototype, NOT a general IMM or arbitrary-PRI change detector.
A fixed-PRI deinterleaver establishes trains; unassigned midpoint pulses
are held until repeated evidence supports a 2x-PRF mode. The extension
does not overwrite the original timing grid, allowing return to old PRI.
No simulator truth is consumed. Provisional pulses are withheld.
"""
from collections import defaultdict
from dataclasses import dataclass
from math import exp
from testbed.streaming import StreamingDeinterleaver
from testbed.mht_deinterleaver import MHTDeinterleaver


@dataclass
class ModeEvidence:
    pending: list
    last_slot: int = -1
    confirmed: bool = False


class PRIChangeDetector:
    def __init__(self, upstream="streaming", bootstrap_pdws=600,
                 confirmation_pulses=4, max_gap_slots=4,
                 timing_gate_s=0.3e-6, aoa_sigma_deg=1.5,
                 frequency_sigma_hz=100_000, pw_sigma_s=0.15e-6,
                 likelihood_margin=4.0):
        if upstream not in ("streaming", "mht"):
            raise ValueError("upstream must be streaming or mht")
        if (confirmation_pulses < 2 or max_gap_slots < 1 or
                min(timing_gate_s, aoa_sigma_deg, frequency_sigma_hz,
                    pw_sigma_s, likelihood_margin) <= 0):
            raise ValueError("Invalid change detector configuration")
        self.upstream = (StreamingDeinterleaver(bootstrap_pdws=bootstrap_pdws)
                         if upstream == "streaming" else
                         MHTDeinterleaver(bootstrap_pdws=bootstrap_pdws))
        self.confirmation_pulses = confirmation_pulses
        self.max_gap_slots = max_gap_slots
        self.timing_gate_s = timing_gate_s
        self.aoa_sigma_deg = aoa_sigma_deg
        self.frequency_sigma_hz = frequency_sigma_hz
        self.pw_sigma_s = pw_sigma_s
        self.likelihood_margin = likelihood_margin
        self.evidence = defaultdict(lambda: ModeEvidence([]))
        self.assignments = {}
        self.pending_ids = set()
        self.mode_events = []

    @property
    def bootstrapped(self):
        return self.upstream.bootstrapped

    def _tracks(self):
        if isinstance(self.upstream, StreamingDeinterleaver):
            return list(self.upstream.tracks.values())
        # The MHT bootstrap's train parameters provide a stable reference
        # for change detection; evolving MHT hypotheses remain independent.
        return list(self.upstream.trains.values())

    def _candidate(self, p):
        candidates = []
        for t in self._tracks():
            if isinstance(self.upstream, StreamingDeinterleaver):
                pri, epoch = t.pri_s, t.epoch_s
            else:
                original = self.upstream.bootstrap.tracks[t.train_id]
                pri, epoch = original.pri_s, original.epoch_s
            # A midpoint on the original PRI grid corresponds to the extra
            # pulse introduced when PRI halves.
            slot = round((p.toa_s-epoch)/pri - 0.5)
            if slot < 0:
                continue
            expected = epoch + (slot+0.5)*pri
            residual = p.toa_s-expected
            if abs(residual) > self.timing_gate_s:
                continue
            aoa = (p.aoa_deg-t.aoa_deg+180)%360-180
            df = (p.frequency_hz-t.frequency_hz)/self.frequency_sigma_hz
            dp = (p.pulse_width_s-t.pulse_width_s)/self.pw_sigma_s
            score = -0.5*((residual/self.timing_gate_s)**2
                          +(aoa/self.aoa_sigma_deg)**2+df**2+dp**2)
            candidates.append((score, t.track_id if hasattr(t, "track_id")
                               else t.train_id, slot))
        candidates.sort(reverse=True)
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[0][0]-candidates[1][0] < self.likelihood_margin:
            return None
        return candidates[0][1], candidates[0][2]

    def process_chunk(self, pdws):
        old = set(self.assignments)
        self.upstream.process_chunk(pdws)
        self.assignments.update(self.upstream.assignments)
        if not self.bootstrapped:
            return {}
        for p in pdws:
            if p.pdw_id in self.assignments:
                continue
            match = self._candidate(p)
            if match is None:
                continue
            train_id, slot = match
            state = self.evidence[train_id]
            if state.last_slot >= 0 and (slot <= state.last_slot
                                         or slot-state.last_slot > self.max_gap_slots):
                state.pending.clear()
                state.confirmed = False
            state.last_slot = slot
            state.pending.append((p.pdw_id, slot))
            self.pending_ids.add(p.pdw_id)
            if not state.confirmed and len(state.pending) >= self.confirmation_pulses:
                state.confirmed = True
                self.mode_events.append((train_id, p.toa_s, "half_pri_confirmed"))
            if state.confirmed:
                for pid, _ in state.pending:
                    self.assignments[pid] = train_id
                    self.pending_ids.discard(pid)
                state.pending.clear()
        return {pid: tid for pid, tid in self.assignments.items() if pid not in old}

    def snapshot(self):
        return dict(assigned=len(self.assignments),
                    pending=len(self.pending_ids),
                    mode_events=list(self.mode_events),
                    upstream=self.upstream.snapshot())
