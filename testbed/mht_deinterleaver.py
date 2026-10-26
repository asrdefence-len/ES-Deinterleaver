"""Experimental bounded, AOA-assisted pulse-train MHT.

This is a *per-train* beam-search MHT, not a global joint-assignment MHT:
competing pulse-to-train histories are retained, but mutual exclusion across
trains is resolved conservatively. Bootstrap uses existing phase folding.
No persistent physical-emitter IDs or behaviour inference are assigned.
"""
from dataclasses import dataclass, field
from math import exp, log, pi, sqrt
from testbed.streaming import StreamingDeinterleaver


@dataclass
class Hypothesis:
    toa_s: float
    pri_s: float
    covariance: tuple
    score: float = 0.0
    pdw_ids: tuple = ()
    last_slot: int = 0
    missed: int = 0


@dataclass
class TrainHypotheses:
    train_id: str
    aoa_deg: float
    frequency_hz: float
    pulse_width_s: float
    hypotheses: list = field(default_factory=list)


class MHTDeinterleaver:
    """Bounded online pulse association after fixed-PRI bootstrap.

    The Kalman state is [predicted TOA, PRI]; hypotheses branch on a
    compatible detection versus a missed pulse. AOA is used as a
    probabilistic likelihood. Per-train top-K pruning bounds memory.
    """

    def __init__(self, bootstrap_pdws=600, max_hypotheses=8,
                 toa_sigma_s=0.15e-6, pri_process_sigma_s=0.02e-6,
                 aoa_sigma_deg=2.0, frequency_gate_hz=500_000,
                 pulse_width_gate_s=3e-6, gate_sigma=4.0,
                 detection_probability=0.95, max_missed=8):
        if (max_hypotheses < 1 or toa_sigma_s <= 0 or
                pri_process_sigma_s <= 0 or aoa_sigma_deg <= 0 or
                gate_sigma <= 0 or not 0 < detection_probability < 1 or
                max_missed < 0):
            raise ValueError("Invalid MHT configuration")
        self.bootstrap = StreamingDeinterleaver(bootstrap_pdws=bootstrap_pdws)
        self.max_hypotheses = max_hypotheses
        self.toa_var = toa_sigma_s ** 2
        self.pri_process_var = pri_process_sigma_s ** 2
        self.aoa_sigma = aoa_sigma_deg
        self.frequency_gate_hz = frequency_gate_hz
        self.pulse_width_gate_s = pulse_width_gate_s
        self.gate_sigma = gate_sigma
        self.pd = detection_probability
        self.max_missed = max_missed
        self.trains = {}
        self.assignments = {}
        self.ambiguous_pdw_ids = []
        self.unexplained_pdw_ids = []
        self._seen = set()
        self._last_toa = float("-inf")
        self._buffer = []
        self.bootstrapped = False

    def _initialize(self):
        for track in self.bootstrap.tracks.values():
            self.trains[track.track_id] = TrainHypotheses(
                track.track_id, track.aoa_deg, track.frequency_hz,
                track.pulse_width_s, [Hypothesis(
                    track.last_toa_s, track.pri_s,
                    (self.toa_var, 0.0, 0.0, self.pri_process_var),
                    pdw_ids=tuple(track.pdw_ids))])
        self.assignments.update(self.bootstrap.assignments)
        self.unexplained_pdw_ids.extend(self.bootstrap.unexplained_pdw_ids)
        self.ambiguous_pdw_ids.extend(self.bootstrap.ambiguous_pdw_ids)
        self._buffer.clear()
        self.bootstrapped = True

    def _predict(self, h, steps):
        p00, p01, p10, p11 = h.covariance
        # F=[[1,steps],[0,1]]; process noise on PRI.
        v00 = p00 + steps * (p01 + p10) + steps * steps * p11 + steps * steps * self.pri_process_var
        v01 = p01 + steps * p11
        v10 = p10 + steps * p11
        v11 = p11 + self.pri_process_var
        return h.toa_s + steps * h.pri_s, (v00, v01, v10, v11)

    def _update(self, h, toa, steps, predicted, cov):
        p00, p01, p10, p11 = cov
        s = p00 + self.toa_var
        k0, k1 = p00 / s, p10 / s
        residual = toa - predicted
        new_cov = ((1-k0)*p00, (1-k0)*p01,
                   p10-k1*p00, p11-k1*p01)
        return Hypothesis(
            predicted + k0*residual, h.pri_s + k1*residual,
            new_cov, h.score, h.pdw_ids, h.last_slot + steps, 0)

    def _process_one(self, p):
        candidates = []
        for train in self.trains.values():
            if (abs(p.frequency_hz-train.frequency_hz) > self.frequency_gate_hz
                    or abs(p.pulse_width_s-train.pulse_width_s) > self.pulse_width_gate_s):
                continue
            aoa_delta = (p.aoa_deg-train.aoa_deg+180) % 360-180
            if abs(aoa_delta) > self.gate_sigma*self.aoa_sigma:
                continue
            for h in train.hypotheses:
                if p.toa_s <= h.toa_s or h.pri_s <= 0:
                    continue
                # Consider plausible integer slots, including missed pulses.
                approx = max(1, round((p.toa_s-h.toa_s)/h.pri_s))
                for steps in range(max(1, approx-1), approx+2):
                    missed = steps-1
                    if missed > self.max_missed:
                        continue
                    predicted, cov = self._predict(h, steps)
                    residual = p.toa_s-predicted
                    s = cov[0]+self.toa_var
                    if abs(residual) > self.gate_sigma*sqrt(s):
                        continue
                    # Likelihood ratio relative to broad timing/AOA clutter.
                    timing = -0.5*(residual*residual/s+log(2*pi*s))
                    bearing = -0.5*(aoa_delta/self.aoa_sigma)**2
                    score = h.score + timing + bearing + log(self.pd) + missed*log(1-self.pd)
                    updated = self._update(h, p.toa_s, steps, predicted, cov)
                    updated.score = score
                    updated.pdw_ids = h.pdw_ids + (p.pdw_id,)
                    candidates.append((score, train.train_id, updated))
        if not candidates:
            self.unexplained_pdw_ids.append(p.pdw_id)
            return
        candidates.sort(key=lambda c: c[0], reverse=True)
        # Competing trains are not forcibly merged. Close competitors are
        # explicitly ambiguous, not arbitrarily assigned.
        if (len(candidates) > 1 and candidates[0][1] != candidates[1][1]
                and candidates[0][0]-candidates[1][0] < 2.0):
            self.ambiguous_pdw_ids.append(p.pdw_id)
            return
        _, chosen_id, chosen = candidates[0]
        train = self.trains[chosen_id]
        # Keep the old history as an alternative (missed detection), but
        # bound it by max_missed and the beam width.
        branches = [chosen]
        for h in train.hypotheses:
            missed_h = Hypothesis(h.toa_s, h.pri_s, h.covariance,
                                  h.score+log(1-self.pd), h.pdw_ids,
                                  h.last_slot, h.missed+1)
            if missed_h.missed <= self.max_missed:
                branches.append(missed_h)
        branches.sort(key=lambda h: h.score, reverse=True)
        train.hypotheses = branches[:self.max_hypotheses]
        self.assignments[p.pdw_id] = chosen_id

    def process_chunk(self, pdws):
        previous = set(self.assignments)
        for p in pdws:
            if p.pdw_id in self._seen or p.toa_s < self._last_toa:
                raise ValueError("Duplicate PDW ID or out-of-order TOA")
            self._seen.add(p.pdw_id)
            self._last_toa = p.toa_s
            if not self.bootstrapped:
                self._buffer.append(p)
                self.bootstrap.process_chunk([p])
                if self.bootstrap.bootstrapped:
                    self._initialize()
            else:
                self._process_one(p)
        return {i: t for i, t in self.assignments.items() if i not in previous}

    def snapshot(self):
        return {"bootstrapped": self.bootstrapped,
                "trains": len(self.trains),
                "hypotheses": sum(len(t.hypotheses) for t in self.trains.values()),
                "assigned": len(self.assignments),
                "ambiguous": len(self.ambiguous_pdw_ids),
                "unexplained": len(self.unexplained_pdw_ids)}
