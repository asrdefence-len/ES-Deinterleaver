"""Experimental joint pulse association and emitter-model tracking.

A bounded beam of *joint* association/state hypotheses processes individual
PDWs. Each association branches over timing models and missed-pulse counts;
new-source and clutter alternatives compete in the same beam. No simulator
identity is consulted. This is a proof-of-architecture, NOT a calibrated IMM
or a production MHT: birth/clutter priors and feature densities are provisional.
"""
from dataclasses import dataclass, replace
from math import exp, isfinite, log, pi, sqrt
from testbed.pri_likelihood import interval_log_likelihood


@dataclass(frozen=True)
class EmitterState:
    emitter_id: str
    last_toa_s: float
    pri_s: float
    jitter_s: float
    aoa_deg: float
    frequency_hz: float
    pulse_width_s: float
    pdw_ids: tuple
    model: str = "constant"


@dataclass(frozen=True)
class JointHypothesis:
    emitters: tuple = ()
    clutter_ids: tuple = ()
    log_weight: float = 0.0
    next_emitter_id: int = 1


class JointDeinterleavingTracker:
    """Beam-search joint estimator using PDWs with toa_s, frequency_hz,
    pulse_width_s, aoa_deg and pdw_id fields.

    Timing association uses missed-pulse marginal likelihoods; an explicit
    'change' branch can adopt a new PRI. Birth and clutter alternatives remain
    available. Log weights are *relative scores*; do not interpret the beam's
    normalized weights as calibrated posterior probabilities.
    """

    def __init__(self, beam_width=20, detection_probability=.95,
                 toa_sigma_s=50e-9, aoa_sigma_deg=1.5,
                 pw_sigma_s=.25e-6, rf_sigma_hz=25e6,
                 birth_score=-3.0, clutter_score=-8.0,
                 change_penalty=5.0, max_gap_s=.3):
        if beam_width < 1 or not 0 < detection_probability <= 1:
            raise ValueError("Invalid beam width or detection probability")
        self.beam_width = beam_width
        self.pd = detection_probability
        self.toa_sigma_s = toa_sigma_s
        self.aoa_sigma_deg = aoa_sigma_deg
        self.pw_sigma_s = pw_sigma_s
        self.rf_sigma_hz = rf_sigma_hz
        self.birth_score = birth_score
        self.clutter_score = clutter_score
        self.change_penalty = change_penalty
        self.max_gap_s = max_gap_s
        self.hypotheses = [JointHypothesis()]
        self._seen = set()
        self._last_toa = float("-inf")

    @staticmethod
    def _feature_score(p, e, aoa_sigma, pw_sigma, rf_sigma):
        # Relative dimensionless compatibility score, not a normalized PDF.
        da = abs(p.aoa_deg-e.aoa_deg)/aoa_sigma
        dp = abs(p.pulse_width_s-e.pulse_width_s)/pw_sigma
        df = abs(p.frequency_hz-e.frequency_hz)/rf_sigma
        if da > 4 or dp > 4:
            return None
        return -.5*(da*da+dp*dp+min(df*df, 16))

    def process(self, pdw):
        if pdw.pdw_id in self._seen:
            raise ValueError("Duplicate PDW ID")
        if pdw.toa_s < self._last_toa:
            raise ValueError("PDWs must be processed in TOA order")
        self._seen.add(pdw.pdw_id)
        self._last_toa = pdw.toa_s
        candidates = []
        for h in self.hypotheses:
            candidates.append(replace(
                h, clutter_ids=h.clutter_ids+(pdw.pdw_id,),
                log_weight=h.log_weight+self.clutter_score))
            eid = f"E{h.next_emitter_id}"
            new = EmitterState(eid, pdw.toa_s, 0.0, 0.0,
                               pdw.aoa_deg, pdw.frequency_hz,
                               pdw.pulse_width_s, (pdw.pdw_id,))
            candidates.append(replace(
                h, emitters=h.emitters+(new,),
                next_emitter_id=h.next_emitter_id+1,
                log_weight=h.log_weight+self.birth_score))
            for index, e in enumerate(h.emitters):
                dt = pdw.toa_s-e.last_toa_s
                if dt <= 0 or dt > self.max_gap_s:
                    continue
                fs = self._feature_score(
                    pdw, e, self.aoa_sigma_deg,
                    self.pw_sigma_s, self.rf_sigma_hz)
                if fs is None:
                    continue
                # A one-pulse track has no established PRI: seed a provisional
                # constant model with the first observed interval.
                if e.pri_s == 0:
                    branches = [(dt, 0.0, "constant", 0.0)]
                else:
                    branches = [(e.pri_s, 0.0, "constant", 0.0),
                                (e.pri_s, .075*e.pri_s, "jitter", -2.0)]
                    # A change hypothesis is penalized for flexibility.
                    # For missed detections, possible new PRIs are dt/k.
                    for k in range(1, min(6, max(1, int(dt/(.4*e.pri_s))))+1):
                        proposed = dt/k
                        if .4*e.pri_s <= proposed <= 2.5*e.pri_s:
                            branches.append((proposed, 0.0, "change",
                                             -self.change_penalty-log(k)))
                for pri, jitter, model, penalty in branches:
                    ll = interval_log_likelihood(
                        dt, pri, jitter,
                        toa_sigma_s=self.toa_sigma_s,
                        detection_probability=self.pd)
                    # Log density has units. Normalize by a common timing
                    # reference so the birth/association scores are comparable.
                    score = fs + ll + log(1e-3) + penalty
                    updated = replace(
                        e, last_toa_s=pdw.toa_s, pri_s=pri,
                        jitter_s=jitter, model=model,
                        pdw_ids=e.pdw_ids+(pdw.pdw_id,))
                    tracks = list(h.emitters)
                    tracks[index] = updated
                    candidates.append(replace(
                        h, emitters=tuple(tracks),
                        log_weight=h.log_weight+score))
        candidates.sort(key=lambda h: h.log_weight, reverse=True)
        # Deduplicate equivalent assignments before pruning, preventing the
        # same explanation from occupying the entire beam.
        unique = []
        signatures = set()
        for h in candidates:
            signature = (tuple((e.pdw_ids, e.model, round(e.pri_s, 10))
                               for e in h.emitters), h.clutter_ids)
            if signature in signatures:
                continue
            signatures.add(signature)
            unique.append(h)
            if len(unique) >= self.beam_width:
                break
        self.hypotheses = unique
        return self.best()

    def best(self):
        return self.hypotheses[0]

    def snapshot(self):
        best = self.best()
        return {
            "emitters": [
                {"id": e.emitter_id, "pdw_ids": e.pdw_ids,
                 "pri_s": e.pri_s, "model": e.model}
                for e in best.emitters],
            "clutter_ids": best.clutter_ids,
            "hypotheses_retained": len(self.hypotheses),
            "limitations": ("Relative hypothesis scores, provisional birth/"
                            "clutter penalties; not calibrated probabilities"),
        }
