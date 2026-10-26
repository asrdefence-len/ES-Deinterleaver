"""Experimental joint Bayesian association and timing tracking for TWO established emitters.

Each global hypothesis contains two Gaussian timing states (epoch, PRI), pulse
indices and a shared immutable association-history chain. Each PDW branches to either emitter and
possible next pulse slots. The measurement likelihood includes both timing
state covariance and SNR-dependent TOA variance. A bounded global beam
approximates the posterior. This is NOT blind acquisition: initial states
and their uncertainties are supplied. No clutter, collisions, births, mode
changes or explicit nondetection likelihood. PDWs must be time ordered.
"""
from dataclasses import dataclass, replace
import math
from two_emitter_snr_baseline import toa_sigma


@dataclass(frozen=True)
class Track:
    epoch: float
    pri: float
    p00: float
    p01: float
    p11: float
    index: int = -1


@dataclass(frozen=True)
class HistoryNode:
    """Shared immutable ancestry; one allocation per branch, not full tuple copies."""
    parent: object
    emitter: int
    index: int


def history_tuples(node):
    """Materialize histories only for diagnostics / external consumers."""
    assignments, indices = [], []
    while node is not None:
        assignments.append(node.emitter)
        indices.append(node.index)
        node = node.parent
    assignments.reverse()
    indices.reverse()
    return tuple(assignments), tuple(indices)


@dataclass(frozen=True)
class JointHypothesis:
    tracks: tuple
    history: object = None
    log_weight: float = 0.

    @property
    def assignments(self):
        return history_tuples(self.history)[0]

    @property
    def indices(self):
        return history_tuples(self.history)[1]

    @property
    def last_emitter(self):
        return self.history.emitter


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x-m) for x in xs))


def update(track, toa, n, measurement_variance):
    """Kalman update and normalized Gaussian predictive log likelihood."""
    h0, h1 = 1., float(n)
    s = (track.p00+2*h1*track.p01+h1*h1*track.p11+
         measurement_variance)
    if s <= 0:
        raise ValueError("Nonpositive innovation variance")
    residual = toa-(track.epoch+h1*track.pri)
    c0 = track.p00+h1*track.p01
    c1 = track.p01+h1*track.p11
    epoch = track.epoch+c0/s*residual
    pri = track.pri+c1/s*residual
    if pri <= 0:
        return None
    new = Track(epoch, pri,
                max(0.,track.p00-c0*c0/s),
                track.p01-c0*c1/s,
                max(0.,track.p11-c1*c1/s),n)
    ll = -.5*(math.log(2*math.pi*s)+residual*residual/s)
    return new,ll


class JointTimingTracker:
    def __init__(self, initial, beam_width=64, max_skip=12,
                 retained_mass=None, min_beam=1):
        """initial: two (epoch,PRI,epoch_sigma,PRI_sigma) tuples.

        Epoch and PRI prior covariance initially assumed zero.
        """
        if len(initial) != 2 or beam_width < 1 or max_skip < 1:
            raise ValueError("Two tracks and positive beam/skip required")
        if retained_mass is not None and not 0 < retained_mass <= 1:
            raise ValueError("retained_mass must be in (0, 1]")
        if not 1 <= min_beam <= beam_width:
            raise ValueError("min_beam must be between 1 and beam_width")
        self.retained_mass = retained_mass
        self.min_beam = min_beam
        self.pruning_diagnostics = {}
        tracks = []
        for epoch,pri,epoch_sigma,pri_sigma in initial:
            if pri <= 0 or epoch_sigma <= 0 or pri_sigma <= 0:
                raise ValueError("Positive PRI and prior sigmas required")
            tracks.append(Track(epoch,pri,epoch_sigma**2,0.,pri_sigma**2))
        self.hypotheses = [JointHypothesis(tuple(tracks))]
        self.beam_width = beam_width
        self.max_skip = max_skip
        self.last_toa = None

    def process(self, toa_s, snr_db):
        if not math.isfinite(toa_s):
            raise ValueError("TOA must be finite")
        if self.last_toa is not None and toa_s <= self.last_toa:
            raise ValueError("TOAs must increase")
        r = toa_sigma(snr_db)**2
        branches = []
        for hypothesis in self.hypotheses:
            for emitter, track in enumerate(hypothesis.tracks):
                # Prior on the next integer index: uniform among allowed
                # skips. This is a deliberately simple conditional model,
                # not a receiver Pd or missed-detection model.
                start = track.index+1
                for n in range(start,start+self.max_skip):
                    outcome = update(track,toa_s,n,r)
                    if outcome is None:
                        continue
                    updated,ll = outcome
                    tracks = list(hypothesis.tracks)
                    tracks[emitter] = updated
                    branches.append(JointHypothesis(
                        tuple(tracks),
                        HistoryNode(hypothesis.history,emitter,n),
                        hypothesis.log_weight+ll-math.log(2*self.max_skip)))
        if not branches:
            raise RuntimeError("No viable association hypotheses")
        branches.sort(key=lambda h:h.log_weight,reverse=True)
        capped = branches[:self.beam_width]
        full_norm = logsumexp([h.log_weight for h in branches])
        cap_norm = logsumexp([h.log_weight for h in capped])
        kept = capped
        if self.retained_mass is not None:
            cumulative = 0.
            count = 0
            for h in capped:
                cumulative += math.exp(h.log_weight-cap_norm)
                count += 1
                if count >= self.min_beam and cumulative >= self.retained_mass-1e-12:
                    break
            kept = capped[:count]
        retained_fraction = math.exp(
            logsumexp([h.log_weight for h in kept])-full_norm)
        self.pruning_diagnostics = {
            "candidate_count": len(branches),
            "cap_mass_before_renormalization": math.exp(cap_norm-full_norm),
            "retained_mass_before_renormalization": retained_fraction,
            "discarded_mass_before_renormalization": max(0.,1.-retained_fraction),
            "adaptive_count": len(kept),
        }
        norm = logsumexp([h.log_weight for h in kept])
        self.hypotheses = [replace(h,log_weight=h.log_weight-norm)
                           for h in kept]
        self.last_toa = toa_s
        return self.snapshot()

    def snapshot(self):
        weights = [math.exp(h.log_weight) for h in self.hypotheses]
        association = tuple(
            sum(w for w,h in zip(weights,self.hypotheses)
                if h.last_emitter == emitter)
            for emitter in range(2))
        best = self.hypotheses[0]
        return {
            "effective_hypothesis_count": 1./sum(w*w for w in weights),
            "pruning_diagnostics": dict(self.pruning_diagnostics),
            "association_probabilities":association,
            "best_assignments":best.assignments,
            "best_pulse_indices":best.indices,
            "best_tracks":best.tracks,
            "best_weight":weights[0],
            "retained_hypotheses":len(self.hypotheses),
            "warning":"Beam-truncated posterior; no clutter, births, or explicit Pd",
        }
