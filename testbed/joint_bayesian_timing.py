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
    accounted_indices: tuple = (-1, -1)

    @property
    def assignments(self):
        return history_tuples(self.history)[0]

    @property
    def indices(self):
        return history_tuples(self.history)[1]

    @property
    def last_emitter(self):
        # No PDW has been associated yet during an idle-only observation.
        return None if self.history is None else self.history.emitter


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
                 retained_mass=None, min_beam=1, detection_probability=None,
                 joint_detection_probability=None, joint_guard_s=5e-6,
                 observation_detection_probability=None, deadline_sigmas=3.):
        """initial: two (epoch,PRI,epoch_sigma,PRI_sigma) tuples.

        Epoch and PRI prior covariance initially assumed zero.
        """
        if len(initial) != 2 or beam_width < 1 or max_skip < 1:
            raise ValueError("Two tracks and positive beam/skip required")
        if retained_mass is not None and not 0 < retained_mass <= 1:
            raise ValueError("retained_mass must be in (0, 1]")
        if not 1 <= min_beam <= beam_width:
            raise ValueError("min_beam must be between 1 and beam_width")
        if detection_probability is not None and not 0 < detection_probability <= 1:
            raise ValueError("detection_probability must be in (0, 1]")
        if joint_detection_probability is not None and not 0 < joint_detection_probability <= 1:
            raise ValueError('joint_detection_probability must be in (0, 1]')
        if joint_detection_probability is not None and detection_probability is not None:
            raise ValueError('Choose either conditional or joint detection model')
        if joint_guard_s < 0 or not math.isfinite(joint_guard_s):
            raise ValueError('joint_guard_s must be finite and nonnegative')
        if observation_detection_probability is not None:
            if not 0 < observation_detection_probability < 1:
                raise ValueError("observation_detection_probability must be in (0,1)")
            if joint_detection_probability is not None or detection_probability is not None:
                raise ValueError("Observation model is mutually exclusive with legacy detection models")
        if not math.isfinite(deadline_sigmas) or deadline_sigmas < 0:
            raise ValueError("deadline_sigmas must be finite and nonnegative")
        self.observation_detection_probability = observation_detection_probability
        self.deadline_sigmas = deadline_sigmas
        self.observation_watermark = None
        self.joint_detection_probability = joint_detection_probability
        self.joint_guard_s = joint_guard_s
        self.detection_probability = detection_probability
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
        self.truth_diagnostic = None

    def process(self, toa_s, snr_db, diagnostic_truth=None):
        if not math.isfinite(toa_s):
            raise ValueError("TOA must be finite")
        if self.last_toa is not None and toa_s <= self.last_toa:
            raise ValueError("TOAs must increase")
        if self.observation_watermark is not None and toa_s <= self.observation_watermark:
            raise ValueError("PDW is not later than the committed observation watermark")
        r = toa_sigma(snr_db)**2
        branches = []
        truth_branch_details = [] if diagnostic_truth is not None else None
        truth_parent_details = [] if diagnostic_truth is not None else None
        if diagnostic_truth is not None:
            expected_assignments, expected_indices = map(tuple, diagnostic_truth)
            if not expected_assignments or len(expected_assignments) != len(expected_indices):
                raise ValueError("diagnostic_truth requires nonempty equal-length histories")
        for hypothesis in self.hypotheses:
            if truth_parent_details is not None:
                parent_a, parent_n = history_tuples(hypothesis.history)
                if parent_a == expected_assignments[:-1] and parent_n == expected_indices[:-1]:
                    e, true_n = expected_assignments[-1], expected_indices[-1]
                    track_true = hypothesis.tracks[e]
                    start_n = (hypothesis.accounted_indices[e]+1
                               if (self.joint_detection_probability is not None or
                                   self.observation_detection_probability is not None)
                               else track_true.index+1)
                    truth_parent_details.append({
                        "parent_weight": math.exp(hypothesis.log_weight),
                        "true_emitter": e, "true_index": true_n,
                        "last_detected_index": track_true.index,
                        "accounted_index": hypothesis.accounted_indices[e],
                        "next_permitted_index": start_n,
                        "already_accounted": true_n < start_n,
                        "beyond_skip_window": true_n >= start_n+self.max_skip,
                        "predicted_toa_s": track_true.epoch+true_n*track_true.pri,
                    })
            for emitter, track in enumerate(hypothesis.tracks):
                # Prior on the next integer index: uniform among allowed
                # skips. This is a deliberately simple conditional model,
                # not a receiver Pd or missed-detection model.
                start = (hypothesis.accounted_indices[emitter]+1
                         if (self.joint_detection_probability is not None or
                             self.observation_detection_probability is not None)
                         else track.index+1)
                for n in range(start,start+self.max_skip):
                    if self.detection_probability == 1. and n > start:
                        continue
                    outcome = update(track,toa_s,n,r)
                    if outcome is None:
                        continue
                    updated,ll = outcome
                    tracks = list(hypothesis.tracks)
                    tracks[emitter] = updated
                    accounted = hypothesis.accounted_indices
                    if self.observation_detection_probability is not None:
                        # Every intervening slot on this emitter is a miss.
                        # Slots already declared missed by advance_time() are
                        # excluded by the accounted-index start bound.
                        pd = self.observation_detection_probability
                        missed = n-accounted[emitter]-1
                        skip_log_prior = math.log(pd)
                        if missed:
                            skip_log_prior += missed*math.log1p(-pd)
                        next_accounted = list(accounted)
                        next_accounted[emitter] = n
                        accounted = tuple(next_accounted)
                    elif self.joint_detection_probability is not None:
                        pd = self.joint_detection_probability
                        other = 1-emitter
                        other_track = hypothesis.tracks[other]
                        cutoff = toa_s-self.joint_guard_s
                        # Account only for other-emitter slots safely before
                        # this observation; defer near-simultaneous slots.
                        due = math.ceil((cutoff-other_track.epoch)/other_track.pri)-1
                        other_accounted = max(accounted[other],due)
                        missed = (n-accounted[emitter]-1
                                  +other_accounted-accounted[other])
                        if missed < 0:
                            continue
                        skip_log_prior = math.log(pd)
                        if missed:
                            if pd == 1.:
                                continue
                            skip_log_prior += missed*math.log1p(-pd)
                        next_accounted = list(accounted)
                        next_accounted[emitter] = n
                        next_accounted[other] = other_accounted
                        accounted = tuple(next_accounted)
                    elif self.detection_probability is None:
                        skip_log_prior = -math.log(2*self.max_skip)
                    elif self.detection_probability == 1.:
                        skip_log_prior = -math.log(2.)
                    else:
                        gap = n-track.index
                        skip_log_prior = (math.log(self.detection_probability)
                                          +(gap-1)*math.log1p(-self.detection_probability)
                                          -math.log(2.))
                    if truth_branch_details is not None:
                        # Instrument the ACTUAL branch calculation, not an
                        # independently reimplemented diagnostic likelihood.
                        expected_assignments, expected_indices = diagnostic_truth
                        parent_assignments, parent_indices = history_tuples(hypothesis.history)
                        if (parent_assignments == tuple(expected_assignments[:-1]) and
                                parent_indices == tuple(expected_indices[:-1]) and
                                emitter == expected_assignments[-1] and
                                n == expected_indices[-1]):
                            truth_branch_details.append({
                                "timing_log_likelihood": ll,
                                "detection_log_likelihood": skip_log_prior,
                                "parent_log_weight": hypothesis.log_weight,
                                "missed_selected": (n-hypothesis.accounted_indices[emitter]-1)
                                    if self.observation_detection_probability is not None else None,
                                "accounted_before": hypothesis.accounted_indices,
                                "accounted_after": accounted,
                                "candidate_index": n,
                                "emitter": emitter,
                            })
                    branches.append(JointHypothesis(
                        tuple(tracks),
                        HistoryNode(hypothesis.history,emitter,n),
                        hypothesis.log_weight+ll+skip_log_prior,
                        accounted))
        if not branches:
            raise RuntimeError("No viable association hypotheses")
        branches.sort(key=lambda h:h.log_weight,reverse=True)
        if diagnostic_truth is not None:
            target_a, target_n = map(tuple, diagnostic_truth)
            truth_rank = next((i+1 for i,h in enumerate(branches)
                               if h.assignments == target_a and h.indices == target_n), None)
            truth_weight = (math.exp(branches[truth_rank-1].log_weight -
                                     logsumexp([h.log_weight for h in branches]))
                            if truth_rank is not None else None)
            self.truth_diagnostic = {
                "rank_before_pruning": truth_rank,
                "posterior_mass_before_pruning": truth_weight,
                "branch_terms": truth_branch_details,
                "matching_parent_count": len(truth_parent_details),
                "matching_parents": truth_parent_details[:3],
                "absence_reason": (
                    "generated" if truth_rank is not None else
                    "no_correct_parent" if not truth_parent_details else
                    "already_accounted" if all(p["already_accounted"] for p in truth_parent_details) else
                    "beyond_skip_window" if all(p["beyond_skip_window"] for p in truth_parent_details) else
                    "branch_not_generated_other_reason"),
                "candidate_count": len(branches),
            }
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

    def advance_time(self, watermark_s):
        """Commit nondetections up to a time watermark (experimental Stage A).

        Caller must first deliver ALL PDWs with TOA <= watermark_s. For each
        hypothesis and emitter, the next unaccounted pulse slot is declared
        missed only when its predicted mean TOA plus deadline_sigmas times
        its timing-state standard deviation is strictly before watermark_s.
        Each slot is penalised once via accounted_indices. A PDW with TOA
        <= the committed watermark is subsequently rejected.

        This is a hard Gaussian deadline approximation, not an integrated
        arrival-time/receiver likelihood. It assumes constant independent Pd,
        ideal pulse resolution, and no mode changes or clutter. The deadline
        is calculated using the current posterior state, not a smoothed
        historical state. Hypotheses can therefore have different deadlines.
        """
        pd = self.observation_detection_probability
        if pd is None:
            raise ValueError("Observation-time model is not enabled")
        if not math.isfinite(watermark_s):
            raise ValueError("Watermark must be finite")
        if self.observation_watermark is not None and watermark_s < self.observation_watermark:
            raise ValueError("Watermark cannot move backwards")
        if self.last_toa is not None and watermark_s < self.last_toa:
            raise ValueError("Watermark cannot precede the latest PDW")
        updated = []
        for h in self.hypotheses:
            accounted = list(h.accounted_indices)
            missed = 0
            for emitter, track in enumerate(h.tracks):
                n = accounted[emitter]+1
                while track.epoch+n*track.pri <= watermark_s:
                    variance = max(0., track.p00+2*n*track.p01+n*n*track.p11)
                    deadline = (track.epoch+n*track.pri+
                                self.deadline_sigmas*math.sqrt(variance))
                    if deadline >= watermark_s:
                        break
                    missed += 1
                    accounted[emitter] = n
                    n += 1
            updated.append(replace(h,accounted_indices=tuple(accounted),
                                   log_weight=h.log_weight+
                                   missed*math.log1p(-pd)))
        normalizer = logsumexp([h.log_weight for h in updated])
        self.hypotheses = [replace(h,log_weight=h.log_weight-normalizer)
                           for h in updated]
        self.observation_watermark = watermark_s
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
            "warning":"Beam-truncated posterior; no clutter, births, collisions; joint Pd uses deterministic slot times and a guard interval",
        }
