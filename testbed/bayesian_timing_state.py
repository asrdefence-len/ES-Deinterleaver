"""Single-emitter Bayesian timing-state estimator with latent pulse indices.

Conjugate Gaussian regression for x=[epoch, PRI]. Each candidate integer
pulse-index increment k is weighted by Geometric(Pd) and the Gaussian
predictive TOA density. A bounded beam approximates the mixture posterior.

IMPORTANT: The first detected pulse is defined as index zero. The PRI
prior must be provided independently (e.g. from a coarse search). This
module does NOT discover an unknown PRI, handle clutter, or separate
emitters. Pruning makes its uncertainty approximate.
"""
from dataclasses import dataclass, replace
import math
from two_emitter_snr_baseline import toa_sigma


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x-m) for x in xs))


@dataclass(frozen=True)
class TimingHypothesis:
    index: int
    mean: tuple
    covariance: tuple  # (P00, P01, P11)
    log_weight: float
    indices: tuple = (0,)


class BayesianTimingEstimator:
    def __init__(self, initial_pri_s, pri_sigma_s, toa_sigma_s=50e-9,
                 detection_probability=.95, max_missing=12, beam_width=32):
        if initial_pri_s <= 0 or pri_sigma_s <= 0 or toa_sigma_s <= 0:
            raise ValueError("Positive PRI and uncertainties required")
        if not 0 < detection_probability <= 1 or max_missing < 1 or beam_width < 1:
            raise ValueError("Invalid detection or search configuration")
        self.initial_pri_s = initial_pri_s
        self.pri_sigma_s = pri_sigma_s
        self.sigma = toa_sigma_s
        self.pd = detection_probability
        self.max_missing = max_missing
        self.beam_width = beam_width
        self.hypotheses = []
        self._last_toa = None
        self.count = 0

    def process(self, toa_s, snr_db=None):
        # Backward compatible: without SNR, use the configured fixed sigma.
        # With SNR, use the illustrative receiver noise model per PDW.
        sigma = self.sigma if snr_db is None else toa_sigma(snr_db)
        if not math.isfinite(toa_s):
            raise ValueError("TOA must be finite")
        if self._last_toa is not None and toa_s <= self._last_toa:
            raise ValueError("TOAs must increase")
        if self._last_toa is None:
            # Anchor epoch to the first detection with measurement uncertainty.
            self.hypotheses = [TimingHypothesis(
                0, (toa_s, self.initial_pri_s),
                (sigma**2, 0.0, self.pri_sigma_s**2), 0.0)]
        else:
            candidates = []
            for h in self.hypotheses:
                p00, p01, p11 = h.covariance
                for k in range(1, self.max_missing+1):
                    if self.pd == 1 and k > 1:
                        break
                    n = h.index+k
                    # H=[1,n]. The innovation variance includes uncertainty
                    # in both epoch and PRI, and their cross-covariance.
                    s = p00+2*n*p01+n*n*p11+sigma**2
                    if s <= 0:
                        continue
                    innovation = toa_s-(h.mean[0]+n*h.mean[1])
                    ll = -.5*(innovation**2/s+math.log(2*math.pi*s))
                    prior = (math.log(self.pd)+(k-1)*math.log1p(-self.pd)
                             if self.pd < 1 else 0.0)
                    c0, c1 = p00+n*p01, p01+n*p11
                    gain0, gain1 = c0/s, c1/s
                    new_pri = h.mean[1]+gain1*innovation
                    if new_pri <= 0:
                        continue
                    # Exact covariance update for scalar Gaussian measurement.
                    new_p = (max(0.0, p00-c0*c0/s),
                             p01-c0*c1/s,
                             max(0.0, p11-c1*c1/s))
                    candidates.append(TimingHypothesis(
                        n, (h.mean[0]+gain0*innovation, new_pri),
                        new_p, h.log_weight+prior+ll,
                        h.indices+(n,)))
            if not candidates:
                raise RuntimeError("No viable timing hypotheses")
            # Preserve competing index histories; normalize AFTER pruning.
            candidates.sort(key=lambda h: h.log_weight, reverse=True)
            kept = candidates[:self.beam_width]
            norm = logsumexp([h.log_weight for h in kept])
            self.hypotheses = [replace(h, log_weight=h.log_weight-norm)
                               for h in kept]
        self._last_toa = toa_s
        self.count += 1
        self.last_measurement_sigma_s = sigma
        return self.snapshot()

    def snapshot(self):
        h = max(self.hypotheses, key=lambda h: h.log_weight)
        # Mixture moments include between-hypothesis PRI ambiguity.
        weights = [math.exp(v.log_weight) for v in self.hypotheses]
        total = sum(weights)
        weights = [w / total for w in weights]
        mixture_pri = sum(w*v.mean[1] for w, v in zip(weights, self.hypotheses))
        mixture_variance = sum(
            w*(v.covariance[2]+(v.mean[1]-mixture_pri)**2)
            for w, v in zip(weights, self.hypotheses))
        return {
            "mixture_pri_s": mixture_pri,
            "mixture_pri_sigma_s": math.sqrt(max(0.0, mixture_variance)),
            "mixture_hypotheses": len(self.hypotheses),
            "observations": self.count,
            "last_measurement_sigma_s": self.last_measurement_sigma_s,
            "pri_s": h.mean[1],
            "epoch_s": h.mean[0],
            "pri_sigma_s": math.sqrt(h.covariance[2]),
            "epoch_sigma_s": math.sqrt(h.covariance[0]),
            "pulse_indices": h.indices,
            "missed_pulses": h.index-(self.count-1),
            "hypotheses_retained": len(self.hypotheses),
            "best_hypothesis_weight": math.exp(h.log_weight),
            "note": "Beam-truncated conditional posterior; PRI prior supplied",
        }
