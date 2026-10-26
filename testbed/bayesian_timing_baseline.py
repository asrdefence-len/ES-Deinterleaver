"""Constant-PRI Bayesian timing baseline (single established emitter).

Generative model: t_i = epoch + n_i*PRI + eps_i, eps_i iid N(0,sigma^2).
Each transmitted pulse is independently detected with probability Pd, so
increments in hidden pulse index follow Geometric(Pd). Parameters epoch and
PRI are conditioned on (known) here: this module validates the *association
likelihood* before fitting states or introducing multiple emitters.

The likelihood sums over missed-pulse counts via a forward recursion on
integer pulse indices. Unlike independent interval likelihoods, this models
shared TOA measurement noise correctly. No clutter/birth model is implied.
"""
import math


def _logsumexp(values):
    if not values:
        return -math.inf
    m = max(values)
    return m + math.log(sum(math.exp(v-m) for v in values)) if math.isfinite(m) else m


def _normal_logpdf(x, mean, sigma):
    return -.5*((x-mean)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi))


def toa_sequence_log_likelihood(toas, epoch_s, pri_s, toa_sigma_s=50e-9,
                                detection_probability=.95, max_missing=12):
    """Log likelihood conditional on first detection being slot zero.

    epoch_s is the expected TOA of the first detected pulse; subsequent
    hidden index increments k>=1 have Geometric(Pd) priors. This conditions
    on the first pulse being detected and excludes clutter and births.
    Absolute TOAs are used, avoiding correlated inter-detection noise.
    """
    if pri_s <= 0 or toa_sigma_s <= 0 or not 0 < detection_probability <= 1:
        raise ValueError("Invalid timing model")
    if max_missing < 1:
        raise ValueError("max_missing must be >= 1")
    if not toas:
        raise ValueError("Need at least one TOA")
    if any(b <= a for a, b in zip(toas, toas[1:])):
        raise ValueError("TOAs must be strictly increasing")
    pd = detection_probability
    previous = {0: _normal_logpdf(toas[0], epoch_s, toa_sigma_s)}
    for observed in toas[1:]:
        current = {}
        # Prune only states that are incompatible at > 8 sigma, to keep
        # computational cost bounded without introducing score heuristics.
        low = max(1, math.floor((observed-epoch_s-8*toa_sigma_s)/pri_s))
        high = math.ceil((observed-epoch_s+8*toa_sigma_s)/pri_s)
        for n in range(low, high+1):
            terms = []
            for k in range(1, max_missing+1):
                prior = previous.get(n-k)
                if prior is None:
                    continue
                log_transition = ((math.log(pd)+(k-1)*math.log1p(-pd))
                                  if pd < 1 else (0.0 if k == 1 else -math.inf))
                terms.append(prior+log_transition)
            if terms:
                current[n] = (_logsumexp(terms) +
                              _normal_logpdf(observed, epoch_s+n*pri_s,
                                             toa_sigma_s))
        if not current:
            return -math.inf
        previous = current
    return _logsumexp(list(previous.values()))
