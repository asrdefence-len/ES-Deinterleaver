"""Signal-pattern likelihoods for a first hybrid IMM-style emitter tracker.

No truth labels or physical-emitter association are used here. Scores are
relative log likelihoods, not calibrated probabilities.
"""
import math
import statistics

PRI_MODELS = ("constant", "change", "jitter", "stagger")
RF_MODELS = ("fixed", "switch", "hop")


def _log_gaussian(residual, sigma):
    sigma = max(float(sigma), 1e-12)
    z = residual / sigma
    return -0.5 * z * z - math.log(sigma)


def _normalized_scores(scores):
    peak = max(scores)
    weights = [math.exp(max(-700.0, x - peak)) for x in scores]
    total = sum(weights)
    return tuple(x / total for x in weights)


def markov_update(prior, likelihood_log, stay=0.94):
    """Predict with symmetric Markov transitions, then Bayesian update."""
    n = len(prior)
    if n != len(likelihood_log) or n < 2:
        raise ValueError("Model probability/likelihood dimension mismatch")
    if not 0 < stay < 1:
        raise ValueError("stay must be between zero and one")
    predicted = [stay * prior[j] + (1 - stay) / (n - 1) *
                 sum(prior[i] for i in range(n) if i != j)
                 for j in range(n)]
    return _normalized_scores([math.log(max(p, 1e-300)) + ll
                               for p, ll in zip(predicted, likelihood_log)])


def pri_evidence(observation, previous_pri=None):
    """Return evidence for constant, abrupt change, jitter, repeating stagger.

    Ordered intervals are essential for distinguishing jitter and stagger.
    A single mean PRI cannot identify these patterns. A conservative
    complexity penalty discourages fitting long patterns to noise.
    """
    values = observation.pri_intervals_s
    if not values:
        values = (observation.pri_s,)
    values = tuple(float(v) for v in values)
    mean = statistics.mean(values)
    sigma = max(observation.pri_sigma_s or 0, mean * 0.008, 1e-8)
    scatter = statistics.pstdev(values) if len(values) > 1 else 0.0
    baseline = previous_pri if previous_pri is not None else mean
    shift = abs(mean - baseline) / max(sigma, baseline * 0.015)
    # Common scale for *relative* evidence; intentionally not a calibrated
    # generative likelihood for all four competing models.
    constant = -0.5 * (scatter / sigma) ** 2 - 0.5 * shift ** 2
    changed = -0.5 * (scatter / sigma) ** 2 - 1.2
    jitter = -0.5 * (shift / 2) ** 2 - 0.8 - abs(scatter / max(mean, 1e-12) - 0.06) * 8
    stagger = -4.0
    if len(values) >= 8:
        for length in range(2, min(6, len(values) // 3) + 1):
            pattern = [statistics.mean(values[i::length]) for i in range(length)]
            residual = statistics.pstdev([v - pattern[i % length]
                                           for i, v in enumerate(values)])
            contrast = statistics.pstdev(pattern)
            if contrast > 3 * sigma:
                score = -0.5 * (residual / sigma) ** 2 - 0.30 * length
                stagger = max(stagger, score)
    return (constant, changed, jitter, stagger)


def frequency_evidence(observation, previous_frequency=None):
    values = observation.frequency_sequence_hz or (observation.frequency_hz,)
    values = tuple(float(v) for v in values)
    mean = statistics.mean(values)
    sigma = max(observation.frequency_sigma_hz or 0, 150_000.0)
    scatter = statistics.pstdev(values) if len(values) > 1 else 0.0
    shift = abs(mean - (previous_frequency if previous_frequency is not None else mean))
    fixed = -0.5 * (shift / sigma) ** 2 - 0.5 * (scatter / sigma) ** 2
    switched = -1.2 - 0.5 * (scatter / sigma) ** 2
    hopping = -2.5 if len(values) < 4 else (-0.5 if scatter > 3 * sigma else -3.0)
    return (fixed, switched, hopping)
