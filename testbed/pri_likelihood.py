"""Missing-pulse-aware constant-PRI and jitter timing likelihoods.

For observed inter-detection interval d, marginalise over k transmitted
intervals. Under independent detections, P(k)=Pd*(1-Pd)**(k-1).
An explicit contamination component handles outliers without deleting data.
All scores are relative likelihoods conditional on the supplied Pd; do not
interpret posterior model weights as calibrated without validation.
"""
import math
import statistics


def _logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v-m) for v in values))


def _normal_logpdf(x, mean, sigma):
    return -0.5*((x-mean)/sigma)**2 - math.log(sigma) - 0.5*math.log(2*math.pi)


def interval_log_likelihood(interval_s, pri_s, jitter_sigma_s=0,
                            toa_sigma_s=50e-9, detection_probability=0.95,
                            max_missing=12, outlier_probability=0.01):
    """Likelihood of one interval, integrating unknown missed-pulse count.

    The contamination model is uniform over [0, max_missing*4*PRI].
    A Gaussian jitter model assumes independent per-transmission PRI errors.
    """
    if interval_s <= 0 or pri_s <= 0 or jitter_sigma_s < 0 or toa_sigma_s < 0:
        raise ValueError("Invalid timing parameters")
    if not 0 < detection_probability <= 1 or not 0 <= outlier_probability < 1:
        raise ValueError("Invalid detection/outlier probability")
    if max_missing < 1:
        raise ValueError("max_missing must be positive")
    pd = detection_probability
    terms = []
    for k in range(1, max_missing+1):
        prior = math.log(pd) + (k-1)*math.log1p(-pd) if pd < 1 else (0.0 if k == 1 else -math.inf)
        sigma = max(math.sqrt(k*jitter_sigma_s**2 + 2*toa_sigma_s**2), 1e-9)
        terms.append(prior + _normal_logpdf(interval_s, k*pri_s, sigma))
    # Truncation is explicit. For Pd>=0.7, tail at k>12 is negligible.
    nominal = _logsumexp(terms)
    if not outlier_probability:
        return nominal
    uniform_density = 1 / (max_missing*4*pri_s)
    return _logsumexp((math.log1p(-outlier_probability)+nominal,
                       math.log(outlier_probability*uniform_density)))


def fit_constant_jitter(observation, detection_probability=0.95,
                        toa_sigma_s=50e-9, outlier_probability=0.01,
                        max_missing=12):
    """Fit both hypotheses on a shared PRI grid, returning diagnostics.

    Grid search is deliberately bounded and slow: an experimental baseline,
    not a streaming implementation. A simple BIC complexity penalty is used.
    Missing-pulse count and contamination are marginalised, not hard-gated.
    """
    intervals = observation.pri_intervals_s
    if len(intervals) < 8:
        raise ValueError("Need at least eight measured intervals")
    values = sorted(intervals)
    # Lower quantile is resilient to missed detections (but not guaranteed
    # identifiable at very low Pd or with severe outliers).
    anchor = statistics.median(values[:max(4, len(values)//3)])
    pri_grid = [anchor*(0.75+0.005*i) for i in range(101)]
    jitter_grid = [0.0, 0.005, 0.015, 0.03, 0.05, 0.075, 0.10, 0.15]
    best = {"constant": (-math.inf, None, None),
            "jitter": (-math.inf, None, None)}
    for pri in pri_grid:
        for fraction in jitter_grid:
            sigma = fraction*pri
            ll = sum(interval_log_likelihood(
                v, pri, sigma, toa_sigma_s, detection_probability,
                max_missing, outlier_probability) for v in intervals)
            key = "constant" if fraction == 0 else "jitter"
            if ll > best[key][0]:
                best[key] = (ll, pri, sigma)
    n = len(intervals)
    # Constant fits PRI (1 parameter); jitter fits PRI and sigma (2).
    bic = {name: -2*row[0]+(1 if name=="constant" else 2)*math.log(n)
           for name, row in best.items()}
    winner = min(bic, key=bic.get)
    return {"winner": winner, "bic": bic,
            "fits": {name: {"log_likelihood": row[0], "pri_s": row[1],
                            "jitter_sigma_s": row[2]} for name, row in best.items()},
            "outlier_probability": outlier_probability,
            "detection_probability": detection_probability,
            "note": "Conditional likelihoods; Pd and contamination fraction assumed"}
