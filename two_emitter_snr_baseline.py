"""Two established emitters: SNR-aware, probabilistic timing-only association.

Oracle baseline: epochs and PRIs are known; each detected pulse is resolved.
No clutter, track updates, births, or receiver collision model.
"""
import argparse
import math
import random


def toa_sigma(snr_db):
    """Illustrative, uncalibrated timing model: 50 ns at 20 dB plus 10 ns floor."""
    if not math.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    return math.hypot(10e-9, 50e-9 * 10**((20-snr_db)/20))


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(x-m) for x in values))


def association(toa, snr_db, models, prior=(.5, .5)):
    """Posterior P(emitter | measured TOA,SNR), given two candidate emitters."""
    if len(models) != 2 or len(prior) != 2 or any(p <= 0 for p in prior):
        raise ValueError("Two emitter models and positive priors required")
    sigma = toa_sigma(snr_db)
    scores = []
    for model, p in zip(models, prior):
        epoch, pri, count = model
        if pri <= 0 or count < 1:
            raise ValueError("Invalid pulse model")
        # Clamp to valid slots: a pulse may fall outside another emitter's
        # finite observation window. Its likelihood is small, not undefined.
        center = max(0, min(count-1, round((toa-epoch)/pri)))
        # Local summation is accurate when timing uncertainty << PRI.
        slots = range(max(0, center-3), min(count-1, center+3)+1)
        terms = [-0.5*((toa-epoch-n*pri)/sigma)**2
                 -math.log(sigma*math.sqrt(2*math.pi))
                 for n in slots]
        scores.append(math.log(p) + logsumexp(terms) - math.log(count))
    norm = logsumexp(scores)
    return tuple(math.exp(x-norm) for x in scores)


def experiment(trials=100, pd=.95, snr_a=20, snr_b=20):
    """Known-model association; missed pulses are generated, not tracked."""
    if trials < 1 or not 0 < pd <= 1:
        raise ValueError("Invalid trials or Pd")
    models = [(0.0, .001, 80), (.00012, .00075, 80)]
    total = correct = ambiguous = 0
    brier = 0.
    for trial in range(trials):
        rng = random.Random(91837 + trial*1009)
        detections = []
        for emitter, snr in enumerate((snr_a, snr_b)):
            epoch, pri, count = models[emitter]
            for n in range(count):
                if rng.random() < pd:
                    detections.append((epoch+n*pri+rng.gauss(0, toa_sigma(snr)),
                                       snr, emitter))
        for toa, snr, truth in sorted(detections):
            probs = association(toa, snr, models)
            total += 1
            correct += int((probs[1] > probs[0]) == bool(truth))
            ambiguous += int(max(probs) < .9)
            brier += (probs[1]-truth)**2
    return dict(trials=trials, pd=pd, snr_a=snr_a, snr_b=snr_b,
                detected=total, accuracy=correct/total,
                ambiguous_fraction=ambiguous/total, brier=brier/total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--pd", type=float, default=.95)
    args = parser.parse_args()
    for snrs in [(20,20), (30,30), (10,10), (30,10), (10,30)]:
        print(experiment(args.trials, args.pd, *snrs))


if __name__ == "__main__":
    main()
