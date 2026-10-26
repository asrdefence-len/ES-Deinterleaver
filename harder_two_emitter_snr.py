"""Harder two-emitter association: crossings, coincidence, and uncertain timing.

This remains a controlled association experiment, NOT a joint tracker:
models are fixed for each run; priors are sampled per trial and are not
updated from the observations. Coincident pulses are individually resolved.
"""
import argparse
import math
import random
from two_emitter_snr_baseline import association, toa_sigma

SCENARIOS = {
    "coincident": ((0.0, .001, 80), (0.0, .00075, 80)),
    "close_pri": ((0.0, .001, 80), (0.0, .00099, 80)),
    "same_pri": ((0.0, .001, 80), (5e-6, .001, 80)),
}


def run(scenario, trials=100, pd=.95, snr_db=10.,
        pri_prior_sigma_s=0., epoch_prior_sigma_s=0., seed=19071):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    if trials < 1 or not 0 < pd <= 1:
        raise ValueError("Invalid trials/Pd")
    if pri_prior_sigma_s < 0 or epoch_prior_sigma_s < 0:
        raise ValueError("Negative prior uncertainty")
    truth_models = SCENARIOS[scenario]
    total = correct = ambiguous = 0
    brier = 0.
    bins = [[0, 0., 0.] for _ in range(10)]
    # Each trial uses a distinct, perturbed model, held fixed over the CPI.
    for trial in range(trials):
        rng = random.Random(seed + 104729*trial)
        models = [
            (epoch+rng.gauss(0, epoch_prior_sigma_s),
             pri+rng.gauss(0, pri_prior_sigma_s), count)
            for epoch, pri, count in truth_models
        ]
        detections = []
        for emitter, (epoch, pri, count) in enumerate(truth_models):
            for n in range(count):
                if rng.random() < pd:
                    detections.append((epoch+n*pri+rng.gauss(0,toa_sigma(snr_db)),
                                       emitter))
        for toa, truth in sorted(detections):
            p = association(toa, snr_db, models)
            # Probability of emitter B; ties assigned to A.
            prob = p[1]
            total += 1
            correct += int((prob > .5) == bool(truth))
            ambiguous += int(max(p) < .9)
            brier += (prob-truth)**2
            bucket = min(9, int(prob*10))
            bins[bucket][0] += 1
            bins[bucket][1] += prob
            bins[bucket][2] += truth
    calibration = [
        {"probability_bin":f"{i/10:.1f}-{(i+1)/10:.1f}",
         "n":v[0], "mean_probability":v[1]/v[0],
         "observed_frequency":v[2]/v[0]}
        for i,v in enumerate(bins) if v[0]
    ]
    return {
        "scenario":scenario, "trials":trials, "pd":pd, "snr_db":snr_db,
        "pri_prior_sigma_us":pri_prior_sigma_s*1e6,
        "epoch_prior_sigma_us":epoch_prior_sigma_s*1e6,
        "pdws":total, "accuracy":correct/total,
        "ambiguous_fraction":ambiguous/total, "brier":brier/total,
        "calibration":calibration,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=100)
    parser.add_argument("--pd",type=float,default=.95)
    parser.add_argument("--snr",type=float,default=10.)
    args=parser.parse_args()
    for scenario in SCENARIOS:
        for pri_sigma, epoch_sigma in ((0.,0.), (0.2e-6,1e-6),
                                        (2e-6,5e-6)):
            r=run(scenario,args.trials,args.pd,args.snr,
                  pri_sigma,epoch_sigma)
            print({k:v for k,v in r.items() if k!="calibration"})
            print("  calibration:",r["calibration"])


if __name__=="__main__":
    main()
