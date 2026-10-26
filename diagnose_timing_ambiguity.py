"""Reproduce and diagnose pulse-index ambiguity, especially low-Pd outliers.

Run: python diagnose_timing_ambiguity.py --trials 20
"""
import argparse
import math
import random

from testbed.bayesian_timing_state import BayesianTimingEstimator


def run(pd, trial, offset, beam, pulses=100):
    rng = random.Random(17011 + trial*7919 + round(pd*1000)*17
                        + round(offset*10000)*97)
    est = BayesianTimingEstimator(.001*(1+offset), .001*.01,
                                  detection_probability=pd, beam_width=beam)
    truth = []
    first_lost = None
    for i in range(pulses):
        if i == 0 or rng.random() < pd:
            truth.append(i)
            est.process(i*.001+rng.gauss(0, 50e-9))
            if first_lost is None and not any(
                h.indices == tuple(truth) for h in est.hypotheses):
                first_lost = (i, len(truth))
    snap = est.snapshot()
    error_ns = (snap["pri_s"]-.001)*1e9
    mix_error_ns = (snap["mixture_pri_s"]-.001)*1e9
    return dict(trial=trial, beam=beam, pd=pd, offset=offset,
                error_ns=error_ns, map_sigma_ns=snap["pri_sigma_s"]*1e9,
                mixture_error_ns=mix_error_ns,
                mixture_sigma_ns=snap["mixture_pri_sigma_s"]*1e9,
                correct=snap["pulse_indices"]==tuple(truth),
                true_history_retained=any(
                    h.indices==tuple(truth) for h in est.hypotheses),
                first_truth_pruned=first_lost,
                map_weight=snap["best_hypothesis_weight"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pd", type=float, default=.5)
    parser.add_argument("--offset", type=float, default=.002)
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.pd <= 1:
        parser.error("Invalid trials or Pd")
    failures = []
    for trial in range(args.trials):
        result = run(args.pd, trial, args.offset, 32)
        if not result["correct"]:
            failures.append(trial)
            print("Failure at beam=32:", result)
    print(f"Pd={args.pd:.0%}, offset={args.offset:.2%}, "
          f"failures={len(failures)}/{args.trials}")
    for trial in failures:
        print(f"Beam-width sensitivity, trial {trial}:")
        for beam in (32, 64, 128, 256):
            print(run(args.pd, trial, args.offset, beam))
    print("Note: mixture variance accounts only for RETAINED hypotheses. "
          "If the truth was pruned, increasing uncertainty post hoc cannot "
          "restore it. A wrong MAP history may also have high posterior weight.")


if __name__ == "__main__":
    main()
