"""Monte Carlo coverage study for conditional Bayesian timing-state estimator.

Run: python monte_carlo_timing_coverage.py --trials 1000
Quick smoke test: python monte_carlo_timing_coverage.py --trials 20
Outputs CSV files; report empirical coverage, bias, RMSE and index recovery.
The estimator is conditional on an externally supplied PRI prior. A biased
prior mean is not automatically a calibrated random draw from that prior.
"""
import argparse
import csv
import math
import random
from collections import defaultdict

from testbed.bayesian_timing_state import BayesianTimingEstimator


def one_trial(pd, trial, initial_error_fraction, pulses=100,
              true_pri_s=.001, toa_sigma_s=50e-9, prior_sigma_fraction=.01):
    rng = random.Random(17011 + trial * 7919 + round(pd*1000)*17
                        + round(initial_error_fraction*10000)*97)
    initial_pri = true_pri_s * (1 + initial_error_fraction)
    est = BayesianTimingEstimator(
        initial_pri_s=initial_pri,
        pri_sigma_s=true_pri_s*prior_sigma_fraction,
        toa_sigma_s=toa_sigma_s,
        detection_probability=pd)
    truth_indices = []
    for i in range(pulses):
        if i == 0 or rng.random() < pd:
            truth_indices.append(i)
            est.process(i*true_pri_s+rng.gauss(0, toa_sigma_s))
    result = est.snapshot()
    err = result["pri_s"]-true_pri_s
    sigma = result["pri_sigma_s"]
    z = abs(err)/sigma if sigma > 0 else math.inf
    return {
        "pd": pd, "trial": trial, "initial_error_pct": 100*initial_error_fraction,
        "detections": len(truth_indices),
        "pri_error_ns": err*1e9, "reported_sigma_ns": sigma*1e9,
        "z_abs": z, "inside_1sigma": int(z<=1),
        "inside_2sigma": int(z<=2), "inside_3sigma": int(z<=3),
        "index_sequence_correct": int(
            result["pulse_indices"] == tuple(truth_indices)),
        "missed_count_correct": int(
            result["missed_pulses"] ==
            truth_indices[-1]+1-len(truth_indices)),
        "hypotheses_retained": result["hypotheses_retained"],
    }


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["pd"], row["initial_error_pct"])].append(row)
    summary = []
    for (pd, initial_error), trials in sorted(groups.items()):
        n = len(trials)
        errors = [r["pri_error_ns"] for r in trials]
        summary.append({
            "pd": pd, "initial_error_pct": initial_error, "trials": n,
            "mean_error_ns": sum(errors)/n,
            "rmse_ns": math.sqrt(sum(e*e for e in errors)/n),
            "mean_reported_sigma_ns": sum(r["reported_sigma_ns"]
                                          for r in trials)/n,
            "coverage_1sigma": sum(r["inside_1sigma"] for r in trials)/n,
            "coverage_2sigma": sum(r["inside_2sigma"] for r in trials)/n,
            "coverage_3sigma": sum(r["inside_3sigma"] for r in trials)/n,
            "index_sequence_accuracy": sum(r["index_sequence_correct"]
                                           for r in trials)/n,
            "missed_count_accuracy": sum(r["missed_count_correct"]
                                         for r in trials)/n,
        })
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--pulses", type=int, default=100)
    parser.add_argument("--output", default="timing_coverage")
    parser.add_argument("--prior-sigma-pct", type=float, default=1.0)
    args = parser.parse_args()
    if args.trials < 1 or args.pulses < 2 or args.prior_sigma_pct <= 0:
        parser.error("Require positive trials/prior sigma and >=2 pulses")
    rows = [
        one_trial(pd, trial, offset, pulses=args.pulses,
                  prior_sigma_fraction=args.prior_sigma_pct/100)
        for pd in (1.0, .95, .70, .50)
        for offset in (0.0, .002, .01, .03)
        for trial in range(args.trials)
    ]
    summary = summarise(rows)
    write_csv(args.output+"_trials.csv", rows)
    write_csv(args.output+"_summary.csv", summary)
    print("Coverage targets: 1sigma=68.3%, 2sigma=95.4%, 3sigma=99.7%")
    print("Pd    prior bias  trials  RMSE(ns)  sigma(ns)  1sigma  2sigma  3sigma  index")
    for s in summary:
        print(f"{s['pd']:.0%}  {s['initial_error_pct']:7.1f}%"
              f"  {s['trials']:5d}  {s['rmse_ns']:8.2f}"
              f"  {s['mean_reported_sigma_ns']:9.2f}"
              f"  {s['coverage_1sigma']:6.1%}"
              f"  {s['coverage_2sigma']:6.1%}"
              f"  {s['coverage_3sigma']:6.1%}"
              f"  {s['index_sequence_accuracy']:6.1%}")
    print(f"Saved {args.output}_trials.csv and {args.output}_summary.csv")
    print("WARNING: beam pruning and conditioning on first detection make "
          "reported posterior uncertainty approximate; fixed prior bias "
          "also violates the generative prior calibration assumption.")


if __name__ == "__main__":
    main()
