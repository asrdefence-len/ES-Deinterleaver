"""Reproducible Monte Carlo validation of constant versus jittered PRI.

Default: 100 trials x 6 detection probabilities x 2 signal types = 1200.
The reference estimator is deliberately unchanged; this script measures it.
Usage:
    python monte_carlo_pri.py --trials 100 --workers 4
    python monte_carlo_pri.py --trials 5 --workers 1
Outputs a CSV of every trial and a summary CSV. Seed is reproducible.
"""
import argparse
import csv
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

from testbed.complex_pri_scenarios import generate_observation
from testbed.pri_likelihood import fit_constant_jitter

DETECTION_PROBABILITIES = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50)
SIGNALS = ("constant", "jitter")


def run_trial(job):
    kind, pd, trial, pulses, assumed_pd = job
    obs, truth = generate_observation(
        kind, seed=trial + 10000, detection_probability=pd,
        toa_sigma_s=50e-9, count=pulses)
    started = time.perf_counter()
    fit = fit_constant_jitter(
        obs, detection_probability=assumed_pd, toa_sigma_s=50e-9)
    elapsed = time.perf_counter() - started
    winner = fit["winner"]
    best = fit["fits"][winner]
    delta_bic = fit["bic"]["jitter"] - fit["bic"]["constant"]
    return {
        "signal": kind, "actual_pd": pd, "assumed_pd": assumed_pd,
        "trial": trial, "transmitted_pulses": truth["transmitted_count"],
        "detected_pulses": truth["detected_count"],
        "winner": winner, "correct": int(winner == kind),
        "pri_estimate_us": best["pri_s"] * 1e6,
        "pri_error_us": (best["pri_s"] - 0.001) * 1e6,
        "jitter_estimate_us": best["jitter_sigma_s"] * 1e6,
        "delta_bic_jitter_minus_constant": delta_bic,
        "runtime_s": elapsed,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes, total, z=1.959963984540054):
    """95% Wilson score interval for independent Bernoulli outcomes."""
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1 + z*z/total
    centre = (p + z*z/(2*total))/denom
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return max(0, centre-half), min(1, centre+half)


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["signal"], row["actual_pd"])].append(row)
    summary = []
    for (kind, pd), group in sorted(groups.items()):
        n = len(group)
        successes = sum(r["correct"] for r in group)
        low, high = wilson_interval(successes, n)
        summary.append({
            "signal": kind, "actual_pd": pd, "trials": n,
            "correct": successes, "accuracy": successes/n,
            "accuracy_95ci_low": low, "accuracy_95ci_high": high,
            "false_classifications": n-successes,
            "mean_abs_pri_error_us": mean(abs(r["pri_error_us"]) for r in group),
            "mean_jitter_estimate_us": mean(r["jitter_estimate_us"] for r in group),
            "mean_runtime_s": mean(r["runtime_s"] for r in group),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1,
                        help="Process workers; increase to use more CPU cores")
    parser.add_argument("--pulses", type=int, default=90)
    parser.add_argument("--assumed-pd", type=float, default=None,
                        help="Override true Pd to test mismatch")
    parser.add_argument("--output", default="pri_monte_carlo")
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1 or args.pulses < 10:
        parser.error("Require trials>=1, workers>=1, pulses>=10")
    if args.assumed_pd is not None and not 0 < args.assumed_pd <= 1:
        parser.error("--assumed-pd must be in (0, 1]")
    jobs = [(kind, pd, trial, args.pulses,
             pd if args.assumed_pd is None else args.assumed_pd)
            for pd in DETECTION_PROBABILITIES
            for kind in SIGNALS
            for trial in range(args.trials)]
    print(f"Running {len(jobs)} Monte Carlo trials with {args.workers} workers",
          flush=True)
    rows = []
    start = time.perf_counter()
    if args.workers == 1:
        iterator = map(run_trial, jobs)
        for row in iterator:
            rows.append(row)
            if len(rows) % max(1, args.trials) == 0:
                print(f"  {len(rows)}/{len(jobs)} complete", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for row in executor.map(run_trial, jobs, chunksize=1):
                rows.append(row)
                if len(rows) % max(1, args.trials) == 0:
                    print(f"  {len(rows)}/{len(jobs)} complete", flush=True)
    summary = summarise(rows)
    write_csv(args.output + "_trials.csv", rows)
    write_csv(args.output + "_summary.csv", summary)
    print("\nSignal     Pd    Correct   Accuracy  95% Wilson CI   Mean |PRI error|")
    for r in summary:
        print(f"{r['signal']:<9} {r['actual_pd']:>4.0%} "
              f"{r['correct']:>4}/{r['trials']:<4} "
              f"{r['accuracy']:>7.1%} "
              f"[{r['accuracy_95ci_low']:.1%}, {r['accuracy_95ci_high']:.1%}] "
              f"{r['mean_abs_pri_error_us']:.2f} us")
    print(f"\nElapsed: {time.perf_counter()-start:.1f}s")
    print(f"Saved: {args.output}_trials.csv and {args.output}_summary.csv")
    print("Caution: same simulator family used in fitting and testing;")
    print("100 trials per cell is exploratory, not an operational validation.")


if __name__ == "__main__":
    main()
