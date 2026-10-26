"""Repeatable frequency-separation sweep for the baseline deinterleaver."""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

from testbed.generator import generate
from testbed.deinterleavers import FrequencyGateDeinterleaver
from testbed.evaluation import evaluate


def sweep(base, separations_hz, seeds, gate_hz):
    if len(base["emitters"]) != 3:
        raise ValueError("Frequency sweep currently requires exactly three emitters")
    results = []
    centre = float(base["emitters"][1]["frequency_hz"])
    for separation in separations_hz:
        for seed in range(seeds):
            scenario = deepcopy(base)
            scenario["seed"] = seed
            for emitter, offset in zip(scenario["emitters"], (-1, 0, 1)):
                emitter["frequency_hz"] = centre + offset * separation
            pdws, truth = generate(scenario)
            assignments = FrequencyGateDeinterleaver(gate_hz).process(pdws)
            scores = evaluate(truth, assignments)
            # Purity can be misleadingly perfect for fragmented tracks.
            # The one-to-one score penalises both merging and fragmentation.
            one_to_one = optimal_one_to_one_accuracy(truth, assignments)
            results.append({
                "separation_hz": separation,
                "seed": seed,
                "pdws": len(pdws),
                "tracks": scores["estimated_tracks"],
                "purity": scores["track_purity"],
                "mean_completeness": sum(scores["emitter_completeness"].values()) / len(scores["emitter_completeness"]),
                "one_to_one_accuracy": one_to_one,
            })
    return results


def optimal_one_to_one_accuracy(truth, assignments):
    """Maximum correctly attributed PDWs under a unique truth-to-track mapping."""
    from collections import Counter
    from functools import lru_cache
    true_ids = sorted(set(truth.values()))
    track_ids = sorted(set(assignments.values()))
    if not truth:
        return 0.0
    counts = Counter((truth[i], assignments[i]) for i in truth)
    # Exact dynamic programming over smaller side; safe for this three-emitter sweep.
    if len(true_ids) <= len(track_ids):
        smaller, larger = true_ids, track_ids
        score = lambda a, b: counts[(a, b)]
    else:
        smaller, larger = track_ids, true_ids
        score = lambda a, b: counts[(b, a)]
    @lru_cache(None)
    def best(index, used):
        if index == len(smaller):
            return 0
        return max(
            [best(index + 1, used)] +
            [score(smaller[index], larger[j]) + best(index + 1, used | (1 << j))
             for j in range(len(larger)) if not (used & (1 << j))]
        )
    return best(0, 0) / len(truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/baseline.json")
    parser.add_argument("--output", default="results/frequency_sweep")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--gate-hz", type=float, default=500_000)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    base = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    separations = [0, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
    rows = sweep(base, separations, args.seeds, args.gate_hz)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = [sum(row["one_to_one_accuracy"] for row in rows if row["separation_hz"] == s) / args.seeds
             for s in separations]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([s / 1e6 for s in separations], means, marker="o", label="Frequency-only baseline")
    ax.set(xlabel="Adjacent emitter frequency separation (MHz)",
           ylabel="Mean one-to-one assignment accuracy", ylim=(0, 1.05),
           title=f"Deinterleaving vs frequency separation ({args.seeds} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "frequency_sweep.png", dpi=160)
    plt.close(fig)
    print("Separation (MHz) | Mean 1:1 accuracy | Mean estimated tracks")
    for s, mean in zip(separations, means):
        subset = [row for row in rows if row["separation_hz"] == s]
        print(f"{s / 1e6:16.2f} | {mean:17.3f} | {sum(r['tracks'] for r in subset)/len(subset):21.2f}")
    print(f"Saved CSV and plot to {out}")


if __name__ == "__main__":
    main()
