"""Diagnose where bounded MHT changes phase-folding associations.

Truth is used ONLY for evaluation, never for association. Track labels are
mapped to physical emitters by optimal global one-to-one overlap. Generates
CSV and a figure under results/mht_diagnostics/.
"""
import csv
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from testbed.generator import generate
from testbed.phase_folding import PhaseFoldingDeinterleaver
from testbed.bounded_mht import BoundedMHTDeinterleaver
from compare_pri import score


def best_mapping(truth, assignments):
    """Globally optimal one-to-one track->emitter mapping for evaluation."""
    tracks = sorted(set(assignments.values()))
    emitters = sorted(set(truth.values()))
    if len(tracks) > 8 or len(emitters) > 8:
        raise ValueError("Diagnostic mapping limited to 8 tracks/emitters")
    best, best_count = {}, -1
    # Allow unmatched tracks by padding with dummy emitter labels.
    targets = emitters + [None] * max(0, len(tracks) - len(emitters))
    for perm in set(itertools.permutations(targets, len(tracks))):
        mapping = dict(zip(tracks, perm))
        count = sum(mapping.get(t) == truth[i] for i, t in assignments.items())
        if count > best_count:
            best, best_count = mapping, count
    return best


def main():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, truth = generate(scenario)
    algorithms = [
        ("phase", PhaseFoldingDeinterleaver()),
        ("mht", BoundedMHTDeinterleaver()),
    ]
    results, durations = {}, {}
    for name, algo in algorithms:
        start = time.perf_counter()
        assigned = algo.process(pdws)
        durations[name] = time.perf_counter() - start
        mapping = best_mapping(truth, assigned)
        results[name] = {
            "assignments": assigned,
            "mapping": mapping,
            "correct": {p.pdw_id: mapping.get(assigned.get(p.pdw_id)) == truth[p.pdw_id]
                        for p in pdws},
            "metrics": score(truth, assigned),
        }
    by_time = sorted(pdws, key=lambda p: p.toa_s)
    emitter_times = {
        e: np.array([p.toa_s for p in by_time if truth[p.pdw_id] == e])
        for e in set(truth.values())
    }

    def nearest_other(p):
        best = float("inf")
        for e, times in emitter_times.items():
            if e == truth[p.pdw_id]:
                continue
            idx = int(np.searchsorted(times, p.toa_s))
            for j in (idx - 1, idx):
                if 0 <= j < len(times):
                    best = min(best, abs(float(times[j]) - p.toa_s))
        return best * 1e6

    counts = Counter()
    rows = []
    for p in by_time:
        i = p.pdw_id
        a = results["phase"]["correct"][i]
        b = results["mht"]["correct"][i]
        category = ("both_correct" if a and b else
                    "mht_hurt" if a else
                    "mht_helped" if b else "both_wrong")
        counts[category] += 1
        rows.append({
            "pdw_id": i,
            "toa_s": p.toa_s,
            "true_emitter": truth[i],
            "phase_track": results["phase"]["assignments"].get(i, ""),
            "mht_track": results["mht"]["assignments"].get(i, ""),
            "phase_correct": int(a),
            "mht_correct": int(b),
            "category": category,
            "nearest_other_true_pulse_us": nearest_other(p),
        })

    out = Path("results/mht_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "pulse_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Observed PDWs: {len(pdws)}")
    for name in ("phase", "mht"):
        m = results[name]["metrics"]
        print(f"{name:5s}: accuracy={m['one_to_one_accuracy']:.3%}, "
              f"coverage={m['coverage']:.3%}, time={durations[name]:.3f}s")
        print(f"  track->emitter (evaluation only): {results[name]['mapping']}")
    for category in ("both_correct", "mht_helped", "mht_hurt", "both_wrong"):
        print(f"{category:14s}: {counts[category]}")
    print(f"Net MHT correct-association change: {counts['mht_helped'] - counts['mht_hurt']:+d}")
    for category in ("mht_helped", "mht_hurt", "both_wrong"):
        sample = [r for r in rows if r["category"] == category]
        if sample:
            separations = np.array([r["nearest_other_true_pulse_us"] for r in sample])
            print(f"{category}: {len(sample)} PDWs, "
                  f"median nearest-other separation={np.median(separations):.3f} us, "
                  f"within 0.5 us={int(np.count_nonzero(separations < 0.5))}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    symbols = {"mht_helped": ("green", "o"), "mht_hurt": ("red", "x"),
               "both_wrong": ("orange", "^")}
    emitter_order = {e: j for j, e in enumerate(sorted(emitter_times))}
    for category, (color, marker) in symbols.items():
        selected = [r for r in rows if r["category"] == category]
        if selected:
            axes[0].scatter([r["toa_s"] for r in selected],
                            [emitter_order[r["true_emitter"]] for r in selected],
                            s=23, c=color, marker=marker,
                            label=f"{category} ({len(selected)})")
    axes[0].set_yticks(list(emitter_order.values()),
                       list(emitter_order.keys()))
    axes[0].set_xlabel("TOA (s)")
    axes[0].set_ylabel("True emitter (evaluation only)")
    axes[0].set_title("Changes in association correctness: phase folding vs bounded MHT")
    axes[0].legend(loc="best")
    categories = ["mht_helped", "mht_hurt", "both_wrong"]
    edges = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, float("inf")]
    labels = [f"{edges[i]:g}-{edges[i+1]:g}" if np.isfinite(edges[i+1])
              else ">500" for i in range(len(edges)-1)]
    x = np.arange(len(labels))
    width = 0.25
    for k, category in enumerate(categories):
        vals = [sum(edges[j] <= r["nearest_other_true_pulse_us"] < edges[j+1]
                    for r in rows if r["category"] == category)
                for j in range(len(labels))]
        axes[1].bar(x + (k-1)*width, vals, width, label=category,
                    color=symbols[category][0])
    axes[1].set_xticks(x, labels, rotation=40, ha="right")
    axes[1].set_ylabel("Number of PDWs")
    axes[1].set_xlabel("Separation to nearest other-emitter pulse (us; truth for evaluation)")
    axes[1].set_title("Error changes versus pulse proximity")
    axes[1].legend()
    png_path = out / "mht_error_comparison.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"CSV: {csv_path}")
    print(f"Plot: {png_path}")
    print("NOTE: This diagnoses association differences; it does not establish "
          "that future observations resolve them.")


if __name__ == "__main__":
    main()
