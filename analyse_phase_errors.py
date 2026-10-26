"""Diagnose phase-folding association errors and temporal pulse collisions.

Truth is used ONLY after inference to score and visualise results.
"""
import argparse
import csv
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from testbed.generator import generate
from testbed.phase_folding import PhaseFoldingDeinterleaver


def best_track_mapping(truth, assignments):
    """Find maximum-overlap one-to-one track-to-emitter mapping (3-emitter study)."""
    tracks = sorted(set(assignments.values()))
    emitters = sorted(set(truth.values()))
    if len(tracks) > 8 or len(emitters) > 8:
        raise ValueError("Exact mapping limited to eight tracks/emitters")
    # Allow unmatched tracks or emitters using None placeholders.
    size = max(len(tracks), len(emitters))
    padded = emitters + [None] * (size - len(emitters))
    counts = Counter((assignments[i], truth[i]) for i in assignments)
    best_score = -1
    best = {}
    for perm in permutations(padded):
        mapping = dict(zip(tracks, perm))
        value = sum(counts[(track, emitter)] for track, emitter in mapping.items()
                    if emitter is not None)
        if value > best_score:
            best_score, best = value, mapping
    return best


def nearest_other_emitter(toa, emitter, indexed):
    """Nearest observed pulse from another *physical* emitter (diagnostic only)."""
    best = float("inf")
    for other, times in indexed.items():
        if other == emitter:
            continue
        i = bisect_left(times, toa)
        if i < len(times):
            best = min(best, abs(times[i] - toa))
        if i:
            best = min(best, abs(times[i - 1] - toa))
    return best


def analyse(pdws, truth, assignments):
    mapping = best_track_mapping(truth, assignments)
    indexed = defaultdict(list)
    for p in pdws:
        indexed[truth[p.pdw_id]].append(p.toa_s)
    rows = []
    for p in pdws:
        actual = truth[p.pdw_id]
        track = assignments.get(p.pdw_id)
        predicted = mapping.get(track)
        correct = predicted == actual
        nearest_us = nearest_other_emitter(p.toa_s, actual, indexed) * 1e6
        rows.append({
            "pdw_id": p.pdw_id, "toa_s": p.toa_s,
            "true_emitter": actual, "assigned_track": track or "",
            "mapped_emitter": predicted or "", "correct": correct,
            "nearest_other_emitter_us": nearest_us,
        })
    return mapping, rows


def make_plot(rows, output):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    emitters = sorted(set(r["true_emitter"] for r in rows))
    levels = {emitter: i for i, emitter in enumerate(emitters)}
    correct = [r for r in rows if r["correct"]]
    errors = [r for r in rows if not r["correct"]]
    for emitter in emitters:
        points = [r for r in correct if r["true_emitter"] == emitter]
        axes[0].scatter([r["toa_s"] for r in points],
                        [levels[emitter]] * len(points), s=5, alpha=0.4,
                        label=f"Correct: {emitter}")
    if errors:
        axes[0].scatter([r["toa_s"] for r in errors],
                        [levels[r["true_emitter"]] for r in errors],
                        s=32, marker="x", color="red", label="Incorrect / unassigned")
    axes[0].set_yticks(list(levels.values()), list(levels.keys()))
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("True emitter (diagnostic)")
    axes[0].set_title("PDW association errors across observation time")
    axes[0].legend(loc="upper right", ncol=4)

    bins = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, np.inf])
    all_sep = np.array([r["nearest_other_emitter_us"] for r in rows])
    err_sep = np.array([r["nearest_other_emitter_us"] for r in errors])
    total, _ = np.histogram(all_sep, bins=bins)
    bad, _ = np.histogram(err_sep, bins=bins)
    rate = np.divide(bad, total, out=np.zeros_like(bad, dtype=float), where=total > 0)
    labels = [f"{bins[i]:g}–{bins[i+1]:g}" for i in range(len(bins)-2)] + [">=500"]
    axes[1].bar(np.arange(len(rate)), rate * 100)
    axes[1].set_xticks(np.arange(len(rate)), labels, rotation=35, ha="right")
    axes[1].set_ylabel("Incorrect assignments (%)")
    axes[1].set_xlabel("Nearest observed pulse from another emitter (µs)")
    axes[1].set_title("Association error rate conditional on cross-emitter TOA separation")
    axes[1].set_ylim(bottom=0)
    for i, (n, k) in enumerate(zip(total, bad)):
        axes[1].text(i, rate[i] * 100 + 0.5, f"{k}/{n}", ha="center", fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/same_frequency.json")
    parser.add_argument("--output", default="results/phase_errors")
    args = parser.parse_args()
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    pdws, truth = generate(scenario)
    algorithm = PhaseFoldingDeinterleaver()
    assignments = algorithm.process(pdws)
    mapping, rows = analyse(pdws, truth, assignments)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pdw_error_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    make_plot(rows, output / "phase_error_analysis.png")
    errors = [r for r in rows if not r["correct"]]
    print(f"PDWs: {len(rows)}; errors: {len(errors)}; accuracy: {1-len(errors)/len(rows):.3%}")
    print(f"Track-to-emitter mapping (scoring only): {mapping}")
    for threshold in [0.5, 1, 2, 5, 10, 20]:
        near = [r for r in rows if r["nearest_other_emitter_us"] <= threshold]
        count = sum(not r["correct"] for r in near)
        print(f"Nearest other emitter <= {threshold:g} us: {count}/{len(near)} errors")
    print(f"Saved CSV and plot to {output}")


if __name__ == "__main__":
    main()
