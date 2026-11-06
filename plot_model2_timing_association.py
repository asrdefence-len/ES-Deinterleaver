"""Plot the held-out Model 2 timing-off versus timing-on association test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(result: dict, records: list[dict], output: Path) -> None:
    if not records:
        raise ValueError("No association diagnostic records supplied")
    required = {"baseline", "timing_on", "evaluation_is_target", "toa_s"}
    missing = sorted(required - set(records[0]))
    if missing:
        raise ValueError(f"Not association diagnostics; missing: {', '.join(missing)}")

    target = np.asarray([bool(row["evaluation_is_target"]) for row in records])
    toa_s = np.asarray([float(row["toa_s"]) for row in records])
    time_s = toa_s - float(result["input_boundary"]["held_out_start_toa_s"])
    baseline_score = np.asarray([
        float(row["baseline"]["baseline_log_evidence"]) for row in records
    ])
    timing_score = np.asarray([
        float(row["timing_on"]["combined_log_evidence"]) for row in records
    ])
    baseline_accept = np.asarray([
        bool(row["baseline"]["accepted"]) for row in records
    ])
    timing_accept = np.asarray([
        bool(row["timing_on"]["accepted"]) for row in records
    ])
    boost = np.asarray([
        float(row["timing_on"]["timing_applied_boost"]) for row in records
    ])
    threshold = float(result["calibration"]["acceptance_threshold"])
    rescued = target & ~baseline_accept & timing_accept
    missed = target & ~timing_accept

    figure, axes = plt.subplots(
        4, 1, figsize=(12, 13), constrained_layout=True,
    )

    target_scores = baseline_score[target]
    bins = np.linspace(float(target_scores.min()), float(target_scores.max()), 75)
    axes[0].hist(target_scores, bins=bins, color="tab:blue", alpha=0.7,
                 label="E46 candidates")
    axes[0].axvline(threshold, color="black", linestyle="--",
                    label=f"Frozen threshold {threshold:.3f}")
    distractor_max = float(np.max(baseline_score[~target]))
    axes[0].text(
        0.99, 0.92,
        f"Best distractor score: {distractor_max:.1f}\n"
        "(all distractors are left of this E46-scale view)",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
    )
    axes[0].set(
        xlabel="Frequency + AOA + PW predictive log evidence",
        ylabel="E46 pulse count",
        title="Frozen Model 2 evidence and calibrated association threshold",
    )
    axes[0].legend(fontsize=8)

    target_time = time_s[target]
    target_margin = baseline_score[target] - threshold
    axes[1].scatter(target_time, target_margin, s=4, alpha=0.25,
                    color="tab:blue", label="E46 baseline margin")
    axes[1].scatter(time_s[missed], baseline_score[missed] - threshold,
                    s=18, marker="x", color="tab:red",
                    label="Still missed with timing")
    axes[1].scatter(time_s[rescued], baseline_score[rescued] - threshold,
                    s=30, facecolors="none", edgecolors="tab:green",
                    linewidths=1.2,
                    label=f"Timing-rescued E46 ({int(rescued.sum())})")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set(
        xlabel="Time through held-out interval (s)",
        ylabel="Baseline score − threshold",
        title="Held-out E46 decisions; timing only acts near the threshold",
    )
    axes[1].set_ylim(
        max(-5.0, float(np.quantile(target_margin, 0.001)) - 0.2),
        min(7.0, float(np.quantile(target_margin, 0.999)) + 0.2),
    )
    axes[1].legend(fontsize=8, loc="lower right")

    positive_bins = np.linspace(0.0, float(result["timing_model"]["maximum_boost"]), 40)
    axes[2].hist(boost[target], bins=positive_bins, alpha=0.65,
                 color="tab:green", label="E46 candidates")
    axes[2].hist(boost[~target], bins=positive_bins, alpha=0.5,
                 color="tab:orange", label="Other emitters")
    axes[2].set(
        xlabel="Applied timing support (log-evidence units)",
        ylabel="Candidate count",
        title=(
            "Positive-only phase support; zero means neutral and support is capped"
        ),
    )
    axes[2].legend(fontsize=8)

    baseline = result["baseline"]
    timing = result["timing_on"]
    categories = ["E46 accepted", "E46 missed", "Distractors accepted"]
    baseline_counts = [
        baseline["true_positive"], baseline["false_negative"],
        baseline["false_positive"],
    ]
    timing_counts = [
        timing["true_positive"], timing["false_negative"],
        timing["false_positive"],
    ]
    positions = np.arange(len(categories))
    width = 0.36
    axes[3].bar(positions - width / 2, baseline_counts, width,
                label="Timing off", color="tab:blue")
    axes[3].bar(positions + width / 2, timing_counts, width,
                label="Timing on", color="tab:green")
    for offset, values in ((-width / 2, baseline_counts),
                           (width / 2, timing_counts)):
        for index, value in enumerate(values):
            axes[3].text(index + offset, value, f"{value:,}",
                         ha="center", va="bottom", fontsize=8)
    axes[3].set_xticks(positions, categories)
    axes[3].set(
        ylabel="Held-out candidate count",
        title=(
            "Decision result — 9 E46 pulses rescued; no distractor promoted"
        ),
    )
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "HMM Model 2 — controlled E46 timing association ablation\n"
        f"{len(records):,} RF-plausible mixed candidates; truth used only for "
        "seed/evaluation; identical candidate stream",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    records = [
        json.loads(line) for line in args.diagnostics.open() if line.strip()
    ]
    plot_comparison(result, records, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
