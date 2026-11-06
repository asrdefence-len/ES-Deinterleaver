"""Plot controlled E46/E38 Model 2 competition and timing ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_confusion(axis, summary: dict, title: str) -> None:
    categories = summary["categories"]
    matrix_map = summary["matrix_truth_rows_prediction_columns"]
    matrix = np.asarray([
        [matrix_map[truth][prediction] for prediction in categories]
        for truth in categories
    ], dtype=float)
    image = axis.imshow(np.log1p(matrix), cmap="Blues", vmin=0.0)
    del image
    row_sum = matrix.sum(axis=1)
    for row in range(len(categories)):
        for column in range(len(categories)):
            percentage = (
                matrix[row, column] / row_sum[row] if row_sum[row] else 0.0
            )
            axis.text(
                column, row,
                f"{int(matrix[row, column]):,}\n{percentage:.2%}",
                ha="center", va="center",
                color=("white" if matrix[row, column] > 0.5 * matrix.max()
                       else "black"),
                fontsize=9,
            )
    labels = [f"E{value}" if value != "neither" else "Neither"
              for value in categories]
    axis.set_xticks(range(len(categories)), labels)
    axis.set_yticks(range(len(categories)), labels)
    axis.set(xlabel="Assigned", ylabel="Evaluation truth", title=title)


def plot_competition(result: dict, records: list[dict], output: Path) -> None:
    if not records:
        raise ValueError("No two-emitter diagnostic records supplied")
    track_ids = result["input_boundary"]["track_ids"]
    if len(track_ids) != 2:
        raise ValueError("Plot expects exactly two competing tracks")
    target_records = [
        row for row in records if row["evaluation_category"] in track_ids
    ]
    if not target_records:
        raise ValueError("Diagnostics contain no evaluation targets")

    start_s = float(result["input_boundary"]["held_out_start_toa_s"])
    time_s = np.asarray([row["toa_s"] - start_s for row in target_records])
    truth = np.asarray([row["evaluation_category"] for row in target_records])
    own_margin = np.asarray([
        row["timing_off"]["tracks"][row["evaluation_category"]]["model2_margin"]
        for row in target_records
    ])
    prediction_off = np.asarray([
        row["timing_off"]["selected_track"] or "neither"
        for row in target_records
    ])
    prediction_on = np.asarray([
        row["timing_on"]["selected_track"] or "neither"
        for row in target_records
    ])
    rescued = (prediction_off != truth) & (prediction_on == truth)
    missed = prediction_on != truth

    figure = plt.figure(figsize=(13, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.25, 1.0))
    off_axis = figure.add_subplot(grid[0, 0])
    on_axis = figure.add_subplot(grid[0, 1])
    margin_axis = figure.add_subplot(grid[1, :])
    outcome_axis = figure.add_subplot(grid[2, :])

    full = result["full_window_results"]
    _plot_confusion(off_axis, full["timing_off"], "Timing off")
    _plot_confusion(on_axis, full["timing_on"], "Timing on")

    colours = {track_ids[0]: "tab:blue", track_ids[1]: "tab:orange"}
    for track_id in track_ids:
        selected = truth == track_id
        margin_axis.scatter(
            time_s[selected], own_margin[selected], s=4, alpha=0.22,
            color=colours[track_id], label=f"E{track_id} own-track margin",
        )
    margin_axis.scatter(
        time_s[missed], own_margin[missed], marker="x", s=20,
        color="tab:red", label="Still missed with timing",
    )
    margin_axis.scatter(
        time_s[rescued], own_margin[rescued], s=34, facecolors="none",
        edgecolors="tab:green", linewidths=1.2,
        label=f"Timing rescued ({int(rescued.sum())})",
    )
    margin_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    margin_axis.set(
        xlabel="Time through common held-out interval (s)",
        ylabel="Own Model 2 score − frozen threshold",
        title=(
            "Each target pulse competes for one update; timing only changes "
            "near-threshold decisions"
        ),
    )
    lower = max(-5.0, float(np.quantile(own_margin, 0.001)) - 0.2)
    upper = min(7.0, float(np.quantile(own_margin, 0.999)) + 0.2)
    margin_axis.set_ylim(lower, upper)
    margin_axis.legend(fontsize=8, loc="lower right")

    off = full["timing_off"]
    on = full["timing_on"]
    categories = [
        f"E{track_ids[0]} accepted",
        f"E{track_ids[1]} accepted",
        "Cross-assigned",
        "Background accepted",
    ]
    off_values = [
        off["per_track"][track_ids[0]]["true_positive"],
        off["per_track"][track_ids[1]]["true_positive"],
        off["cross_assignments"],
        off["background_false_accepts"],
    ]
    on_values = [
        on["per_track"][track_ids[0]]["true_positive"],
        on["per_track"][track_ids[1]]["true_positive"],
        on["cross_assignments"],
        on["background_false_accepts"],
    ]
    positions = np.arange(len(categories))
    width = 0.36
    outcome_axis.bar(positions - width / 2, off_values, width,
                     color="tab:blue", label="Timing off")
    outcome_axis.bar(positions + width / 2, on_values, width,
                     color="tab:green", label="Timing on")
    for offset, values in ((-width / 2, off_values), (width / 2, on_values)):
        for index, value in enumerate(values):
            outcome_axis.text(index + offset, value, f"{value:,}",
                              ha="center", va="bottom", fontsize=9)
    outcome_axis.set_xticks(positions, categories)
    outcome_axis.set(
        ylabel="Held-out pulse count",
        title=(
            f"Timing rescues {result['timing_comparison']['target_rescued']} "
            "target pulses with no cross-assignment or background promotion"
        ),
    )
    outcome_axis.legend(fontsize=8)

    for axis in (off_axis, on_axis, margin_axis, outcome_axis):
        axis.grid(alpha=0.18)
    figure.suptitle(
        "HMM Model 2 — controlled two-emitter competition\n"
        f"E{track_ids[0]} versus E{track_ids[1]}; "
        f"{result['input_boundary']['window_pdws']:,} chronological PDWs; "
        "truth used only for seeds/evaluation",
        fontsize=14,
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
    plot_competition(result, records, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
