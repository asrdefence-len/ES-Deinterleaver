"""Plot truth-blind Model 2 birth on the controlled E46/E38 mixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_confusion(axis, summary: dict, emitters: list[str], title: str) -> None:
    predictions = [*emitters, "rejected"]
    matrix_map = summary["matrix_truth_rows_prediction_columns"]
    matrix = np.asarray([
        [matrix_map[truth][prediction] for prediction in predictions]
        for truth in emitters
    ], dtype=float)
    axis.imshow(np.log1p(matrix), cmap="Blues", vmin=0.0)
    for row in range(len(emitters)):
        total = matrix[row].sum()
        for column in range(len(predictions)):
            fraction = matrix[row, column] / total if total else 0.0
            axis.text(
                column, row,
                f"{int(matrix[row, column]):,}\n{fraction:.2%}",
                ha="center", va="center", fontsize=9,
            )
    axis.set_xticks(
        range(len(predictions)),
        [f"E{x}" if x != "rejected" else "Rejected" for x in predictions],
    )
    axis.set_yticks(range(len(emitters)), [f"E{x}" for x in emitters])
    axis.set(xlabel="Mapped automatic-track decision",
             ylabel="Evaluation truth", title=title)


def plot_birth(result: dict, records: list[dict], output: Path) -> None:
    if not records:
        raise ValueError("No automatic-birth diagnostic records supplied")
    emitters = list(result["input_boundary"][
        "selected_truth_counts_evaluation_only"
    ])
    if len(emitters) != 2:
        raise ValueError("Plot expects exactly two controlled emitters")

    figure = plt.figure(figsize=(14, 15))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.25, 1.0))
    off_axis = figure.add_subplot(grid[0, 0])
    on_axis = figure.add_subplot(grid[0, 1])
    timeline_axis = figure.add_subplot(grid[1, :])
    composition_axis = figure.add_subplot(grid[2, 0])
    recall_axis = figure.add_subplot(grid[2, 1])

    _plot_confusion(off_axis, result["timing_off"], emitters, "Timing off")
    _plot_confusion(on_axis, result["timing_on"], emitters, "Timing on")

    time = np.asarray([row["toa_s"] for row in records])
    time -= time[0]
    truth = np.asarray([str(row["evaluation_truth_label"]) for row in records])
    accepted = np.asarray([
        row["timing_on"]["track_id"] is not None for row in records
    ])
    colours = {emitters[0]: "tab:blue", emitters[1]: "tab:orange"}
    levels = {emitters[0]: 1.0, emitters[1]: 0.0}
    for emitter in emitters:
        selected = (truth == emitter) & accepted
        timeline_axis.scatter(
            time[selected], np.full(int(selected.sum()), levels[emitter]),
            s=4, alpha=0.25, color=colours[emitter],
            label=f"Accepted E{emitter}",
        )
    rejected = ~accepted
    timeline_axis.scatter(
        time[rejected], np.asarray([levels[value] for value in truth[rejected]]),
        marker="x", s=16, color="tab:red", label="Rejected target pulse",
    )
    confirm_rows = [
        (index, row) for index, row in enumerate(records)
        if row["timing_on"]["outcome"] == "confirmed"
    ]
    for number, (index, row) in enumerate(confirm_rows):
        timeline_axis.axvline(
            time[index], color="tab:green", linestyle="--", linewidth=1.0,
            label="Automatic confirmation" if number == 0 else None,
        )
    timeline_axis.set_yticks(
        [levels[emitters[1]], levels[emitters[0]]],
        [f"E{emitters[1]}", f"E{emitters[0]}"],
    )
    timeline_axis.set(
        xlabel="Time from first selected pulse (s)",
        ylabel="Evaluation emitter",
        title=(
            "Chronological unlabelled input — two births, two confirmations, "
            "one identity per emitter"
        ),
    )
    timeline_axis.legend(fontsize=8, loc="upper right")

    tracks = result["timing_on"]["tracks"]
    positions = np.arange(len(tracks))
    bottoms = np.zeros(len(tracks))
    for emitter in emitters:
        values = np.asarray([
            row["truth_counts_evaluation_only"].get(emitter, 0)
            for row in tracks
        ])
        composition_axis.bar(
            positions, values, bottom=bottoms, color=colours[emitter],
            label=f"E{emitter} pulses",
        )
        bottoms += values
    composition_axis.set_xticks(
        positions,
        [
            f"{row['track_id']} → E{row['mapped_evaluation_emitter']}\n"
            f"RF {row['rf_state_count']}; PW {row['pulse_width_state_count']}"
            for row in tracks
        ],
    )
    composition_axis.set(
        ylabel="Assigned pulses",
        title="Automatic-track composition (evaluation labels only)",
    )
    composition_axis.legend(fontsize=8)

    width = 0.36
    off_recall = [
        result["timing_off"]["per_emitter"][emitter]["recall"]
        for emitter in emitters
    ]
    on_recall = [
        result["timing_on"]["per_emitter"][emitter]["recall"]
        for emitter in emitters
    ]
    recall_axis.bar(positions - width / 2, off_recall, width,
                    color="tab:blue", label="Timing off")
    recall_axis.bar(positions + width / 2, on_recall, width,
                    color="tab:green", label="Timing on")
    for offset, values in ((-width / 2, off_recall), (width / 2, on_recall)):
        for index, value in enumerate(values):
            recall_axis.text(index + offset, value, f"{value:.2%}",
                             ha="center", va="bottom", fontsize=9)
    recall_axis.set_xticks(positions, [f"E{x}" for x in emitters])
    recall_axis.set_ylim(0.94, 1.005)
    recall_axis.set(
        ylabel="Recall",
        title=(
            f"Timing rescues {result['timing_comparison']['target_rescued']} "
            "pulses and loses none"
        ),
    )
    recall_axis.legend(fontsize=8, loc="lower right")

    for axis in (off_axis, on_axis, timeline_axis,
                 composition_axis, recall_axis):
        axis.grid(alpha=0.18)
    figure.suptitle(
        "HMM Model 2 — automatic birth on a controlled unlabelled mixture\n"
        f"{result['input_boundary']['selected_pdws']:,} interleaved "
        f"E{emitters[0]}/E{emitters[1]} pulses; background emitters excluded "
        "outside the tracker",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.06, top=0.91,
        hspace=0.42, wspace=0.25,
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
    plot_birth(result, records, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
