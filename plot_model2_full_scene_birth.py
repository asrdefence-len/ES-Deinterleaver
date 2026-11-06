"""Plot the frozen full-scene Model 2 automatic-birth stress replay."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_full_scene(result: dict, output: Path) -> None:
    tracks = result["tracker"]["tracks"]
    events = result["tracker"]["lifecycle_events"]
    if not tracks or not events:
        raise ValueError("Result contains no tracks or lifecycle events")

    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    lifecycle_axis, composition_axis, rejection_axis, target_axis = axes.flat

    track_ids = [row["track_id"] for row in tracks]
    track_positions = {track_id: index for index, track_id in enumerate(track_ids)}
    for event_name, marker, colour in (
        ("birth", "o", "tab:blue"),
        ("confirmed", "s", "tab:green"),
    ):
        selected = [event for event in events if event["event"] == event_name]
        lifecycle_axis.scatter(
            [event["pdw_id"] + 1 for event in selected],
            [track_positions[event["track_id"]] for event in selected],
            marker=marker, s=48, color=colour, label=event_name.title(),
        )
    capacity = result["tracker"]["capacity_first_reached"]
    lifecycle_axis.axvline(
        capacity["processed_pdws"], color="tab:red", linestyle="--",
        label=f"Capacity full at pulse {capacity['processed_pdws']}",
    )
    lifecycle_axis.set_xscale("log")
    lifecycle_axis.set_yticks(range(len(track_ids)), track_ids)
    lifecycle_axis.set(
        xlabel="Chronological input pulse (log scale)",
        ylabel="Automatic track",
        title="All eight birth slots are consumed before confirmation",
    )
    lifecycle_axis.legend(fontsize=8)

    label_totals = Counter()
    for track in tracks:
        label_totals.update(track["truth_counts_evaluation_only"])
    display_labels = [label for label, _count in label_totals.most_common(8)]
    positions = np.arange(len(tracks))
    bottoms = np.zeros(len(tracks))
    colour_map = plt.get_cmap("tab10")
    for label_index, label in enumerate(display_labels):
        values = np.asarray([
            track["truth_counts_evaluation_only"].get(label, 0)
            for track in tracks
        ])
        composition_axis.bar(
            positions, values, bottom=bottoms,
            color=colour_map(label_index % 10), label=f"E{label}",
        )
        bottoms += values
    other = np.asarray([
        track["assigned_count_evaluation"]
        - sum(track["truth_counts_evaluation_only"].get(label, 0)
              for label in display_labels)
        for track in tracks
    ])
    if np.any(other > 0):
        composition_axis.bar(
            positions, other, bottom=bottoms, color="0.65", label="Other",
        )
    composition_axis.set_xticks(
        positions,
        [
            f"{track['track_id']}\nE{track['dominant_truth_emitter_evaluation_only']}\n"
            f"{track['evaluation_purity']:.1%}"
            for track in tracks
        ],
    )
    composition_axis.set(
        ylabel="Assigned PDWs",
        title="Final automatic-track composition and evaluation purity",
    )
    composition_axis.legend(fontsize=7, ncol=3)

    assigned = result["tracker"]["assigned_pdws"]
    rejected = result["input_boundary"]["source_pdws"] - assigned
    capacity_rejected = result["tracker"]["rejection_reasons"].get(
        "track_capacity", 0,
    )
    suppressed = result["tracker"]["rejection_reasons"].get(
        "confirmed_aoa_birth_suppression", 0,
    )
    values = [assigned, rejected, capacity_rejected, suppressed]
    labels = ["Assigned", "Rejected", "Capacity\nrejections",
              "Near-track\nsuppression"]
    bars = rejection_axis.bar(
        np.arange(len(values)), values,
        color=("tab:blue", "tab:red", "tab:orange", "tab:purple"),
    )
    rejection_axis.set_xticks(range(len(labels)), labels)
    for bar, value in zip(bars, values):
        rejection_axis.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
            ha="center", va="bottom", fontsize=9,
        )
    rejection_axis.set(
        ylabel="PDW count",
        title=(
            f"Full-scene assignment coverage: "
            f"{result['tracker']['assignment_coverage']:.2%}"
        ),
    )

    targets = ["46", "38"]
    controlled = [
        result["controlled_reference"][f"E{target}_recall"]
        for target in targets
    ]
    full_scene = [
        result["target_evaluation"][target]["assigned_coverage"]
        for target in targets
    ]
    width = 0.36
    x = np.arange(len(targets))
    target_axis.bar(
        x - width / 2, controlled, width, color="tab:green",
        label="Controlled E46/E38 mixture",
    )
    target_axis.bar(
        x + width / 2, full_scene, width, color="tab:red",
        label="Frozen full scene",
    )
    for offset, values_to_label in (
        (-width / 2, controlled), (width / 2, full_scene),
    ):
        for index, value in enumerate(values_to_label):
            target_axis.text(
                index + offset, value + 0.015, f"{value:.2%}",
                ha="center", va="bottom", fontsize=9,
            )
    target_axis.set_xticks(x, [f"E{target}" for target in targets])
    target_axis.set_ylim(0.0, 1.08)
    target_axis.set(
        ylabel="Assigned coverage",
        title="Target retention collapses when background birth is enabled",
    )
    target_axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle(
        "HMM Model 2 — frozen full-scene automatic-birth stress replay\n"
        f"{result['input_boundary']['source_pdws']:,} unfiltered PDWs; "
        "eight-track limit unchanged; E46/E38 retention failed",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.08, top=0.89,
        hspace=0.38, wspace=0.25,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    plot_full_scene(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
