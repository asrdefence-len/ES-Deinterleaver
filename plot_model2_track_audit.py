"""Overlay Model 2 track assignments on the raw chronological PDW scene."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _sample_indices(mask: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=int)
    return indices[positions]


def select_default_tracks(result: dict, maximum: int = 6) -> list[str]:
    """Choose target, mixed and pure-tentative examples for visual audit."""
    rows = result["tracker"]["tracks"]
    selected = []

    def add(track_id: str | None) -> None:
        if track_id and track_id not in selected and len(selected) < maximum:
            selected.append(track_id)

    for target in ("46", "38"):
        add(result.get("target_evaluation", {}).get(target, {}).get(
            "primary_track_id",
        ))
    mixed = sorted(
        (
            row for row in rows
            if row["status"] == "confirmed"
            and row.get("evaluation_purity") is not None
            and row["evaluation_purity"] < 0.9
        ),
        key=lambda row: row["assigned_count_evaluation"],
        reverse=True,
    )
    for row in mixed[:2]:
        add(row["track_id"])
    tentative = sorted(
        (row for row in rows if row["status"] == "tentative"),
        key=lambda row: row["assigned_count_evaluation"],
        reverse=True,
    )
    for row in tentative:
        add(row["track_id"])
    return selected


def _trace_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as trace:
        if int(trace["schema_version"][0]) != 1:
            raise ValueError("unsupported Model 2 audit trace schema")
        arrays = {name: np.array(trace[name]) for name in trace.files}
    lengths = {
        len(arrays[name]) for name in (
            "pdw_id", "toa_s", "frequency_mhz", "aoa_deg",
            "pulse_width_us", "truth_code", "track_code", "outcome_code",
        )
    }
    if len(lengths) != 1:
        raise ValueError("audit trace arrays have inconsistent lengths")
    return arrays


def plot_track_audit(
    result: dict,
    trace_path: Path,
    output: Path,
    track_ids: list[str] | None = None,
    background_points: int = 15_000,
    page_label: str | None = None,
) -> list[str]:
    trace = _trace_arrays(trace_path)
    rows_by_id = {
        row["track_id"]: row for row in result["tracker"]["tracks"]
    }
    selected = track_ids or select_default_tracks(result)
    if not selected:
        raise ValueError("no tracks available for the audit overlay")
    missing = [track_id for track_id in selected if track_id not in rows_by_id]
    if missing:
        raise ValueError(f"unknown track IDs: {', '.join(missing)}")

    track_vocab = [str(value) for value in trace["track_ids"]]
    truth_vocab = [str(value) for value in trace["truth_labels"]]
    missing_trace = [track_id for track_id in selected
                     if track_id not in track_vocab]
    if missing_trace:
        raise ValueError(
            "tracks absent from audit trace: " + ", ".join(missing_trace)
        )
    track_codes = {value: index for index, value in enumerate(track_vocab)}
    truth_codes = {value: index for index, value in enumerate(truth_vocab)}

    time_s = trace["toa_s"] - trace["toa_s"][0]
    frequency_ghz = trace["frequency_mhz"] / 1000.0
    aoa_deg = trace["aoa_deg"]
    pulse_width_us = trace["pulse_width_us"]
    feature_values = (frequency_ghz, aoa_deg, pulse_width_us)
    feature_labels = (
        "Frequency (GHz)", "AOA (deg)", "Pulse width (µs; log scale)",
    )

    total = len(time_s)
    background_mask = np.ones(total, dtype=bool)
    background = _sample_indices(background_mask, background_points)
    figure, axes = plt.subplots(
        len(selected), 3,
        figsize=(19, 2.75 * len(selected) + 2.0),
        sharex=True,
        squeeze=False,
    )

    events_by_track: dict[str, list[dict]] = {}
    for event in result["tracker"].get("lifecycle_events", []):
        events_by_track.setdefault(event["track_id"], []).append(event)

    for row_index, track_id in enumerate(selected):
        row = rows_by_id[track_id]
        dominant_truth = str(row["dominant_truth_emitter_evaluation_only"])
        dominant_code = truth_codes.get(dominant_truth, -2)
        assigned = trace["track_code"] == track_codes[track_id]
        dominant = trace["truth_code"] == dominant_code
        correct = assigned & dominant
        contaminant = assigned & ~dominant
        missed = ~assigned & dominant
        correct_indices = _sample_indices(correct, 35_000)
        contaminant_indices = _sample_indices(contaminant, 20_000)
        missed_indices = _sample_indices(missed, 20_000)

        composition = list(row["truth_counts_evaluation_only"].items())[:3]
        composition_text = ", ".join(
            f"E{truth} {count:,}" for truth, count in composition
        )
        seed_indices = np.flatnonzero(assigned)[:8]
        seed_counts = Counter(
            truth_vocab[int(code)] for code in trace["truth_code"][seed_indices]
        )
        seed_text = ", ".join(
            f"E{truth} {count}" for truth, count in seed_counts.most_common()
        )
        purity = row.get("evaluation_purity")
        purity_text = "n/a" if purity is None else f"{purity:.2%}"
        row_label = (
            f"{track_id} · {row['status']} · dominant E{dominant_truth} · "
            f"{row['assigned_count_evaluation']:,} assigned · "
            f"purity {purity_text}\n{composition_text}"
            f"\nfirst-eight AOA seed (evaluation): {seed_text}"
        )

        for column, (values, y_label) in enumerate(zip(
            feature_values, feature_labels,
        )):
            axis = axes[row_index, column]
            axis.scatter(
                time_s[background], values[background],
                s=1.0, color="0.45", alpha=0.08,
                linewidths=0, rasterized=True,
            )
            axis.scatter(
                time_s[correct_indices], values[correct_indices],
                s=2.5, color="tab:blue", alpha=0.75,
                linewidths=0, rasterized=True,
            )
            if len(missed_indices):
                axis.scatter(
                    time_s[missed_indices], values[missed_indices],
                    s=9, color="tab:orange", alpha=0.8,
                    marker="x", linewidths=0.6, rasterized=True,
                )
            if len(contaminant_indices):
                axis.scatter(
                    time_s[contaminant_indices], values[contaminant_indices],
                    s=10, color="tab:red", alpha=0.85,
                    marker="x", linewidths=0.7, rasterized=True,
                )
            for event in events_by_track.get(track_id, []):
                event_time = event["toa_s"] - trace["toa_s"][0]
                style = {
                    "birth": ("0.15", ":"),
                    "confirmed": ("tab:green", "--"),
                    "validation_rejected": ("tab:red", "-."),
                }.get(event["event"])
                if style:
                    axis.axvline(
                        event_time, color=style[0], linestyle=style[1],
                        linewidth=0.9, alpha=0.85,
                    )
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.16)
            if column == 2:
                axis.set_yscale("log")
            if row_index == len(selected) - 1:
                axis.set_xlabel("Time from first PDW (s)")
        axes[row_index, 0].text(
            0.01, 0.97, row_label,
            transform=axes[row_index, 0].transAxes,
            ha="left", va="top", fontsize=8.5,
            bbox={
                "facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                "pad": 2.5,
            },
        )

    for column, title in enumerate((
        "RF behaviour", "Bearing behaviour", "Pulse-width behaviour",
    )):
        axes[0, column].set_title(title)
    axes[0, 1].set_ylim(-180.0, 180.0)
    for row_index in range(1, len(selected)):
        axes[row_index, 1].set_ylim(-180.0, 180.0)

    legend = (
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               color="0.45", alpha=0.4, label="Raw scene PDWs (sample)"),
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               color="tab:blue", label="Assigned dominant-emitter pulse"),
        Line2D([], [], marker="x", linestyle="none", markersize=5,
               color="tab:orange", label="Dominant-emitter pulse missed by track"),
        Line2D([], [], marker="x", linestyle="none", markersize=5,
               color="tab:red", label="Contaminant assigned to track"),
        Line2D([], [], color="0.15", linestyle=":", label="Birth"),
        Line2D([], [], color="tab:green", linestyle="--", label="Confirmation"),
        Line2D([], [], color="tab:red", linestyle="-.", label="Birth rejection"),
    )
    figure.legend(
        handles=legend, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.955), frameon=False,
    )
    figure.suptitle(
        "Model 2 track assignments superimposed on raw Config 101 PDWs"
        + (f" — {page_label}" if page_label else "")
        + "\n"
        "Truth labels colour evaluation overlays only; they were unavailable "
        "to the tracker",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return selected


def plot_track_audit_pages(
    result: dict,
    trace_path: Path,
    output_dir: Path,
    *,
    track_ids: list[str] | None = None,
    tracks_per_page: int = 5,
    background_points: int = 15_000,
) -> list[Path]:
    if tracks_per_page < 1:
        raise ValueError("tracks per page must be positive")
    all_tracks = track_ids or sorted(
        (row["track_id"] for row in result["tracker"]["tracks"]),
        key=lambda track_id: int(track_id.removeprefix("T")),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    page_count = (len(all_tracks) + tracks_per_page - 1) // tracks_per_page
    outputs = []
    for page_index, start in enumerate(
        range(0, len(all_tracks), tracks_per_page), start=1,
    ):
        page_tracks = all_tracks[start:start + tracks_per_page]
        output = output_dir / f"track_audit_page_{page_index:02d}.png"
        plot_track_audit(
            result,
            trace_path,
            output,
            track_ids=page_tracks,
            background_points=background_points,
            page_label=f"page {page_index}/{page_count}",
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("audit_trace", type=Path)
    parser.add_argument("--tracks", nargs="+")
    parser.add_argument("--tracks-per-page", type=int, default=5)
    parser.add_argument("--background-points", type=int, default=15_000)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "results/sequential_pdw/hmm_model2_birth/"
            "full_scene_track_audit"
        ),
    )
    args = parser.parse_args()
    if args.background_points < 1:
        raise ValueError("background point count must be positive")
    result = json.loads(args.result.read_text())
    outputs = plot_track_audit_pages(
        result,
        args.audit_trace,
        args.output_dir,
        track_ids=args.tracks,
        tracks_per_page=args.tracks_per_page,
        background_points=args.background_points,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
