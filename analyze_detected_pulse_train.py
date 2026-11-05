"""Analyze a detected TOA train without declaring observed gaps to be PRI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def analyze_toas(toa_s: np.ndarray, *, blocks: int = 12) -> dict:
    toa_s = np.asarray(toa_s, dtype=float)
    if len(toa_s) < 3 or np.any(np.diff(toa_s) < 0.0):
        raise ValueError("need at least three non-decreasing detected TOAs")
    gaps_us = np.diff(toa_s) * 1e6
    positive = gaps_us[gaps_us > 0.0]
    if not len(positive):
        raise ValueError("detected train has no positive gaps")

    lag_differences = []
    for lag in range(1, min(8, len(toa_s) - 1) + 1):
        values = (toa_s[lag:] - toa_s[:-lag]) * 1e6
        lag_differences.append(values[(values > 0.0) & (values <= 100.0)])
    lag_differences_us = np.concatenate(lag_differences)
    lag_edges = np.arange(0.0, 100.25, 0.25)
    lag_counts, _ = np.histogram(lag_differences_us, bins=lag_edges)
    peak_indices, properties = find_peaks(
        lag_counts,
        prominence=max(2.0, 0.01 * float(lag_counts.max(initial=0))),
        distance=2,
    )
    peak_prominence = dict(zip(peak_indices, properties["prominences"]))
    peak_order = sorted(
        peak_indices,
        key=lambda index: float(peak_prominence[index]),
        reverse=True,
    )[:12]
    lag_peaks = [
        {
            "lag_us": float((lag_edges[index] + lag_edges[index + 1]) / 2.0),
            "count": int(lag_counts[index]),
        }
        for index in peak_order
    ]

    block_rows = []
    for indices in np.array_split(np.arange(len(positive)), min(blocks, len(positive))):
        values = positive[indices]
        block_rows.append({
            "start_gap_index": int(indices[0]),
            "end_gap_index": int(indices[-1]),
            "p10_us": float(np.quantile(values, 0.10)),
            "median_us": float(np.median(values)),
            "p90_us": float(np.quantile(values, 0.90)),
        })

    return {
        "status": "detected-train timing description; not transmitter PRI inference",
        "transmitter_pri_estimate": None,
        "detected_pulses": int(len(toa_s)),
        "positive_adjacent_gaps": int(len(positive)),
        "duration_s": float(toa_s[-1] - toa_s[0]),
        "adjacent_gap_quantiles_us": {
            "minimum": float(np.min(positive)),
            "p05": float(np.quantile(positive, 0.05)),
            "p25": float(np.quantile(positive, 0.25)),
            "median": float(np.median(positive)),
            "p75": float(np.quantile(positive, 0.75)),
            "p95": float(np.quantile(positive, 0.95)),
            "maximum": float(np.max(positive)),
        },
        "lag_difference_definition": (
            "TOA differences for detected-pulse lags 1..8, capped at 100 us; "
            "peaks may be PRI, sums, harmonics, interleaving or detection effects"
        ),
        "lag_difference_peaks": lag_peaks,
        "chronological_gap_blocks": block_rows,
    }


def plot_analysis(toa_s: np.ndarray, analysis: dict, output: Path) -> None:
    gaps_us = np.diff(toa_s) * 1e6
    positive = gaps_us[gaps_us > 0.0]
    lag_differences = []
    for lag in range(1, min(8, len(toa_s) - 1) + 1):
        values = (toa_s[lag:] - toa_s[:-lag]) * 1e6
        lag_differences.append(values[(values > 0.0) & (values <= 100.0)])
    lag_differences_us = np.concatenate(lag_differences)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].hist(positive, bins=np.geomspace(max(positive.min(), 0.25), positive.max(), 100))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Adjacent detected-pulse gap (µs)", ylabel="Count",
                title="Received inter-arrival distribution")

    axes[1].hist(lag_differences_us, bins=np.arange(0.0, 100.25, 0.25), color="tab:green")
    for peak in analysis["lag_difference_peaks"][:8]:
        axes[1].axvline(peak["lag_us"], color="tab:red", alpha=0.45, linewidth=0.8)
    axes[1].set(xlim=(0, 100), xlabel="Detected-train lag difference (µs)", ylabel="Count",
                title="Lag-difference structure; marked peaks are not automatically PRI")

    rows = analysis["chronological_gap_blocks"]
    x = np.arange(1, len(rows) + 1)
    axes[2].plot(x, [row["median_us"] for row in rows], marker="o", label="Median")
    axes[2].fill_between(
        x,
        [row["p10_us"] for row in rows],
        [row["p90_us"] for row in rows],
        alpha=0.2,
        label="10–90%",
    )
    axes[2].set(xlabel="Chronological pulse-train block", ylabel="Observed gap (µs)",
                title="Stability of the detected inter-arrival process")
    axes[2].set_yscale("log")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Detected pulse-train timing shadow — {analysis['detected_pulses']:,} pulses\n"
        "No transmitter PRI is asserted and no timing result feeds association",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.diagnostics.open() if line.strip()]
    toa_s = np.asarray([record["toa_s"] for record in records], dtype=float)
    analysis = analyze_toas(toa_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    plot_analysis(toa_s, analysis, args.plot)
    print(args.output)
    print(args.plot)


if __name__ == "__main__":
    main()
