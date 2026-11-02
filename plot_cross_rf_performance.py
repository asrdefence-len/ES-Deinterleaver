"""Create a focused tracker-performance report for a cross-RF experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("result")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result_path = Path(args.result)
    result = json.loads(result_path.read_text())
    family = result["families"][0]
    train_metrics = result["train_metrics"]
    window, window_size = result["window"], 10000

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        start = window * window_size
        data = np.asarray(handle["data"][start:start + window_size], float)
    populations, count = batch_aoa_populations(data[:, names.index("AoA")])
    blobs = group_window(data, names, populations, count, 8, .20, 6., 8)
    selected = np.isin(blobs, result["selected_blobs"])
    selected_indices = np.flatnonzero(selected)
    pdw_ids = start + selected_indices + 1
    associated_ids = {pdw_id for train in train_metrics for pdw_id in train["pdw_ids"]}
    associated = np.array([pdw_id in associated_ids for pdw_id in pdw_ids])
    toa_ms = (data[selected, names.index("ToA")] - data[selected, names.index("ToA")].min()) / 1000.
    rf_ghz = data[selected, names.index("Frequency")] / 1000.
    selected_blobs = blobs[selected]

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    ax = axes[0, 0]
    palette = plt.get_cmap("tab10")
    for index, blob in enumerate(sorted(result["selected_blobs"])):
        mask = selected_blobs == blob
        ax.scatter(toa_ms[mask & associated], rf_ghz[mask & associated], s=10,
                   color=palette(index % 10), label=f"Blob {blob}", alpha=.8)
        ax.scatter(toa_ms[mask & ~associated], rf_ghz[mask & ~associated], s=9,
                   facecolors="none", edgecolors=palette(index % 10), alpha=.45)
    ax.set(title="Eight RF groups retained in candidate family F1",
           xlabel="TOA from focused-window start (ms)", ylabel="Frequency (GHz)")
    ax.grid(True, alpha=.18)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[0, 1]
    ordered = sorted(train_metrics, key=lambda metric:
                     (metric["blob_id"], metric["train_id"]))
    train_y = {metric["train_id"]: index for index, metric in enumerate(ordered)}
    blob_ids = sorted({metric["blob_id"] for metric in ordered})
    blob_y = {blob: float(np.mean([train_y[metric["train_id"]]
                    for metric in ordered if metric["blob_id"] == blob]))
              for blob in blob_ids}
    family_y = .5 * (len(ordered) - 1)
    for metric in ordered:
        ty, by = train_y[metric["train_id"]], blob_y[metric["blob_id"]]
        ax.plot([0, 1], [by, ty], color="0.65", lw=1)
        ax.plot([1, 2], [ty, family_y], color="tab:green", lw=1, alpha=.65)
    ax.scatter(np.zeros(len(blob_ids)), [blob_y[blob] for blob in blob_ids],
               s=70, color="tab:blue", zorder=3)
    ax.scatter(np.ones(len(ordered)), list(range(len(ordered))),
               s=70, color="tab:orange", zorder=3)
    ax.scatter([2], [family_y], s=180, color="tab:green", zorder=3)
    for blob in blob_ids:
        frequencies = [metric["frequency_hz"] / 1e9 for metric in ordered
                       if metric["blob_id"] == blob]
        ax.annotate(f"B{blob}  {np.mean(frequencies):.1f} GHz", (0, blob_y[blob]),
                    xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8)
    for metric in ordered:
        label = metric["train_id"].split(":", 1)[1]
        ax.annotate(label, (1, train_y[metric["train_id"]]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.annotate(f"F1\n{len(ordered)} trains\n{family['phase_cluster_count']} phases",
                (2, family_y), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8)
    ax.set(title=f"Integration path: {len(blob_ids)} blobs → "
                 f"{len(ordered)} trains → candidate family F1",
           xlim=(-.35, 2.3), ylim=(-.7, len(ordered) - .3))
    ax.set_xticks([0, 1, 2], ["RF blobs", "Pulse trains", "Candidate family"])
    ax.set_yticks([])
    ax.grid(False)

    ax = axes[1, 0]
    y = np.arange(len(train_metrics))
    raw_pri = np.array([metric["pri_s"] for metric in train_metrics]) * 1e6
    base_pri = family["base_pri_s"] * 1e6
    harmonic = np.maximum(1, np.rint(raw_pri / base_pri)).astype(int)
    normalised = raw_pri / harmonic
    ax.scatter(y, raw_pri, marker="o", s=55, label="Discovered PRI")
    ax.scatter(y, normalised, marker="D", s=38, label="Harmonic-normalised PRI")
    ax.axhline(base_pri, color="black", ls="--", lw=1,
               label=f"Family base PRI {base_pri:.3f} µs")
    ax.set(title="PRI-family integration", xlabel="Discovered train", ylabel="PRI (µs)")
    ax.set_xticks(y, [metric["train_id"].split(":", 1)[1] for metric in train_metrics],
                  rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=.18)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    mean = np.array([metric["mean_residual_s"] for metric in train_metrics]) * 1e6
    mae = np.array([metric["mean_absolute_residual_s"] for metric in train_metrics]) * 1e6
    rms = np.array([metric["rms_residual_s"] for metric in train_metrics]) * 1e6
    width = .34
    ax.bar(y - width / 2, mae, width, label="Mean absolute error")
    ax.bar(y + width / 2, rms, width, label="RMS error")
    ax.scatter(y, mean, marker="x", color="black", label="Signed mean error", zorder=3)
    ax.axhline(0, color="black", lw=.6)
    ax.set(title="Pulse-to-train timing residuals",
           xlabel="Discovered train", ylabel="Residual error (µs)")
    ax.set_xticks(y, [metric["train_id"].split(":", 1)[1] for metric in train_metrics],
                  rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=.18)
    ax.legend(fontsize=8)

    coverage = 100 * result["associated_pdws"] / result["selected_pdws"]
    truth_count = next(iter(family.get("truth_counts_eval", {}).values()), 0)
    purity = 100 * truth_count / family["pdw_count"] if family["pdw_count"] else 0
    fig.suptitle(f"Emitter 46 focused cross-RF tracker performance — "
                 f"coverage {coverage:.1f}%, family purity {purity:.1f}%, "
                 f"unexplained {result['unexplained_pdws']}/{result['selected_pdws']}",
                 fontsize=15)
    output = Path(args.output) if args.output else result_path.with_name("tracker_performance.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output.with_name("train_residual_metrics.csv")
    fields = ["train_id", "blob_id", "frequency_ghz", "pri_us", "normalised_pri_us",
              "harmonic", "pulse_count", "mean_residual_us", "mae_us", "rms_us"]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, raw, norm, h, signed, absolute, root_mean_square in zip(
                train_metrics, raw_pri, normalised, harmonic, mean, mae, rms):
            writer.writerow({"train_id": metric["train_id"], "blob_id": metric["blob_id"],
                "frequency_ghz": metric["frequency_hz"] / 1e9, "pri_us": raw,
                "normalised_pri_us": norm, "harmonic": h,
                "pulse_count": metric["pulse_count"], "mean_residual_us": signed,
                "mae_us": absolute, "rms_us": root_mean_square})
    print(output)
    print(metrics_path)


if __name__ == "__main__":
    main()
