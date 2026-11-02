"""Plot blob-by-blob confidence, state covariance and association gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("monitor", help="blob_monitor.jsonl from run_blob_tracker.py")
    parser.add_argument("--output")
    args = parser.parse_args()

    source = Path(args.monitor)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("track") is not None]
    if not rows:
        parser.error("Monitor contains no assigned blob-track updates")
    output = Path(args.output) if args.output else source.with_name("blob_tracker_monitor.png")

    x = np.arange(1, len(rows) + 1)
    labels = [f"B{row['blob_id']}" for row in rows]
    confidence = 100 * np.asarray([row["best_confidence"] for row in rows])
    existence = 100 * np.asarray([row["track"]["track_confidence"] for row in rows])
    aoa = np.asarray([row["track"]["state"]["aoa_deg"] for row in rows])
    aoa_gate = np.asarray([row["track"]["gates"]["aoa_half_width_deg"] for row in rows])
    pw = 1e6 * np.asarray([row["track"]["state"]["pw_median_s"] for row in rows])
    pw_gate = 1e6 * np.asarray([row["track"]["gates"]["pw_half_width_s"] for row in rows])
    aoa_sigma = np.sqrt([row["track"]["covariance"]["aoa_variance_deg2"]
                         for row in rows])
    pw_sigma = 1e6 * np.sqrt([row["track"]["covariance"]["pw_variance_s2"]
                              for row in rows])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, confidence, "o-", lw=2, label="Association confidence")
    ax.plot(x, existence, "s--", lw=1.8, label="Operational track confidence")
    ax.set(title="One update per blob", ylabel="Relative confidence (%)", ylim=(0, 105))
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, aoa, "o-", lw=2, color="tab:blue", label="AOA estimate")
    ax.fill_between(x, aoa - aoa_gate, aoa + aoa_gate, color="tab:blue", alpha=.16,
                    label="Covariance-derived gate")
    ax.set(title="AOA state and gate", ylabel="AOA (degrees)")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(x, pw, "o-", lw=2, color="tab:orange", label="Pulse-width estimate")
    ax.fill_between(x, pw - pw_gate, pw + pw_gate, color="tab:orange", alpha=.16,
                    label="Covariance-derived gate")
    ax.set(title="Pulse-width state and gate", ylabel="Pulse width (µs)")
    ax.legend()

    timing_rows = [row for row in rows if row["track"]["state"].get("pri_s") is not None]
    tx = np.asarray([rows.index(row) + 1 for row in timing_rows])
    tracked_pri = 1e6 * np.asarray([row["track"]["state"]["pri_s"]
                                    for row in timing_rows])
    measured_pri = 1e6 * np.asarray([row["measurement"]["pri_s"]
                                     for row in timing_rows])
    pri_gate = 1e6 * np.asarray([row["track"]["gates"]["pri_half_width_s"]
                                 for row in timing_rows])
    ax = axes[1, 0]
    ax.plot(tx, tracked_pri, "o-", lw=2, label="Tracked PRI")
    ax.scatter(tx, measured_pri, marker="x", s=55, label="Blob-train PRI")
    ax.fill_between(tx, tracked_pri - pri_gate, tracked_pri + pri_gate,
                    color="tab:blue", alpha=.14, label="Covariance-derived gate")
    ax.set(title="PRI timing state and gate", ylabel="PRI (µs)")
    ax.legend()

    ax = axes[1, 1]
    residual = 1e6 * np.asarray([row["measurement"]["timing_rms_residual_s"]
                                 for row in timing_rows])
    ax.plot(tx, residual, "o-", lw=2, color="tab:green")
    ax.axhline(5., ls="--", lw=1.5, color="tab:red", label="5 µs association tolerance")
    ax.set(title="Timing fit quality exported by each blob", ylabel="RMS residual (µs)")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(x, aoa_sigma, "o-", lw=2, label="AOA σ (degrees)")
    ax.plot(x, pw_sigma, "s--", lw=1.8, label="PW σ (µs)")
    ax.set(title="State uncertainty after each blob update", ylabel="1σ uncertainty")
    ax.legend()

    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_xlabel("Blob update order")
        ax.grid(alpha=.25)
    fig.suptitle("Covariance-based blob-derived emitter tracker\n"
                 "No individual PDWs are processed by the tracker", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
