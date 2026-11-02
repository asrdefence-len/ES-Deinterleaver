"""Plot persistent blob-tracker confidence, covariance and gates over time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.result)
    result = json.loads(source.read_text())
    rows = [row for row in result["monitor"] if row["track"] is not None]
    if not rows:
        parser.error("No persistent track was monitored")
    output = Path(args.output) if args.output else source.with_suffix(".png")

    time = np.asarray([row["time_from_start_s"] for row in rows])
    confidence = 100 * np.asarray([row["track"]["track_confidence"] for row in rows])
    aoa = np.asarray([row["track"]["state"]["aoa_deg"] for row in rows])
    aoa_gate = np.asarray([row["track"]["gates"]["aoa_half_width_deg"] for row in rows])
    pw = 1e6 * np.asarray([row["track"]["state"]["pw_median_s"] for row in rows])
    pw_gate = 1e6 * np.asarray([row["track"]["gates"]["pw_half_width_s"] for row in rows])
    pri = 1e6 * np.asarray([row["track"]["state"]["pri_s"] for row in rows])
    pri_gate = 1e6 * np.asarray([row["track"]["gates"]["pri_half_width_s"] for row in rows])
    measurements = np.asarray([row["measurement_count"] for row in rows])
    association = np.asarray([np.nan if row["last_association_confidence"] is None
                              else 100 * row["last_association_confidence"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time, confidence, lw=2.4, label="Operational track confidence")
    ax.scatter(time, association, s=18, alpha=.8, label="Latest blob association")
    ax.axhline(80, color="tab:green", ls="--", lw=1, label="Confirmed threshold")
    ax.axhline(30, color="tab:red", ls="--", lw=1, label="Lost threshold")
    ax.set(title="Persistent track confidence", ylabel="Relative confidence (%)", ylim=(0, 105))
    ax.legend()

    ax = axes[0, 1]
    ax.plot(time, aoa, lw=2.2, label="AOA estimate")
    ax.fill_between(time, aoa - aoa_gate, aoa + aoa_gate, alpha=.16,
                    label="Covariance-derived gate")
    ax.set(title="AOA prediction while updating and coasting", ylabel="AOA (degrees)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(time, pri, lw=2.2, color="tab:purple", label="Base PRI estimate")
    ax.fill_between(time, pri - pri_gate, pri + pri_gate, color="tab:purple", alpha=.16,
                    label="PRI gate")
    ax.set(title="PRI state and uncertainty", ylabel="PRI (µs)")
    ax.legend()

    ax = axes[1, 1]
    ax.step(time, measurements, where="post", lw=2, color="tab:orange",
            label="Blob-train measurements")
    ax.fill_between(time, 0, measurements, step="post", color="tab:orange", alpha=.14)
    ax.set(title="Tracker update opportunities", ylabel="Measurements per window")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlim(0, result["duration_s"])
        ax.set_xlabel("Time from Config 101 start (s)")
        ax.grid(alpha=.25)
    fig.suptitle("Persistent blob-derived emitter track — full Config 101 recording\n"
                 "Covariance grows and confidence decays when blob evidence stops", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
