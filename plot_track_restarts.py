"""Plot truth-aligned AOA/time waterfalls around injected emitter dropouts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history")
    parser.add_argument("summary")
    parser.add_argument("--output", default="track_restarts.png")
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.history).open()))
    gaps = json.loads(Path(args.summary).read_text())["dropouts"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    for ax, gap in zip(axes.flat, gaps):
        emitter = str(gap["truth_emitter"])
        selected = [row for row in rows
                    if row["truth_emitter"] == emitter and row["track_id"]]
        times = np.array([float(row["time_s"]) for row in selected])
        aoas = np.array([float(row["aoa_deg"]) for row in selected])
        ax.plot(aoas, times, "--", color="black", lw=.7, alpha=.35,
                label=f"Truth emitter {emitter}")
        track_ids = list(dict.fromkeys(row["track_id"] for row in selected))
        for index, track_id in enumerate(track_ids):
            track_rows = [row for row in selected if row["track_id"] == track_id]
            track_rows.sort(key=lambda row: float(row["time_s"]))
            x = [float(row["aoa_deg"]) for row in track_rows]
            y = [float(row["time_s"]) for row in track_rows]
            ax.plot(x, y, "o-", ms=2, lw=1.2,
                    color=plt.get_cmap("tab10")(index % 10), label=track_id)
        ax.axhspan(gap["start_s"], gap["end_s"], color="tab:red", alpha=.10,
                   label="Injected dropout")
        padding = max(1., .12 * float(np.ptp(aoas)))
        ax.set_xlim(float(np.min(aoas) - padding), float(np.max(aoas) + padding))
        ax.set_ylim(max(0., gap["start_s"] - 3.), min(60., gap["end_s"] + 3.))
        ax.set_title(f"Truth emitter {emitter}: track restart")
        ax.set_xlabel("AOA (degrees)")
        ax.set_ylabel("Simulated time (s) — newest at top")
        ax.grid(True, alpha=.18)
        ax.legend(fontsize=8, loc="best")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
