"""Plot state confidence, covariance and gates from the stateful tracker."""
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
    path = Path(args.result)
    result = json.loads(path.read_text())
    t = np.asarray([row["time_from_start_s"] for row in result["monitor"]])
    by_track = {}
    for row in result["monitor"]:
        for track in row["tracks"]:
            by_track.setdefault(track["emitter_id"], []).append(track)
    colors = {"E46_TRACK": "tab:blue", "COMPETITOR_TRACK": "tab:orange"}
    labels = {"E46_TRACK": "Emitter 46", "COMPETITOR_TRACK": "Competitor"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True,
                             constrained_layout=True)
    for ax in axes.flat:
        for start, end in result["design"]["gaps_s"]:
            ax.axvspan(start, end, color="0.5", alpha=.10)

    ax = axes[0, 0]
    for name, rows in by_track.items():
        confidence = 100 * np.asarray([row["existence_confidence"] for row in rows])
        association = np.asarray([np.nan if row["last_innovation"] is None else
            100 * row["last_innovation"]["association_confidence"] for row in rows])
        ax.plot(t, confidence, color=colors[name], lw=2, label=f"{labels[name]} existence")
        ax.plot(t, association, color=colors[name], lw=1, ls="--",
                label=f"{labels[name]} last update")
    ax.set(title="Track and association confidence", ylabel="Relative confidence (%)",
           ylim=(-2, 103))
    ax.grid(True, alpha=.18)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[0, 1]
    for name, rows in by_track.items():
        mean = np.asarray([row["state"]["aoa_deg"] for row in rows])
        half = np.asarray([row["gates"]["aoa_half_width_deg"] for row in rows])
        ax.plot(t, mean, color=colors[name], lw=2, label=labels[name])
        ax.fill_between(t, mean - half, mean + half, color=colors[name], alpha=.10,
                        label=f"{labels[name]} 3σ gate")
    ax.set(title="AOA state and covariance-derived gate", ylabel="AOA (degrees)")
    ax.grid(True, alpha=.18)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    for name, rows in by_track.items():
        mean = 1e6 * np.asarray([row["state"]["pulse_width_s"] for row in rows])
        half = 1e6 * np.asarray([row["gates"]["pulse_width_half_width_s"] for row in rows])
        ax.plot(t, mean, color=colors[name], lw=2, label=labels[name])
        ax.fill_between(t, mean - half, mean + half, color=colors[name], alpha=.10)
    ax.set(title="Pulse-width state and covariance-derived gate",
           xlabel="Time from Config 101 start (s)", ylabel="Pulse width (µs)")
    ax.grid(True, alpha=.18)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, rows in by_track.items():
        aoa_sigma = np.sqrt([row["covariance"]["aoa_variance_deg2"] for row in rows])
        pw_sigma = 1e6 * np.sqrt([row["covariance"]["pulse_width_variance_s2"]
                                  for row in rows])
        ax.plot(t, aoa_sigma, color=colors[name], lw=1.8,
                label=f"{labels[name]} AOA σ (deg)")
        ax.plot(t, pw_sigma, color=colors[name], lw=1.2, ls="--",
                label=f"{labels[name]} PW σ (µs)")
    ax.set(title="State covariance grows while coasting and contracts on updates",
           xlabel="Time from Config 101 start (s)", ylabel="1σ uncertainty")
    ax.grid(True, alpha=.18)
    ax.legend(ncol=2, fontsize=8)

    summary = result["summary"]
    wrong = summary["e46_wrong_track"] + summary["competitor_wrong_track"]
    fig.suptitle(f"Stateful emitter tracker monitor — {summary['duration_s']:.2f} s\n"
        f"wrong-track assignments {wrong:,} · background false claims "
        f"{summary['background_false_claims']:,} · shaded regions are competitor gaps",
        fontsize=14)
    output = Path(args.output) if args.output else path.with_name(
        "stateful_tracker_monitor.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
