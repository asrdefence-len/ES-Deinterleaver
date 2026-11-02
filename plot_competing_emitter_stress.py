"""Plot separation and ambiguity in the competing-emitter stress test."""
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
    rows = result["non_unclaimed_decisions"]
    source_order = ["real_emitter46", "synthetic_competitor",
                    "deliberately_ambiguous"]
    labels = {"real_emitter46": "Real Emitter 46",
              "synthetic_competitor": "Similar competitor",
              "deliberately_ambiguous": "Midpoint: withheld"}
    colors = {"real_emitter46": "tab:blue",
              "synthetic_competitor": "tab:orange",
              "deliberately_ambiguous": "tab:purple"}
    markers = {"real_emitter46": "o", "synthetic_competitor": "s",
               "deliberately_ambiguous": "X"}
    origin = min(row["toa_s"] for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax = axes[0]
    e46_aoa = float(np.mean([row["aoa_deg"] for row in rows
                             if row["source"] == "real_emitter46"]))
    competitor_aoa = float(np.mean([row["aoa_deg"] for row in rows
                                    if row["source"] == "synthetic_competitor"]))
    ax.axhspan(e46_aoa - 1.5, e46_aoa + 1.5, color="tab:blue", alpha=.08)
    ax.axhspan(competitor_aoa - 1.5, competitor_aoa + 1.5,
               color="tab:orange", alpha=.08)
    ax.axhline(e46_aoa, color="tab:blue", ls="--", lw=1,
               label="E46 predicted AOA")
    ax.axhline(competitor_aoa, color="tab:orange", ls="--", lw=1,
               label="Competitor predicted AOA")
    for source in source_order:
        selected = [row for row in rows if row["source"] == source]
        ax.scatter([(row["toa_s"] - origin) * 1e3 for row in selected],
                   [row["aoa_deg"] for row in selected], s=70,
                   marker=markers[source], color=colors[source],
                   edgecolors="white", linewidths=.7, label=labels[source], zorder=3)
    for row in rows:
        if row["status"] == "confirmed":
            ax.annotate(f"{row['selected_emitter_id']} confirmed",
                ((row["toa_s"] - origin) * 1e3, row["aoa_deg"]),
                xytext=(5, 10), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.7, color="0.35"))
    ax.set(title="Interleaved pulses and predicted AOA bands",
           xlabel="Time from first evaluated pulse (ms)", ylabel="AOA (degrees)")
    ax.grid(True, alpha=.18)
    ax.legend(fontsize=8, ncol=2, loc="best")

    ax = axes[1]
    for source in source_order:
        selected = [row for row in rows if row["source"] == source]
        x, y = [], []
        for row in selected:
            scores = {item["emitter_id"]: item["score"] for item in row["candidates"]}
            x.append(scores["E46_TRACK"])
            y.append(scores["COMPETITOR_TRACK"])
        ax.scatter(x, y, s=75, marker=markers[source], color=colors[source],
                   edgecolors="white", linewidths=.7, label=labels[source], zorder=3)
    limits = (.5, 1.01)
    ax.plot(limits, limits, color="0.35", ls="--", lw=1,
            label="Equal evidence: withhold")
    ax.fill_between(limits, [value - .08 for value in limits],
                    [value + .08 for value in limits], color="0.5", alpha=.08,
                    label="Ambiguity margin")
    ax.annotate("6 competitor pulses", (.72, .977), xytext=(7, -12),
                textcoords="offset points", fontsize=8)
    ax.annotate("3 ambiguous pulses", (.901, .901), xytext=(-92, 8),
                textcoords="offset points", fontsize=8)
    ax.annotate("5 E46 pulses", (.985, .71), xytext=(-66, -16),
                textcoords="offset points", fontsize=8)
    ax.set(title="Competing track-association scores",
           xlabel="Score against Emitter 46 track",
           ylabel="Score against competitor track", xlim=limits, ylim=limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=.18)
    ax.legend(fontsize=8, loc="lower right")

    summary = result["summary"]
    background = summary.get("real_background -> unclaimed", 0)
    fig.suptitle("Track-aided association with a deliberately similar emitter\n"
                 f"0 wrong-track assignments · {background:,} background PDWs rejected · "
                 "3 midpoint pulses withheld", fontsize=14)
    output = Path(args.output) if args.output else path.with_name(
        "competing_emitter_stress.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
