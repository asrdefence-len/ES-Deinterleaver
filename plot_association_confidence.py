"""Plot sequential association evidence and evaluation diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def values(rows, name):
    return np.asarray([np.nan if row.get(name) is None else row[name] for row in rows], float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--output")
    args = parser.parse_args()
    path = Path(args.result)
    result = json.loads(path.read_text())
    rows = result["windows"]
    evaluation = result["evaluation_windows"]
    x = values(rows, "time_from_start_s")

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.15, 1., 1.]})
    ax = axes[0]
    track = 100 * values(rows, "track_score")
    instant = 100 * values(rows, "instantaneous_score")
    ax.plot(x, track, lw=2.2, color="tab:blue", label="Persistent track score")
    ax.plot(x, instant, lw=1, ls="--", color="0.4", alpha=.8,
            label="Current-window evidence")
    ax.fill_between(x, 0, track, color="tab:blue", alpha=.10)
    empty = values(rows, "candidate_pulses") == 0
    if np.any(empty):
        ax.scatter(x[empty], track[empty], marker="x", color="tab:red", s=35,
                   label="No candidate observation")
    track_aided = values(rows, "track_aided_pulses") > 0
    if np.any(track_aided):
        ax.scatter(x[track_aided], track[track_aided], marker="D",
                   facecolors="white", edgecolors="tab:purple", s=38,
                   linewidths=1.4, label="Track-aided raw-PDW recovery", zorder=4)
    ax.set(ylabel="Association score (%)", ylim=(0, 103),
           title="Persistent association score: one cross-RF emitter hypothesis")
    ax.grid(True, alpha=.18)
    ax.legend(ncol=3, fontsize=9, loc="lower left")

    ax = axes[1]
    components = [("timing_score", "Timing"), ("rf_score", "RF behaviour"),
                  ("signal_score", "AOA / pulse width"),
                  ("continuity_score", "Continuity")]
    for name, label in components:
        ax.plot(x, 100 * values(rows, name), marker="o", ms=2.7, lw=1.2, label=label)
    ax.set(ylabel="Component score (%)", ylim=(0, 103),
           title="Evidence supporting the association")
    ax.grid(True, alpha=.18)
    ax.legend(ncol=4, fontsize=9, loc="lower left")

    ax = axes[2]
    selected = values(rows, "candidate_pulses")
    associated = values(rows, "associated_pulses")
    width = max(np.median(np.diff(x)) * .7 if len(x) > 1 else .03, .005)
    ax.bar(x, selected, width=width, color="0.75", label="Candidate pulses")
    ax.bar(x, associated, width=width, color="tab:green", alpha=.8,
           label="Timing-associated pulses")
    purity = np.asarray([
        100 * row.get("selected_truth_46", 0) / row.get("selected_pulses", 1)
        if row.get("selected_pulses", 0) else np.nan for row in evaluation])
    twin = ax.twinx()
    twin.plot(x, purity, color="tab:orange", lw=1.3, marker=".",
              label="Truth purity (evaluation only)")
    twin.set(ylabel="Evaluation purity (%)", ylim=(0, 103))
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, ncol=3, fontsize=9,
              loc="upper right")
    ax.set(xlabel="Time from Config 101 start (s)", ylabel="Pulse count",
           title="Observations and truth-only evaluation")
    ax.grid(True, axis="y", alpha=.18)

    summary = result["evaluation_summary"]
    purity_total = 100 * summary["selection_purity"]
    coverage_total = 100 * summary["selection_coverage"]
    fig.suptitle("Emitter 46 sequential cross-RF association — "
                 f"selection purity {purity_total:.2f}%, coverage {coverage_total:.2f}%\n"
                 "Score is evidence-weighted and uncalibrated; it is not yet a probability",
                 fontsize=15)
    output = Path(args.output) if args.output else path.with_name("association_confidence.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
