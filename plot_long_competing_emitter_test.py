"""Plot the multi-second competing-emitter association test."""
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
    timeline, summary = result["timeline"], result["summary"]
    t = np.asarray(timeline["time_s"])
    has_ambiguity = (summary["e46_ambiguous"] +
                     summary["competitor_ambiguous"]) > 0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
        constrained_layout=True, gridspec_kw={"height_ratios": [1, 1.1, .8]})
    for ax in axes:
        for start, end in result["design"]["gaps_s"]:
            ax.axvspan(start, end, color="0.5", alpha=.10)
    ax = axes[0]
    ax.axhline(timeline["e46_reference_aoa"], color="tab:blue", ls="--",
               lw=1, label="E46 reference AOA")
    ax.plot(t, timeline["e46_predicted_aoa"], color="tab:blue", lw=2,
            label="E46 predicted AOA")
    ax.plot(t, timeline["competitor_true_aoa"], color="tab:orange", ls="--",
            lw=1.3, label="Competitor generated AOA")
    ax.plot(t, timeline["competitor_predicted_aoa"], color="tab:orange", lw=2,
            label="Competitor predicted AOA")
    ax.axvline(result["design"]["closest_approach_s"], color="tab:purple",
               ls=":", lw=1.2, label="Closest approach")
    ax.set(title=("Track predictions through approach, ambiguity and gaps"
                  if has_ambiguity else
                  "Track predictions through separation and signal gaps"),
           ylabel="AOA (degrees)")
    ax.grid(True, alpha=.18)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1]
    ax.plot(t, 100 * np.asarray(timeline["e46_correct_rate"]), color="tab:blue",
            lw=1.8, label="E46 correctly associated")
    ax.plot(t, 100 * np.asarray(timeline["competitor_correct_rate"]),
            color="tab:orange", lw=1.8, label="Competitor correctly associated")
    ax.plot(t, 100 * np.asarray(timeline["e46_ambiguous_rate"]),
            color="tab:blue", ls="--", lw=1, label="E46 withheld ambiguous")
    ax.plot(t, 100 * np.asarray(timeline["competitor_ambiguous_rate"]),
            color="tab:orange", ls="--", lw=1, label="Competitor withheld ambiguous")
    ax.set(title="Association outcome in 100 ms intervals",
           ylabel="Pulses in class (%)", ylim=(-2, 103))
    ax.grid(True, alpha=.18)
    ax.legend(ncol=2, fontsize=8, loc="lower left")

    ax = axes[2]
    wrong = np.cumsum(np.asarray(timeline["e46_wrong"]) +
                      np.asarray(timeline["competitor_wrong"]))
    ax.step(t, wrong, where="mid", color="tab:red", lw=2,
            label="Cumulative wrong-track assignments")
    ax.bar(t, timeline["e46_total"], width=result["bin_s"] * .8,
           color="tab:blue", alpha=.22, label="E46 pulse opportunities")
    ax.bar(t, timeline["competitor_total"], width=result["bin_s"] * .55,
           color="tab:orange", alpha=.28, label="Competitor pulse opportunities")
    ax.set(title="Identity errors and observation opportunities",
           xlabel="Time from Config 101 start (s)", ylabel="Count")
    ax.grid(True, axis="y", alpha=.18)
    ax.legend(ncol=3, fontsize=8)

    total_wrong = summary["e46_wrong_track"] + summary["competitor_wrong_track"]
    fig.suptitle(f"Long-duration two-emitter "
        f"{'stress' if has_ambiguity else 'baseline'} test — {result['duration_s']:.2f} s\n"
        f"wrong-track assignments {total_wrong:,} · background false claims "
        f"{summary['background_false_claims']:,} · deliberate gaps shaded", fontsize=14)
    output = Path(args.output) if args.output else path.with_name(
        "long_competing_emitter.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
