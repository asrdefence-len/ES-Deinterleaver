"""Plot two persistent blob-derived tracks across the full recording."""
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
    output = Path(args.output) if args.output else source.with_suffix(".png")
    rows = result["monitor"]
    family_names = list(result["families"])
    colors = dict(zip(family_names, ("tab:blue", "tab:orange")))

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    for name in family_names:
        valid = [row for row in rows if name in row["tracks"]]
        t = np.asarray([row["time_from_start_s"] for row in valid])
        confidence = 100 * np.asarray([row["tracks"][name]["track_confidence"]
                                       for row in valid])
        association = np.asarray([np.nan if row["latest_association_confidence"][name] is None
            else 100 * row["latest_association_confidence"][name] for row in valid])
        axes[0, 0].plot(t, confidence, lw=2.3, color=colors[name],
                        label=f"{name} existence probability")
        axes[0, 0].scatter(t, association, s=16, alpha=.65, color=colors[name],
                           label=f"{name} blob association")

        aoa = np.asarray([row["tracks"][name]["state"]["aoa_deg"] for row in valid])
        gate = np.asarray([row["tracks"][name]["gates"]["aoa_half_width_deg"]
                           for row in valid])
        axes[0, 1].plot(t, aoa, lw=2.3, color=colors[name], label=f"{name} AOA")
        axes[0, 1].fill_between(t, aoa - gate, aoa + gate, color=colors[name], alpha=.13)

        pri = 1e6 * np.asarray([row["tracks"][name]["state"]["pri_s"] for row in valid])
        pri_gate = 1e6 * np.asarray([row["tracks"][name]["gates"]["pri_half_width_s"]
                                     for row in valid])
        axes[1, 0].plot(t, pri, lw=2.3, color=colors[name], label=f"{name} base PRI")
        axes[1, 0].fill_between(t, pri - pri_gate, pri + pri_gate,
                                color=colors[name], alpha=.13)

        pw = 1e6 * np.asarray([row["tracks"][name]["state"]["pw_median_s"]
                               for row in valid])
        pw_gate = 1e6 * np.asarray([
            row["tracks"][name]["gates"]["pw_half_width_s"] for row in valid])
        axes[1, 1].plot(t, pw, lw=2.3, color=colors[name], label=f"{name} PW")
        axes[1, 1].fill_between(t, pw - pw_gate, pw + pw_gate,
                                color=colors[name], alpha=.13)

        nis_ratio = np.asarray([np.nan if row["track_updates"][name] == 0
            or row["tracks"][name]["last_innovation"]["nis"] is None else
            row["tracks"][name]["last_innovation"]["nis"] /
            row["tracks"][name]["last_innovation"]["chi_square_gate"]
            for row in valid])
        axes[2, 0].scatter(t, nis_ratio, s=18, alpha=.7,
                           color=colors[name], label=f"{name} accepted update")

        count = np.asarray([row["measurement_counts"][name] for row in valid])
        axes[2, 1].step(t, count, where="post", lw=2, color=colors[name],
                        label=f"{name} measurements")

    axes[0, 0].axhline(80, color="tab:green", ls="--", lw=1)
    axes[0, 0].axhline(30, color="tab:red", ls="--", lw=1)
    axes[0, 0].set(title="Bernoulli track existence probability",
                   ylabel="Existence probability (%)",
                   ylim=(0, 105))
    axes[0, 1].set(title="AOA states and covariance-derived gates", ylabel="AOA (degrees)")
    axes[1, 0].set(title="Base PRI states and gates", ylabel="PRI (µs)")
    axes[1, 1].set(title="Pulse-width states and gates", ylabel="Pulse width (µs)")
    axes[2, 0].axhline(1., color="tab:red", ls="--", lw=1,
                       label="Chi-square gate")
    axes[2, 0].set(title="Normalized innovation squared", ylabel="NIS / gate threshold",
                   ylim=(0, 1.1))
    axes[2, 1].set(title="Blob-train updates per processing window",
                   ylabel="Measurements per window")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[2, 0], axes[2, 1]):
        ax.set_xlim(0, result["duration_s"])
        ax.set_xlabel("Time from Config 101 start (s)")
        ax.grid(alpha=.25)
        ax.legend()
    cross = result["cross_assigned_measurements"]
    retained = result["canonical_primary_retention"]
    cross_text = " · ".join(f"{key.replace('_to_', '→')} {value}"
                            for key, value in cross.items())
    retention_text = ", ".join(f"{name} {retained[name]['retained']}/"
        f"{retained[name]['total']}" for name in family_names)
    fig.suptitle(f"Two real emitters tracked simultaneously — {result['scenario']} scenario\n"
        f"cross-assigned measurements: {cross_text} · primary-track retention: "
        f"{retention_text}", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
