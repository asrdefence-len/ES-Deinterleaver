"""Plot HMM Model 2 state uncertainty, PW modes and predictive evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_hmm_model1_performance import posterior_entropy_bits, rolling_median


def plot_performance(
    records: list[dict],
    output: Path,
    *,
    rolling_window: int = 200,
    title: str = "HMM Model 2 — E46 uncertainty and predictive performance",
) -> None:
    if not records:
        raise ValueError("No Model 2 diagnostic records supplied")
    required = {
        "pulse_width_s",
        "pulse_width_posterior",
        "pulse_width_map_mean_s",
        "pulse_width_log_evidence",
        "model1_log_evidence",
    }
    missing = sorted(required - set(records[0]))
    if missing:
        raise ValueError(f"Not Model 2 diagnostics; missing: {', '.join(missing)}")

    toa_s = np.asarray([float(item["toa_s"]) for item in records])
    time_s = toa_s - toa_s[0]
    aoa_variance = np.square([
        float(item["aoa_sigma_deg"]) for item in records
    ])
    rf_entropy = np.asarray([
        posterior_entropy_bits(item["rf_posterior"]) for item in records
    ])
    pulse_width_us = np.asarray([
        float(item["pulse_width_s"]) * 1e6 for item in records
    ])
    pulse_width_map_us = np.asarray([
        float(item["pulse_width_map_mean_s"]) * 1e6 for item in records
    ])
    pulse_width_entropy = np.asarray([
        posterior_entropy_bits(item["pulse_width_posterior"])
        for item in records
    ])
    gap_us = np.asarray([
        np.nan if item["shadow_gap_s"] is None
        else float(item["shadow_gap_s"]) * 1e6
        for item in records
    ])
    coast_probability = np.asarray([
        float(item["shadow_coast_posterior"]) for item in records
    ])
    pulse_width_evidence = np.asarray([
        float(item["pulse_width_log_evidence"]) for item in records
    ])
    state_birth = np.asarray([
        bool(item["pulse_width_state_created"]) for item in records
    ])

    figure, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True,
                                constrained_layout=True)
    axes[0].plot(time_s, aoa_variance, linewidth=0.8)
    axes[0].set(ylabel="AOA covariance\n(deg²)",
                title="Continuous-state posterior covariance")

    axes[1].plot(time_s, rf_entropy, color="tab:orange", linewidth=0.8)
    axes[1].set(ylabel="RF entropy\n(bits)",
                title="Discrete RF-state posterior uncertainty")

    axes[2].scatter(time_s, pulse_width_us, s=2.0, alpha=0.28,
                    label="Observed PW")
    axes[2].scatter(time_s, pulse_width_map_us, s=2.0, color="tab:red",
                    alpha=0.55, label="MAP PW-state mean")
    axes[2].set(ylabel="Pulse width (µs)",
                title="Pulse-width HMM observations and selected state")
    if float(np.max(pulse_width_entropy)) > 1e-6:
        entropy_axis = axes[2].twinx()
        entropy_axis.plot(time_s, pulse_width_entropy, color="tab:purple",
                          linewidth=0.5, alpha=0.6)
        entropy_axis.set_ylabel("PW entropy (bits)", color="tab:purple")
    else:
        axes[2].text(
            0.99, 0.94, "PW posterior entropy ≈ 0",
            transform=axes[2].transAxes, ha="right", va="top",
            color="tab:purple", fontsize=8,
        )
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].plot(time_s, gap_us, color="tab:green", linewidth=0.45,
                 label="Detected inter-arrival gap")
    axes[3].set_yscale("log")
    axes[3].set(ylabel="Observed gap (µs)",
                title="TOA timing shadow — feedback remains zero")
    visibility_axis = axes[3].twinx()
    visibility_axis.plot(time_s, coast_probability, color="tab:purple",
                         linewidth=0.45, alpha=0.65)
    visibility_axis.set_ylabel("Coast probability", color="tab:purple")
    visibility_axis.set_ylim(-0.03, 1.03)
    axes[3].legend(loc="upper left", fontsize=8)

    axes[4].plot(time_s, pulse_width_evidence, color="0.72", linewidth=0.4,
                 label="Per-pulse PW evidence")
    axes[4].plot(
        time_s,
        rolling_median(pulse_width_evidence, rolling_window),
        color="tab:red",
        linewidth=1.25,
        label=f"PW median ({rolling_window} pulses)",
    )
    if state_birth.any():
        axes[4].scatter(
            time_s[state_birth],
            pulse_width_evidence[state_birth],
            marker="x",
            color="tab:purple",
            s=28,
            label="PW state birth (not calibrated)",
        )
    axes[4].set(
        xlabel="Time from first selected E46 pulse (s)",
        ylabel="Log evidence",
        title=(
            "Incremental PW predictive fit in log-width space "
            "(not an accuracy score)"
        ),
    )
    axes[4].legend(loc="lower right", fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{title}\n{len(records):,} truth-isolated pulses; amplitude excluded; "
        "TOA timing feedback weight 0",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=200)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in args.diagnostics.open() if line.strip()
    ]
    plot_performance(
        records,
        args.output,
        rolling_window=args.rolling_window,
    )
    print(args.output)


if __name__ == "__main__":
    main()
