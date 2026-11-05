"""Plot uncertainty and predictive performance for HMM emitter Model 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def posterior_entropy_bits(probabilities: list[float]) -> float:
    """Return Shannon entropy without treating absent states as uncertainty."""
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    return float(-np.sum(values * np.log2(values))) if len(values) else 0.0


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing median with a shorter window at the beginning of the run."""
    if window <= 0:
        raise ValueError("rolling window must be positive")
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = float(np.median(values[start:index + 1]))
    return result


def load_diagnostics(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
    if not records:
        raise ValueError(f"no diagnostic records in {path}")
    return records


def plot_performance(
    records: list[dict],
    output: Path,
    *,
    rolling_window: int = 200,
    title: str = "HMM Model 1 — E46 uncertainty and predictive performance",
) -> None:
    first_toa = float(records[0]["toa_s"])
    time_s = np.asarray([float(item["toa_s"]) - first_toa for item in records])
    aoa_variance = np.asarray([float(item["aoa_sigma_deg"]) ** 2 for item in records])
    rf_entropy = np.asarray([
        posterior_entropy_bits(item["rf_posterior"]) for item in records
    ])
    shadow_timing = "shadow_gap_s" in records[0]
    if shadow_timing:
        timing_pri_us = np.asarray([
            np.nan if item["shadow_gap_s"] is None
            else float(item["shadow_gap_s"]) * 1e6
            for item in records
        ])
        timing_mean_us = timing_pri_us
        timing_sigma_us = np.full(len(records), np.nan)
        coast_probability = np.asarray([
            float(item["shadow_coast_posterior"]) for item in records
        ])
    else:
        timing_pri_us = np.asarray([
            np.nan if item["timing_map_pri_s"] is None
            else float(item["timing_map_pri_s"]) * 1e6
            for item in records
        ])
        timing_mean_us = np.asarray([
            timing_pri_us[index]
            if item.get("timing_posterior_mean_pri_s") is None
            else float(item["timing_posterior_mean_pri_s"]) * 1e6
            for index, item in enumerate(records)
        ])
        timing_sigma_us = np.asarray([
            np.nan
            if item.get("timing_posterior_sigma_s") is None
            else float(item["timing_posterior_sigma_s"]) * 1e6
            for item in records
        ])
        coast_probability = np.asarray([
            np.nan
            if item.get("timing_coast_posterior") is None
            else float(item["timing_coast_posterior"])
            for item in records
        ])
    evidence = np.asarray([float(item["total_log_evidence"]) for item in records])
    evidence_median = rolling_median(evidence, rolling_window)

    figure, axes = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True, constrained_layout=True,
    )

    axes[0].plot(time_s, aoa_variance, color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel("AOA covariance\n(deg²)")
    axes[0].set_title("Continuous-state posterior covariance")

    axes[1].plot(time_s, rf_entropy, color="tab:orange", linewidth=0.8)
    axes[1].set_ylabel("RF entropy\n(bits)")
    axes[1].set_title("Discrete RF-state posterior uncertainty (not covariance)")

    axes[2].plot(
        time_s, timing_pri_us, color="tab:green", linewidth=0.55,
        label="Detected inter-arrival gap" if shadow_timing else "MAP PRI mode",
    )
    if not shadow_timing:
        axes[2].plot(
            time_s, timing_mean_us, color="tab:olive", linewidth=0.65,
            linestyle="--", alpha=0.8, label="Posterior mean",
        )
    if np.isfinite(timing_sigma_us).any():
        axes[2].fill_between(
            time_s,
            timing_mean_us - timing_sigma_us,
            timing_mean_us + timing_sigma_us,
            color="tab:green",
            alpha=0.18,
            label="±1 posterior sigma",
        )
    axes[2].set_ylabel("Observed gap (µs)" if shadow_timing else "PRI (µs)")
    axes[2].set_title(
        "Detected-pulse timing shadow — observed gaps are not asserted as PRI"
        if shadow_timing else
        "TOA timing posterior — convergence alone does not establish the fundamental PRI"
    )
    if shadow_timing:
        axes[2].set_yscale("log")
    axes[2].legend(loc="upper left", fontsize=8)
    if np.isfinite(coast_probability).any():
        visibility_axis = axes[2].twinx()
        visibility_axis.plot(
            time_s, coast_probability, color="tab:purple", linewidth=0.6,
            alpha=0.7, label="Coast posterior",
        )
        visibility_axis.set_ylabel("Coast probability", color="tab:purple")
        visibility_axis.set_ylim(-0.03, 1.03)
        visibility_axis.tick_params(axis="y", colors="tab:purple")

    axes[3].plot(
        time_s, evidence, color="0.75", linewidth=0.35, label="Per-pulse evidence",
    )
    axes[3].plot(
        time_s, evidence_median, color="tab:red", linewidth=1.2,
        label=f"Trailing median ({rolling_window} pulses)",
    )
    axes[3].set_ylabel("Log evidence")
    axes[3].set_xlabel("Time from first selected E46 pulse (s)")
    axes[3].set_title(
        "Active predictive fit of Frequency + AOA (timing shadow excluded)"
        if shadow_timing else "Predictive fit of Frequency + TOA + AOA"
    )
    axes[3].legend(loc="lower right", fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(time_s[0], time_s[-1] if time_s[-1] > time_s[0] else time_s[0] + 1.0)

    finite_pri = timing_pri_us[np.isfinite(timing_pri_us)]
    timing_note = (
        "TOA shadow feedback weight: 0; no PRI asserted"
        if shadow_timing else
        (f"Final timing estimate: {finite_pri[-1]:.3f} µs"
         if len(finite_pri) else "Timing estimate unavailable")
    )
    figure.suptitle(
        f"{title}\n{len(records):,} truth-isolated pulses; PW and amplitude excluded; "
        f"{timing_note}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/sequential_pdw/hmm_model1/e46_performance.png"),
    )
    parser.add_argument("--rolling-window", type=int, default=200)
    args = parser.parse_args()
    records = load_diagnostics(args.diagnostics)
    plot_performance(
        records, args.output, rolling_window=args.rolling_window,
    )
    print(args.output)


if __name__ == "__main__":
    main()
