"""Fit and evaluate a zero-feedback phase-coherent timing shadow."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sequential_pdw_tracker.phase_timing_shadow import (
    PhaseTimingShadowConfig,
    analyze_phase_timing_shadow,
)
from sequential_pdw_tracker.provenance import sha256_file


def analysis_provenance(diagnostics: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ("git", *args), text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = {}
    for name in ("numpy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    source_result = diagnostics.with_suffix(".json")
    source_result_record = None
    if source_result.is_file():
        source = json.loads(source_result.read_text())
        source_result_record = {
            "path": str(source_result),
            "size_bytes": source_result.stat().st_size,
            "sha256": sha256_file(source_result),
            "embedded_run_provenance": source.get("provenance"),
        }
    return {
        "processing_path": (
            "existing Model 1 diagnostics -> causal visibility selection -> "
            "training-only interval-lattice fit -> held-out phase evaluation"
        ),
        "diagnostics": {
            "path": str(diagnostics),
            "size_bytes": diagnostics.stat().st_size,
            "sha256": sha256_file(diagnostics),
        },
        "source_model_result": source_result_record,
        "analysis_git": {
            "revision": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "packages": packages,
    }


def plot_phase_timing_shadow(result: dict, diagnostics: dict, output: Path) -> None:
    periods_us = diagnostics["candidate_periods_s"] * 1e6
    scores = diagnostics["candidate_scores"]
    finite = np.isfinite(scores)
    residual_us = diagnostics["held_out_residual_s"] * 1e6
    relative_time_s = (
        diagnostics["held_out_toa_s"] - diagnostics["held_out_toa_s"][0]
        if len(diagnostics["held_out_toa_s"]) else np.empty(0)
    )
    log_ratio = diagnostics["held_out_phase_log_likelihood_ratio"]
    period_us = result["fit"]["dominant_interval_s"] * 1e6
    sigma_us = result["fit"]["phase_sigma_s"] * 1e6

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(periods_us[finite], scores[finite] - np.max(scores[finite]))
    axes[0].axvline(period_us, color="tab:red", linestyle="--",
                    label=f"Selected {period_us:.3f} µs")
    axes[0].set(
        xlabel="Candidate dominant interval (µs)",
        ylabel="Training objective relative to best",
        title="Training-only interval-lattice selection",
    )
    axes[0].legend()

    axes[1].scatter(relative_time_s, residual_us, s=3, alpha=0.35)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].axhspan(-sigma_us, sigma_us, color="tab:green", alpha=0.14,
                    label=f"Training ±1σ ({sigma_us:.3f} µs)")
    axes[1].set(
        xlabel="Time through held-out portion (s)",
        ylabel="Nearest-lattice phase residual (µs)",
        title="Held-out phase residuals; model is not refitted",
    )
    axes[1].legend()

    axes[2].hist(log_ratio, bins=80, alpha=0.75, label="Observed held-out gaps")
    axes[2].hist(
        diagnostics["negative_control_log_likelihood_ratio"],
        bins=80,
        alpha=0.55,
        label="Half-cycle negative control",
    )
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set(
        xlabel="Conditional phase log-likelihood ratio",
        ylabel="Count",
        title="Predictive phase evidence versus uniform phase",
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)

    held_out = result["held_out"]
    phase = held_out["phase_log_likelihood_ratio"]
    figure.suptitle(
        "Phase-coherent timing shadow — held-out evaluation\n"
        f"feedback weight 0; {held_out['scored_gaps']:,} scored gaps; "
        f"mean phase evidence {phase['mean']:.3f}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.diagnostics.open() if line.strip()]
    toa_s = np.asarray([record["toa_s"] for record in records], dtype=float)
    active_posterior = np.asarray([
        record["shadow_active_posterior"] for record in records
    ], dtype=float)
    config = PhaseTimingShadowConfig(train_fraction=args.train_fraction)
    result, diagnostics = analyze_phase_timing_shadow(
        toa_s, active_posterior, config,
    )
    result["provenance"] = analysis_provenance(args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_phase_timing_shadow(result, diagnostics, args.plot)

    held_out = result["held_out"]
    phase = held_out["phase_log_likelihood_ratio"]
    control = held_out["half_cycle_negative_control_log_likelihood_ratio"]
    print(f"Dominant training interval: {result['fit']['dominant_interval_s'] * 1e6:.3f} us")
    print(f"Training phase sigma: {result['fit']['phase_sigma_s'] * 1e6:.3f} us")
    print(f"Held-out scored gaps: {held_out['scored_gaps']}")
    print(f"Held-out mean phase log-likelihood ratio: {phase['mean']:.3f}")
    print(f"Held-out positive phase evidence: {phase['positive_fraction']:.3f}")
    print(f"Half-cycle control mean log-likelihood ratio: {control['mean']:.3f}")
    print("Timing feedback weight: 0.000 (shadow only)")
    print(args.output)
    print(args.plot)


if __name__ == "__main__":
    main()
