"""Run HMM Model 2 (Frequency + TOA + AOA + PW) on one controlled emitter."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from sequential_pdw_tracker.hmm_emitter_model2 import (
    HMMEmitterModel2,
    HMMModel2Config,
)
from sequential_pdw_tracker.io import inspect_dataset, iter_truth_isolated_pdws
from sequential_pdw_tracker.provenance import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--truth-emitter", type=int, required=True)
    parser.add_argument("--limit", type=int, default=12530)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sequential_pdw/hmm_model2/result.json"),
    )
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()

    info = inspect_dataset(args.dataset)
    config = HMMModel2Config()
    tracker = HMMEmitterModel2(config)
    updates = []
    diagnostic_handle = None
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        for pdw, _label in iter_truth_isolated_pdws(
            args.dataset, args.truth_emitter, limit=args.limit,
        ):
            update = tracker.process(pdw)
            updates.append(update)
            if diagnostic_handle:
                diagnostic_handle.write(
                    json.dumps(asdict(update), allow_nan=False) + "\n"
                )
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    snapshot = tracker.snapshot()
    rf_errors = [abs(sum(update.rf_posterior) - 1.0) for update in updates]
    pw_errors = [
        abs(sum(update.pulse_width_posterior) - 1.0) for update in updates
    ]
    shadow_errors = [
        abs(update.shadow_active_posterior + update.shadow_coast_posterior - 1.0)
        for update in updates
    ]
    rf_transition = np.asarray(snapshot["rf_transition_matrix"], dtype=float)
    pw_transition = np.asarray(
        snapshot["pulse_width_transition_matrix"], dtype=float,
    )
    total_evidence = np.asarray(
        [update.total_log_evidence for update in updates], dtype=float,
    )
    pulse_width_evidence = np.asarray(
        [update.pulse_width_log_evidence for update in updates], dtype=float,
    )
    provenance = run_provenance(args.dataset)
    provenance["processing_path"] = (
        "truth-isolated raw PDWs -> frozen Model 1 RF HMM + circular AOA "
        "filter + zero-feedback TOA shadow -> added log-PW HMM"
    )
    result = {
        "status": (
            "proper HMM Model 2 component test; not a deinterleaving benchmark"
        ),
        "input_selection": {
            "method": "evaluation_truth_isolation",
            "truth_emitter": args.truth_emitter,
            "limit": args.limit,
            "truth_available_to_filter": False,
        },
        "dataset_schema": asdict(info),
        "provenance": provenance,
        "processed_pdws": len(updates),
        "update_coverage": len(updates) / args.limit if args.limit else None,
        "checks": {
            "max_rf_posterior_sum_error": max(rf_errors, default=0.0),
            "max_pulse_width_posterior_sum_error": max(pw_errors, default=0.0),
            "max_shadow_visibility_sum_error": max(shadow_errors, default=0.0),
            "rf_transition_row_sums": (
                rf_transition.sum(axis=1).tolist() if rf_transition.size else []
            ),
            "pulse_width_transition_row_sums": (
                pw_transition.sum(axis=1).tolist() if pw_transition.size else []
            ),
            "all_total_log_evidence_finite": bool(
                np.isfinite(total_evidence).all()
            ),
            "all_pulse_width_log_evidence_finite": bool(
                np.isfinite(pulse_width_evidence).all()
            ),
            "pulse_width_state_birth_count": sum(
                update.pulse_width_state_created for update in updates
            ),
            "timing_feedback_weight_zero": all(
                update.timing_feedback_weight == 0.0 for update in updates
            ),
            "transmitter_pri_not_inferred": (
                snapshot["detected_train_timing_shadow"]
                ["transmitter_pri_estimate"] is None
            ),
            "pulse_width_included": (
                "PulseWidth" in snapshot["active_evidence_features"]
            ),
            "amplitude_excluded": "Amplitude" in snapshot["excluded_features"],
        },
        "log_evidence": {
            "total_minimum": (
                float(total_evidence.min()) if len(total_evidence) else None
            ),
            "total_median": (
                float(np.median(total_evidence)) if len(total_evidence) else None
            ),
            "total_maximum": (
                float(total_evidence.max()) if len(total_evidence) else None
            ),
            "pulse_width_minimum": (
                float(pulse_width_evidence.min())
                if len(pulse_width_evidence) else None
            ),
            "pulse_width_median": (
                float(np.median(pulse_width_evidence))
                if len(pulse_width_evidence) else None
            ),
            "pulse_width_maximum": (
                float(pulse_width_evidence.max())
                if len(pulse_width_evidence) else None
            ),
        },
        "final_state": snapshot,
    }
    result["dataset_schema"]["path"] = str(result["dataset_schema"]["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(f"Processed {len(updates)} PDWs; every pulse updated one Model 2 filter")
    print(f"RF states learned: {len(snapshot['rf_states'])}")
    print(
        f"AOA: {snapshot['aoa']['mean_deg']:.3f} deg +/- "
        f"{snapshot['aoa']['sigma_deg']:.3f} deg"
    )
    print(f"PW states learned: {len(snapshot['pulse_width_states'])}")
    print(
        "PW state births marked in diagnostics: "
        f"{result['checks']['pulse_width_state_birth_count']}"
    )
    print(
        "PW state means: "
        + ", ".join(
            f"{state['mean_s'] * 1e6:.3f} us"
            for state in sorted(
                snapshot["pulse_width_states"],
                key=lambda item: item["mean_s"],
            )
        )
    )
    print("Timing feedback weight: 0.000 (shadow only)")
    print(f"Amplitude excluded: {result['checks']['amplitude_excluded']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
