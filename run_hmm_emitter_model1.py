"""Run proper HMM Model 1 (Frequency + TOA + AOA) on one controlled emitter."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from sequential_pdw_tracker.hmm_emitter import HMMEmitterModel1, HMMModel1Config
from sequential_pdw_tracker.io import inspect_dataset, iter_truth_isolated_pdws
from sequential_pdw_tracker.provenance import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--truth-emitter", type=int, required=True)
    parser.add_argument("--limit", type=int, default=12530)
    parser.add_argument("--output", type=Path,
                        default=Path("results/sequential_pdw/hmm_model1/result.json"))
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()

    info = inspect_dataset(args.dataset)
    config = HMMModel1Config()
    tracker = HMMEmitterModel1(config)
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
                diagnostic_handle.write(json.dumps(asdict(update), allow_nan=False) + "\n")
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    snapshot = tracker.snapshot()
    posterior_errors = [abs(sum(update.rf_posterior) - 1.0) for update in updates]
    transition = np.asarray(snapshot["rf_transition_matrix"], dtype=float)
    log_evidence = np.asarray([update.total_log_evidence for update in updates], dtype=float)
    provenance = run_provenance(args.dataset)
    provenance["processing_path"] = (
        "truth-isolated raw PDWs -> RF HMM forward filter + circular AOA Markov "
        "filter + semi-Markov TOA/PRI hypotheses"
    )
    result = {
        "status": "proper HMM Model 1 component test; not a deinterleaving benchmark",
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
            "max_rf_posterior_sum_error": max(posterior_errors, default=0.0),
            "rf_transition_row_sums": transition.sum(axis=1).tolist() if transition.size else [],
            "all_log_evidence_finite": bool(np.isfinite(log_evidence).all()),
            "pulse_width_excluded": "PulseWidth" in snapshot["excluded_features"],
            "amplitude_excluded": "Amplitude" in snapshot["excluded_features"],
        },
        "log_evidence": {
            "minimum": float(log_evidence.min()) if len(log_evidence) else None,
            "median": float(np.median(log_evidence)) if len(log_evidence) else None,
            "maximum": float(log_evidence.max()) if len(log_evidence) else None,
        },
        "final_state": snapshot,
    }
    result["dataset_schema"]["path"] = str(result["dataset_schema"]["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Processed {len(updates)} PDWs; every pulse updated one HMM filter")
    print(f"RF states learned: {len(snapshot['rf_states'])}")
    print(f"AOA: {snapshot['aoa']['mean_deg']:.3f} deg +/- {snapshot['aoa']['sigma_deg']:.3f} deg")
    print(f"Timing hypotheses retained: {len(snapshot['timing_hypotheses'])}")
    print(f"PW excluded: {result['checks']['pulse_width_excluded']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
