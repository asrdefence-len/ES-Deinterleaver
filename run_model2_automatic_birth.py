"""Evaluate truth-blind Model 2 birth on a controlled E46/E38 mixture."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from sequential_pdw_tracker.io import iter_pdws
from sequential_pdw_tracker.model2_birth import (
    Model2AutomaticBirthTracker,
    Model2BirthConfig,
    Model2BirthDecision,
)
from sequential_pdw_tracker.provenance import run_provenance, sha256_file


REJECTED = "rejected"


def evaluate(
    records: list[tuple[str, Model2BirthDecision]],
    snapshots: list[dict],
    truth_emitters: list[str],
) -> dict:
    composition = {
        snapshot["track_id"]: Counter(
            truth for truth, decision in records
            if decision.track_id == snapshot["track_id"]
        )
        for snapshot in snapshots
    }
    mapping = {
        track_id: counts.most_common(1)[0][0]
        for track_id, counts in composition.items() if counts
    }
    categories = [*truth_emitters, REJECTED]
    confusion = {
        truth: Counter({prediction: 0 for prediction in categories})
        for truth in truth_emitters
    }
    for truth, decision in records:
        prediction = (
            REJECTED if decision.track_id is None
            else mapping.get(decision.track_id, REJECTED)
        )
        confusion[truth][prediction] += 1

    matrix = {
        truth: {prediction: int(confusion[truth][prediction])
                for prediction in categories}
        for truth in truth_emitters
    }
    per_emitter = {}
    for emitter in truth_emitters:
        true_positive = matrix[emitter][emitter]
        false_negative = sum(
            matrix[emitter][prediction]
            for prediction in categories if prediction != emitter
        )
        false_positive = sum(
            matrix[truth][emitter]
            for truth in truth_emitters if truth != emitter
        )
        per_emitter[emitter] = {
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive else None
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative else None
            ),
        }

    track_rows = []
    for snapshot in snapshots:
        track_id = snapshot["track_id"]
        counts = composition[track_id]
        total = sum(counts.values())
        track_rows.append({
            "track_id": track_id,
            "mapped_evaluation_emitter": mapping.get(track_id),
            "truth_counts_evaluation_only": dict(counts.most_common()),
            "evaluation_purity": (
                counts.most_common(1)[0][1] / total if total else None
            ),
            "status": snapshot["status"],
            "first_toa_s": snapshot["first_toa_s"],
            "last_toa_s": snapshot["last_toa_s"],
            "pulse_count": snapshot["pulse_count"],
            "threshold": snapshot["threshold"],
            "timing_fit": snapshot["timing_fit"],
            "rf_state_count": len(snapshot["model"]["rf_states"]),
            "pulse_width_state_count": len(
                snapshot["model"]["pulse_width_states"]
            ),
            "aoa": snapshot["model"]["aoa"],
        })
    fragments = {
        emitter: sum(
            row["mapped_evaluation_emitter"] == emitter for row in track_rows
        )
        for emitter in truth_emitters
    }
    return {
        "categories": categories,
        "matrix_truth_rows_prediction_columns": matrix,
        "per_emitter": per_emitter,
        "cross_assignments": int(sum(
            matrix[truth][prediction]
            for truth in truth_emitters
            for prediction in truth_emitters
            if truth != prediction
        )),
        "rejected_target_pulses": int(sum(
            matrix[truth][REJECTED] for truth in truth_emitters
        )),
        "outcomes": dict(Counter(
            decision.outcome for _truth, decision in records
        )),
        "tracks": track_rows,
        "fragments_per_emitter": fragments,
        "confirmed_tracks": sum(
            snapshot["status"] == "confirmed" for snapshot in snapshots
        ),
        "total_tracks": len(snapshots),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--truth-emitters", nargs=2, default=("46", "38"),
        metavar=("FIRST", "SECOND"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/sequential_pdw/hmm_model2_birth/e46_e38.json"),
    )
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()

    truth_emitters = [str(value) for value in args.truth_emitters]
    if len(set(truth_emitters)) != 2:
        raise ValueError("two distinct truth emitters are required")
    off_config = Model2BirthConfig(enable_timing_after_confirmation=False)
    on_config = Model2BirthConfig(enable_timing_after_confirmation=True)
    timing_off = Model2AutomaticBirthTracker(off_config)
    timing_on = Model2AutomaticBirthTracker(on_config)
    records_off: list[tuple[str, Model2BirthDecision]] = []
    records_on: list[tuple[str, Model2BirthDecision]] = []
    source_pdws = 0
    selected_pdws = 0
    selected_truth_counts: Counter[str] = Counter()
    diagnostic_handle = None
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        for pdw, label in iter_pdws(args.dataset):
            source_pdws += 1
            truth = str(label)
            if truth not in truth_emitters:
                continue
            selected_pdws += 1
            selected_truth_counts[truth] += 1
            off = timing_off.process(pdw)
            on = timing_on.process(pdw)
            records_off.append((truth, off))
            records_on.append((truth, on))
            if diagnostic_handle:
                diagnostic_handle.write(json.dumps({
                    "pdw_id": pdw.pdw_id,
                    "toa_s": pdw.toa_s,
                    "evaluation_truth_label": label,
                    "timing_off": asdict(off),
                    "timing_on": asdict(on),
                }, allow_nan=False) + "\n")
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    snapshots_off = timing_off.snapshots()
    snapshots_on = timing_on.snapshots()
    result_off = evaluate(records_off, snapshots_off, truth_emitters)
    result_on = evaluate(records_on, snapshots_on, truth_emitters)
    map_off = {
        row["track_id"]: row["mapped_evaluation_emitter"]
        for row in result_off["tracks"]
    }
    map_on = {
        row["track_id"]: row["mapped_evaluation_emitter"]
        for row in result_on["tracks"]
    }
    comparison = Counter()
    for (truth, off), (_same_truth, on) in zip(records_off, records_on):
        off_prediction = (
            REJECTED if off.track_id is None
            else map_off.get(off.track_id, REJECTED)
        )
        on_prediction = (
            REJECTED if on.track_id is None
            else map_on.get(on.track_id, REJECTED)
        )
        if off_prediction != on_prediction:
            comparison["changed_assignments"] += 1
        if off_prediction != truth and on_prediction == truth:
            comparison["target_rescued"] += 1
        if off_prediction == truth and on_prediction != truth:
            comparison["target_lost"] += 1

    result = {
        "status": (
            "truth-blind automatic Model 2 birth on a truth-selected controlled "
            "two-emitter mixture; not a full-scene birth benchmark"
        ),
        "provenance": run_provenance(args.dataset),
        "input_boundary": {
            "source_pdws": source_pdws,
            "selected_pdws": selected_pdws,
            "selected_truth_counts_evaluation_only": dict(
                selected_truth_counts.most_common()
            ),
            "selection": (
                "E46 and E38 are interleaved in original TOA order outside the "
                "tracker; labels are removed before each process call"
            ),
            "truth_available_to_tracker": False,
            "background_emitters_present": False,
        },
        "timing_off": result_off,
        "timing_on": result_on,
        "timing_comparison": {
            key: int(comparison[key])
            for key in ("changed_assignments", "target_rescued", "target_lost")
        },
        "checks": {
            "same_input_pulse_count": len(records_off) == len(records_on),
            "truth_not_in_tracker_api": True,
            "timing_off_exactly_two_confirmed_tracks": (
                result_off["total_tracks"] == 2
                and result_off["confirmed_tracks"] == 2
            ),
            "timing_on_exactly_two_confirmed_tracks": (
                result_on["total_tracks"] == 2
                and result_on["confirmed_tracks"] == 2
            ),
        },
        "config": {
            "timing_off_birth": asdict(off_config),
            "timing_on_birth": asdict(on_config),
        },
        "final_snapshots": {
            "timing_off": snapshots_off,
            "timing_on": snapshots_on,
        },
    }
    result["provenance"]["processing_path"] = (
        "truth-selected interleaved E46/E38 fixture -> truth-free staged HMM "
        "birth -> mutually exclusive Model 2 association -> truth-only evaluation"
    )
    source_root = Path(__file__).resolve().parent
    result["provenance"]["active_components"] = {
        "state_and_evidence": {
            "function": "HMMEmitterModel2.process / predictive_evidence",
            "source": "sequential_pdw_tracker/hmm_emitter_model2.py",
        },
        "automatic_birth": {
            "function": "Model2AutomaticBirthTracker.process",
            "source": "sequential_pdw_tracker/model2_birth.py",
        },
        "timing_support": {
            "function": "positive_phase_timing_support",
            "source": "sequential_pdw_tracker/model2_association.py",
        },
        "input": {
            "function": "iter_pdws",
            "source": "sequential_pdw_tracker/io.py",
        },
        "runner": {
            "function": "main",
            "source": "run_model2_automatic_birth.py",
        },
    }
    for component in result["provenance"]["active_components"].values():
        component["source_sha256"] = sha256_file(source_root / component["source"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Controlled unlabelled mixture: {selected_pdws} PDWs "
        f"({selected_truth_counts[truth_emitters[0]]}/"
        f"{selected_truth_counts[truth_emitters[1]]})"
    )
    for name, summary in (("Timing off", result_off), ("Timing on", result_on)):
        print(
            f"{name}: {summary['confirmed_tracks']} confirmed tracks; "
            f"cross-assignments {summary['cross_assignments']}; "
            f"rejected {summary['rejected_target_pulses']}"
        )
        for emitter in truth_emitters:
            metrics = summary["per_emitter"][emitter]
            print(
                f"  E{emitter} precision/recall: "
                f"{metrics['precision']:.4f}/{metrics['recall']:.4f}"
            )
    print(
        "Timing changes: "
        f"{result['timing_comparison']['changed_assignments']}; "
        f"rescued {result['timing_comparison']['target_rescued']}; "
        f"lost {result['timing_comparison']['target_lost']}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
