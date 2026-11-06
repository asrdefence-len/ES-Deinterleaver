"""Run frozen Model 2 automatic birth on the complete Config 101 stream."""
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
)
from sequential_pdw_tracker.provenance import run_provenance, sha256_file


TARGETS = ("46", "38")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/sequential_pdw/hmm_model2_birth/full_scene.json"),
    )
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument(
        "--max-tracks", type=int, default=8,
        help="Frozen simultaneous track capacity (default: 8)",
    )
    parser.add_argument(
        "--end-activity-window-s", type=float, default=0.5,
        help=(
            "Evaluation-only recency window used to report tracks updated "
            "near the end of the recording"
        ),
    )
    args = parser.parse_args()
    if args.progress_every < 1:
        raise ValueError("progress interval must be positive")
    if args.max_tracks < 1:
        raise ValueError("maximum tracks must be positive")
    if args.end_activity_window_s <= 0.0:
        raise ValueError("end activity window must be positive")

    birth_config = Model2BirthConfig(
        max_tracks=args.max_tracks,
        enable_timing_after_confirmation=True,
        enable_safe_confirmed_pruning=True,
    )
    tracker = Model2AutomaticBirthTracker(birth_config)
    truth_counts: Counter[str] = Counter()
    track_truth_counts: dict[str, Counter[str]] = {}
    target_track_counts = {target: Counter() for target in TARGETS}
    outcomes: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    lifecycle_events = []
    score_upper_bound_pruned = 0
    assigned_pdws = 0
    diagnostic_records = 0
    capacity_first_reached = None
    processed = 0
    final_toa_s = None
    diagnostic_handle = None
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        for pdw, label in iter_pdws(args.dataset):
            processed += 1
            final_toa_s = pdw.toa_s
            truth = str(label)
            truth_counts[truth] += 1
            decision = tracker.process(pdw)
            outcomes[decision.outcome] += 1
            score_upper_bound_pruned += len(
                decision.confirmed_score_upper_bound_pruned
            )
            if decision.rejection_reason is not None:
                rejection_reasons[decision.rejection_reason] += 1
            if decision.track_id is not None:
                assigned_pdws += 1
                track_truth_counts.setdefault(
                    decision.track_id, Counter(),
                )[truth] += 1
                if truth in target_track_counts:
                    target_track_counts[truth][decision.track_id] += 1
            if decision.outcome in ("birth", "confirmed"):
                lifecycle_events.append({
                    "pdw_id": pdw.pdw_id,
                    "toa_s": pdw.toa_s,
                    "event": decision.outcome,
                    "track_id": decision.track_id,
                    "evaluation_truth_label": label,
                })
            if (capacity_first_reached is None
                    and tracker.track_count >= birth_config.max_tracks):
                capacity_first_reached = {
                    "pdw_id": pdw.pdw_id,
                    "toa_s": pdw.toa_s,
                    "processed_pdws": processed,
                }

            log_diagnostic = (
                truth in TARGETS
                or decision.outcome in ("birth", "confirmed")
            )
            if diagnostic_handle and log_diagnostic:
                diagnostic_handle.write(json.dumps({
                    "pdw_id": pdw.pdw_id,
                    "toa_s": pdw.toa_s,
                    "evaluation_truth_label": label,
                    "decision": asdict(decision),
                }, allow_nan=False) + "\n")
                diagnostic_records += 1
            if processed % args.progress_every == 0:
                print(
                    f"Processed {processed:,} PDWs; "
                    f"tracks {tracker.track_count}/{birth_config.max_tracks}; "
                    f"assigned {assigned_pdws:,}"
                )
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    snapshots = tracker.snapshots()
    if final_toa_s is None:
        raise ValueError("dataset contains no PDWs")
    mapping = {}
    track_rows = []
    for snapshot in snapshots:
        track_id = snapshot["track_id"]
        counts = track_truth_counts.get(track_id, Counter())
        total = sum(counts.values())
        dominant = counts.most_common(1)[0] if counts else (None, 0)
        mapping[track_id] = dominant[0]
        track_rows.append({
            "track_id": track_id,
            "status": snapshot["status"],
            "pulse_count": snapshot["pulse_count"],
            "assigned_count_evaluation": total,
            "dominant_truth_emitter_evaluation_only": dominant[0],
            "dominant_truth_count_evaluation_only": dominant[1],
            "evaluation_purity": dominant[1] / total if total else None,
            "truth_counts_evaluation_only": dict(counts.most_common()),
            "first_toa_s": snapshot["first_toa_s"],
            "last_toa_s": snapshot["last_toa_s"],
            "age_at_recording_end_s": final_toa_s - snapshot["last_toa_s"],
            "updated_within_end_activity_window": (
                final_toa_s - snapshot["last_toa_s"]
                <= args.end_activity_window_s
            ),
            "threshold": snapshot["threshold"],
            "timing_fit": snapshot["timing_fit"],
            "rf_state_count": len(snapshot["model"]["rf_states"]),
            "pulse_width_state_count": len(
                snapshot["model"]["pulse_width_states"]
            ),
            "aoa": snapshot["model"]["aoa"],
        })

    target_evaluation = {}
    for target in TARGETS:
        counts = target_track_counts[target]
        total = truth_counts[target]
        assigned = sum(counts.values())
        primary = counts.most_common(1)[0] if counts else (None, 0)
        assigned_to_other_target = sum(
            count for track_id, count in counts.items()
            if mapping.get(track_id) in TARGETS
            and mapping.get(track_id) != target
        )
        target_evaluation[target] = {
            "truth_pulses": total,
            "assigned_to_any_track": assigned,
            "assigned_coverage": assigned / total if total else None,
            "rejected": total - assigned,
            "track_ids_touched": len(counts),
            "track_assignment_counts": dict(counts.most_common()),
            "primary_track_id": primary[0],
            "primary_track_retained": primary[1],
            "primary_track_retention": primary[1] / total if total else None,
            "primary_track_dominant_truth": mapping.get(primary[0]),
            "assigned_to_other_target_dominant_track": assigned_to_other_target,
            "tracks_dominated_by_target": sum(
                dominant == target for dominant in mapping.values()
            ),
        }

    dominant_truth_total = sum(
        row["dominant_truth_count_evaluation_only"] for row in track_rows
    )
    identity_evaluation = {
        "weighted_track_purity": (
            dominant_truth_total / assigned_pdws if assigned_pdws else None
        ),
        "distinct_dominant_truth_emitters": len({
            row["dominant_truth_emitter_evaluation_only"]
            for row in track_rows
            if row["dominant_truth_emitter_evaluation_only"] is not None
        }),
        "tracks_with_multiple_truth_emitters": sum(
            len(row["truth_counts_evaluation_only"]) > 1
            for row in track_rows
        ),
        "tracks_below_90_percent_purity": sum(
            row["evaluation_purity"] is not None
            and row["evaluation_purity"] < 0.9
            for row in track_rows
        ),
        "maximum_truth_emitters_in_one_track": max(
            (len(row["truth_counts_evaluation_only"])
             for row in track_rows),
            default=0,
        ),
    }

    result = {
        "status": (
            "frozen Model 2 automatic-birth full-scene stress replay; no truth "
            "input selection and no lifecycle/reacquisition"
        ),
        "provenance": run_provenance(args.dataset),
        "input_boundary": {
            "source_pdws": processed,
            "pdws_passed_to_tracker": processed,
            "truth_available_to_tracker": False,
            "truth_counts_evaluation_only": dict(truth_counts.most_common()),
            "diagnostic_records": diagnostic_records,
            "diagnostic_logging": (
                "all E46/E38 pulses plus automatic birth/confirmation events; "
                "aggregates cover every PDW"
            ),
        },
        "tracker": {
            "total_tracks": len(snapshots),
            "confirmed_tracks": sum(
                row["status"] == "confirmed" for row in snapshots
            ),
            "tentative_tracks": sum(
                row["status"] == "tentative" for row in snapshots
            ),
            "capacity_first_reached": capacity_first_reached,
            "assigned_pdws": assigned_pdws,
            "assignment_coverage": assigned_pdws / processed,
            "outcomes": dict(outcomes),
            "rejection_reasons": dict(rejection_reasons),
            "safe_upper_bound_pruned_track_scores": score_upper_bound_pruned,
            "end_activity_window_s_evaluation_only": (
                args.end_activity_window_s
            ),
            "tracks_updated_within_end_activity_window": sum(
                row["updated_within_end_activity_window"]
                for row in track_rows
            ),
            "tracks": track_rows,
            "lifecycle_events": lifecycle_events,
        },
        "target_evaluation": target_evaluation,
        "identity_evaluation": identity_evaluation,
        "controlled_reference": {
            "description": (
                "same birth/HMM/timing settings on the prior background-free "
                "E46/E38 mixture"
            ),
            "E46_recall": 0.9903,
            "E38_recall": 0.9726,
            "tracks": 2,
            "fragments_per_target": 1,
        },
        "config": {
            "birth": asdict(birth_config),
            "end_activity_window_s_evaluation_only": (
                args.end_activity_window_s
            ),
        },
        "final_snapshots": snapshots,
    }
    result["provenance"]["processing_path"] = (
        "all chronological Config 101 PDWs -> truth-blind staged HMM birth -> "
        "frozen Model 2 association with bounded timing -> truth-only evaluation"
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
        "input": {
            "function": "iter_pdws",
            "source": "sequential_pdw_tracker/io.py",
        },
        "runner": {
            "function": "main",
            "source": "run_model2_full_scene_birth.py",
        },
    }
    for component in result["provenance"]["active_components"].values():
        component["source_sha256"] = sha256_file(source_root / component["source"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Full scene: {processed:,} PDWs; {len(snapshots)} tracks "
        f"({result['tracker']['confirmed_tracks']} confirmed); "
        f"assigned {assigned_pdws:,} ({assigned_pdws / processed:.2%})"
    )
    if capacity_first_reached:
        print(
            "Track capacity first reached at PDW "
            f"{capacity_first_reached['pdw_id']:,} "
            f"(processed pulse {capacity_first_reached['processed_pdws']:,})"
        )
    for target in TARGETS:
        row = target_evaluation[target]
        print(
            f"E{target}: assigned {row['assigned_to_any_track']:,}/"
            f"{row['truth_pulses']:,} ({row['assigned_coverage']:.2%}); "
            f"tracks touched {row['track_ids_touched']}; "
            f"primary {row['primary_track_id']} retained "
            f"{row['primary_track_retained']:,}"
        )
    print(
        "Evaluation weighted track purity: "
        f"{identity_evaluation['weighted_track_purity']:.2%}; "
        "distinct dominant emitters "
        f"{identity_evaluation['distinct_dominant_truth_emitters']}; "
        "mixed tracks "
        f"{identity_evaluation['tracks_with_multiple_truth_emitters']}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
