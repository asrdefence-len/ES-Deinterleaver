"""Run controlled E46/E38 Model 2 competition with timing off versus on."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from sequential_pdw_tracker.hmm_emitter_model2 import (
    HMMEmitterModel2,
    HMMModel2Config,
)
from sequential_pdw_tracker.io import iter_pdws, iter_truth_isolated_pdws
from sequential_pdw_tracker.model2_association import Model2AssociationConfig
from sequential_pdw_tracker.model2_competition import (
    Model2CompetitionDecision,
    Model2CompetitiveAssociation,
)
from sequential_pdw_tracker.phase_timing_shadow import (
    PhaseTimingFit,
    PhaseTimingShadowConfig,
    fit_phase_timing_shadow,
)
from sequential_pdw_tracker.provenance import run_provenance, sha256_file


NEITHER = "neither"


@dataclass
class TrainedSeed:
    model: HMMEmitterModel2
    total_target_pulses: int
    state_training_pulses: int
    calibration_pulses: int
    training_end_toa_s: float
    target_end_toa_s: float
    threshold: float
    calibration_scores: np.ndarray
    timing_fit: PhaseTimingFit


def train_seed(
    dataset: Path,
    truth_emitter: str,
    association_config: Model2AssociationConfig,
    model_config: HMMModel2Config,
) -> TrainedSeed:
    rows = list(iter_truth_isolated_pdws(dataset, truth_emitter))
    state_end = int(math.floor(
        len(rows) * association_config.state_training_fraction
    ))
    association_start = int(math.floor(
        len(rows) * association_config.association_start_fraction
    ))
    if state_end < 1 or association_start <= state_end:
        raise ValueError(f"insufficient training pulses for emitter {truth_emitter}")

    model = HMMEmitterModel2(model_config)
    updates = []
    for pdw, _label in rows[:state_end]:
        updates.append(model.process(pdw))

    calibration_scores = []
    for pdw, _label in rows[state_end:association_start]:
        calibration_scores.append(model.predictive_evidence(pdw).total_log_evidence)
        updates.append(model.process(pdw))
    calibration_array = np.asarray(calibration_scores, dtype=float)
    threshold = float(np.quantile(
        calibration_array,
        1.0 - association_config.calibration_target_recall,
    ))

    training_toa = np.asarray([
        pdw.toa_s for pdw, _label in rows[:association_start]
    ])
    active = np.asarray([
        update.shadow_active_posterior for update in updates
    ])
    gaps_s = np.diff(training_toa)
    active_gap = (
        active[1:] >= PhaseTimingShadowConfig().active_posterior_threshold
    )
    timing_fit, _periods, _scores = fit_phase_timing_shadow(
        gaps_s[(gaps_s > 0.0) & active_gap],
        PhaseTimingShadowConfig(),
    )
    return TrainedSeed(
        model=model,
        total_target_pulses=len(rows),
        state_training_pulses=state_end,
        calibration_pulses=association_start - state_end,
        training_end_toa_s=training_toa[-1],
        target_end_toa_s=rows[-1][0].toa_s,
        threshold=threshold,
        calibration_scores=calibration_array,
        timing_fit=timing_fit,
    )


def empty_confusion(categories: list[str]) -> dict[str, Counter[str]]:
    return {truth: Counter({prediction: 0 for prediction in categories})
            for truth in categories}


def add_decision(
    confusion: dict[str, Counter[str]],
    truth: str,
    selected_track: str | None,
) -> None:
    confusion[truth][NEITHER if selected_track is None else selected_track] += 1


def summarize_confusion(
    confusion: dict[str, Counter[str]],
    track_ids: list[str],
) -> dict:
    categories = [*track_ids, NEITHER]
    matrix = {
        truth: {prediction: int(confusion[truth][prediction])
                for prediction in categories}
        for truth in categories
    }
    per_track = {}
    for track_id in track_ids:
        true_positive = matrix[track_id][track_id]
        false_negative = sum(
            matrix[track_id][prediction]
            for prediction in categories if prediction != track_id
        )
        false_positive = sum(
            matrix[truth][track_id]
            for truth in categories if truth != track_id
        )
        per_track[track_id] = {
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
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[category][category] for category in categories)
    return {
        "categories": categories,
        "matrix_truth_rows_prediction_columns": matrix,
        "per_track": per_track,
        "cross_assignments": int(sum(
            matrix[truth][prediction]
            for truth in track_ids
            for prediction in track_ids
            if truth != prediction
        )),
        "background_false_accepts": int(sum(
            matrix[NEITHER][track_id] for track_id in track_ids
        )),
        "overall_accuracy": correct / total if total else None,
        "total_pdws": total,
    }


def compact_decision(decision: Model2CompetitionDecision) -> dict:
    return {
        "selected_track": decision.selected_track,
        "neither_weight": decision.neither_weight,
        "tracks": {
            track_id: {
                "model2_margin": score.model2_margin,
                "timing_boost": score.timing_applied_boost,
                "combined_margin": score.combined_margin,
                "weight": score.normalized_hypothesis_weight,
                "rf_log_evidence": score.rf_log_evidence,
                "aoa_log_evidence": score.aoa_log_evidence,
                "pulse_width_log_evidence": score.pulse_width_log_evidence,
            }
            for track_id, score in decision.track_scores.items()
        },
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
        default=Path(
            "results/sequential_pdw/hmm_model2_two_emitter/e46_e38.json"
        ),
    )
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()

    track_ids = [str(value) for value in args.truth_emitters]
    if len(set(track_ids)) != 2:
        raise ValueError("two distinct truth emitters are required")
    association_config = Model2AssociationConfig()
    model_config = HMMModel2Config()
    seeds = {
        track_id: train_seed(
            args.dataset, track_id, association_config, model_config,
        )
        for track_id in track_ids
    }
    start_toa_s = max(seed.training_end_toa_s for seed in seeds.values())
    end_toa_s = min(seed.target_end_toa_s for seed in seeds.values())
    if end_toa_s <= start_toa_s:
        raise ValueError("trained emitters have no overlapping held-out interval")

    thresholds = {key: seed.threshold for key, seed in seeds.items()}
    accepted_toa = {
        key: seed.training_end_toa_s for key, seed in seeds.items()
    }
    timing_fits = {key: seed.timing_fit for key, seed in seeds.items()}
    timing_off = Model2CompetitiveAssociation(
        {key: copy.deepcopy(seed.model) for key, seed in seeds.items()},
        thresholds=thresholds,
        last_accepted_toa_s=accepted_toa,
        config=association_config,
    )
    timing_on = Model2CompetitiveAssociation(
        {key: copy.deepcopy(seed.model) for key, seed in seeds.items()},
        thresholds=thresholds,
        last_accepted_toa_s=accepted_toa,
        timing_fits=timing_fits,
        config=association_config,
    )

    rf_centres = {
        key: np.asarray([
            state["mean_hz"] for state in seed.model.snapshot()["rf_states"]
        ])
        for key, seed in seeds.items()
    }
    rf_gate_hz = model_config.model1.rf_new_state_gate_hz
    categories = [*track_ids, NEITHER]
    candidate_confusion_off = empty_confusion(categories)
    candidate_confusion_on = empty_confusion(categories)
    full_confusion_off = empty_confusion(categories)
    full_confusion_on = empty_confusion(categories)
    candidate_truth_counts: Counter[str] = Counter()
    window_truth_counts: Counter[str] = Counter()
    pruned_truth_counts: Counter[str] = Counter()
    comparison = Counter()
    diagnostic_handle = None
    diagnostic_records = 0
    candidate_count = 0
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        for pdw, label in iter_pdws(args.dataset):
            if pdw.toa_s <= start_toa_s:
                continue
            if pdw.toa_s > end_toa_s:
                break
            label_text = str(label)
            truth = label_text if label_text in seeds else NEITHER
            window_truth_counts[label_text] += 1
            plausible = any(
                float(np.min(np.abs(centres - pdw.frequency_hz))) <= rf_gate_hz
                for centres in rf_centres.values()
            )
            if not plausible:
                pruned_truth_counts[label_text] += 1
                add_decision(full_confusion_off, truth, None)
                add_decision(full_confusion_on, truth, None)
                continue

            candidate_count += 1
            candidate_truth_counts[label_text] += 1
            off = timing_off.process(pdw)
            on = timing_on.process(pdw)
            add_decision(candidate_confusion_off, truth, off.selected_track)
            add_decision(candidate_confusion_on, truth, on.selected_track)
            add_decision(full_confusion_off, truth, off.selected_track)
            add_decision(full_confusion_on, truth, on.selected_track)

            off_prediction = off.selected_track or NEITHER
            on_prediction = on.selected_track or NEITHER
            if off_prediction != on_prediction:
                comparison["changed_assignments"] += 1
            if truth in seeds:
                if off_prediction != truth and on_prediction == truth:
                    comparison["target_rescued"] += 1
                    comparison[f"target_rescued_{truth}"] += 1
                if off_prediction == truth and on_prediction != truth:
                    comparison["target_lost"] += 1
                    comparison[f"target_lost_{truth}"] += 1
                if off_prediction in seeds and off_prediction != truth:
                    comparison["cross_assignment_off"] += 1
                if on_prediction in seeds and on_prediction != truth:
                    comparison["cross_assignment_on"] += 1
            else:
                if off_prediction == NEITHER and on_prediction != NEITHER:
                    comparison["background_promoted"] += 1
                if off_prediction != NEITHER and on_prediction == NEITHER:
                    comparison["background_removed"] += 1

            log_diagnostic = (
                truth in seeds
                or off.selected_track is not None
                or on.selected_track is not None
                or off_prediction != on_prediction
            )
            if diagnostic_handle and log_diagnostic:
                diagnostic_handle.write(json.dumps({
                    "pdw_id": pdw.pdw_id,
                    "toa_s": pdw.toa_s,
                    "evaluation_truth_label": label,
                    "evaluation_category": truth,
                    "timing_off": compact_decision(off),
                    "timing_on": compact_decision(on),
                }, allow_nan=False) + "\n")
                diagnostic_records += 1
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    result = {
        "status": (
            "controlled two-emitter Model 2 competition; seeded filters and "
            "RF pruning are not automatic track birth"
        ),
        "provenance": run_provenance(args.dataset),
        "input_boundary": {
            "truth_used_for_seed_and_evaluation_only": True,
            "truth_available_to_competition": False,
            "track_ids": track_ids,
            "held_out_start_toa_s": start_toa_s,
            "held_out_end_toa_s": end_toa_s,
            "held_out_duration_s": end_toa_s - start_toa_s,
            "window_pdws": int(sum(window_truth_counts.values())),
            "candidate_pdws": candidate_count,
            "diagnostic_records": diagnostic_records,
            "diagnostic_logging": (
                "all E46/E38 evaluation targets plus every accepted background "
                "or timing-off/on disagreement; aggregation still covers every "
                "candidate"
            ),
            "rf_pruned_pdws_assigned_neither": int(sum(pruned_truth_counts.values())),
            "candidate_selection": (
                "union of frozen learned RF-state discovery envelopes; no AOA, "
                "PW, timing or truth gate"
            ),
            "rf_candidate_gate_hz": rf_gate_hz,
            "window_truth_counts_evaluation_only": dict(
                window_truth_counts.most_common()
            ),
            "candidate_truth_counts_evaluation_only": dict(
                candidate_truth_counts.most_common()
            ),
            "rf_pruned_truth_counts_evaluation_only": dict(
                pruned_truth_counts.most_common()
            ),
        },
        "seeds": {
            track_id: {
                "total_target_pulses": seed.total_target_pulses,
                "state_training_pulses": seed.state_training_pulses,
                "calibration_pulses": seed.calibration_pulses,
                "training_end_toa_s": seed.training_end_toa_s,
                "target_end_toa_s": seed.target_end_toa_s,
                "acceptance_threshold": seed.threshold,
                "calibration_score_minimum": float(seed.calibration_scores.min()),
                "calibration_score_median": float(np.median(seed.calibration_scores)),
                "calibration_score_maximum": float(seed.calibration_scores.max()),
                "initial_model_state": seed.model.snapshot(),
            }
            for track_id, seed in seeds.items()
        },
        "timing_models": {
            track_id: asdict(seed.timing_fit)
            for track_id, seed in seeds.items()
        },
        "candidate_results": {
            "timing_off": summarize_confusion(
                candidate_confusion_off, track_ids,
            ),
            "timing_on": summarize_confusion(
                candidate_confusion_on, track_ids,
            ),
        },
        "full_window_results": {
            "timing_off": summarize_confusion(full_confusion_off, track_ids),
            "timing_on": summarize_confusion(full_confusion_on, track_ids),
        },
        "timing_comparison": {
            key: int(comparison[key])
            for key in (
                "changed_assignments", "target_rescued", "target_lost",
                *(f"target_rescued_{track_id}" for track_id in track_ids),
                *(f"target_lost_{track_id}" for track_id in track_ids),
                "cross_assignment_off", "cross_assignment_on",
                "background_promoted", "background_removed",
            )
        },
        "checks": {
            "same_candidate_stream": (
                sum(timing_off.selected_counts.values()) + timing_off.neither_count
                == sum(timing_on.selected_counts.values()) + timing_on.neither_count
                == candidate_count
            ),
            "mutually_exclusive_updates": True,
            "target_candidate_coverage": {
                track_id: (
                    candidate_truth_counts[track_id] / window_truth_counts[track_id]
                )
                for track_id in track_ids
            },
        },
        "final_states": {
            "timing_off": {
                key: model.snapshot() for key, model in timing_off.models.items()
            },
            "timing_on": {
                key: model.snapshot() for key, model in timing_on.models.items()
            },
        },
        "config": {
            "association": asdict(association_config),
            "model2": asdict(model_config),
        },
    }
    result["provenance"]["processing_path"] = (
        "truth-isolated E46/E38 seed training -> union RF candidate envelope -> "
        "mutually exclusive Model 2 competition with timing off/on -> "
        "truth-only evaluation"
    )
    source_root = Path(__file__).resolve().parent
    result["provenance"]["active_components"] = {
        "state_and_evidence": {
            "function": "HMMEmitterModel2.predictive_evidence",
            "source": "sequential_pdw_tracker/hmm_emitter_model2.py",
        },
        "competition": {
            "function": "Model2CompetitiveAssociation.process",
            "source": "sequential_pdw_tracker/model2_competition.py",
        },
        "timing_support": {
            "function": "positive_phase_timing_support",
            "source": "sequential_pdw_tracker/model2_association.py",
        },
        "timing_fit": {
            "function": "fit_phase_timing_shadow",
            "source": "sequential_pdw_tracker/phase_timing_shadow.py",
        },
        "input": {
            "function": "iter_pdws / iter_truth_isolated_pdws",
            "source": "sequential_pdw_tracker/io.py",
        },
        "runner": {
            "function": "main",
            "source": "run_model2_two_emitter_competition.py",
        },
    }
    for component in result["provenance"]["active_components"].values():
        component["source_sha256"] = sha256_file(source_root / component["source"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    off = result["full_window_results"]["timing_off"]
    on = result["full_window_results"]["timing_on"]
    print(
        f"Held-out window: {result['input_boundary']['window_pdws']} PDWs; "
        f"{candidate_count} RF-plausible candidates"
    )
    for track_id in track_ids:
        print(
            f"E{track_id} timing-off precision/recall: "
            f"{off['per_track'][track_id]['precision']:.4f}/"
            f"{off['per_track'][track_id]['recall']:.4f}"
        )
        print(
            f"E{track_id} timing-on precision/recall: "
            f"{on['per_track'][track_id]['precision']:.4f}/"
            f"{on['per_track'][track_id]['recall']:.4f}"
        )
    print(
        "Timing-off/on cross-assignments: "
        f"{off['cross_assignments']}/{on['cross_assignments']}"
    )
    print(
        "Timing-off/on background false accepts: "
        f"{off['background_false_accepts']}/{on['background_false_accepts']}"
    )
    print(
        "Timing changes: "
        f"{result['timing_comparison']['changed_assignments']}; "
        f"targets rescued {result['timing_comparison']['target_rescued']}; "
        f"targets lost {result['timing_comparison']['target_lost']}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
