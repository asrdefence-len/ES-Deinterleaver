"""Compare Model 2 association with timing support off versus on."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from sequential_pdw_tracker.hmm_emitter_model2 import (
    HMMEmitterModel2,
    HMMModel2Config,
)
from sequential_pdw_tracker.io import iter_pdws, iter_truth_isolated_pdws
from sequential_pdw_tracker.model2_association import (
    Model2AssociationConfig,
    Model2SingleTrackAssociationArm,
)
from sequential_pdw_tracker.phase_timing_shadow import (
    PhaseTimingShadowConfig,
    fit_phase_timing_shadow,
)
from sequential_pdw_tracker.provenance import run_provenance, sha256_file


def confusion(decisions: list, is_target: list[bool]) -> dict:
    accepted = np.asarray([decision.accepted for decision in decisions], dtype=bool)
    target = np.asarray(is_target, dtype=bool)
    true_positive = int(np.sum(accepted & target))
    false_positive = int(np.sum(accepted & ~target))
    false_negative = int(np.sum(~accepted & target))
    true_negative = int(np.sum(~accepted & ~target))
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else None
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else None
        ),
        "false_positive_rate": (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--truth-emitter", type=int, default=46)
    parser.add_argument("--target-pulses", type=int, default=12530)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sequential_pdw/model2_timing_association/result.json"),
    )
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()

    association_config = Model2AssociationConfig()
    model_config = HMMModel2Config()
    target_rows = list(iter_truth_isolated_pdws(
        args.dataset,
        args.truth_emitter,
        limit=args.target_pulses,
    ))
    if len(target_rows) != args.target_pulses:
        raise ValueError(
            f"requested {args.target_pulses} target pulses; found {len(target_rows)}"
        )
    state_end = int(math.floor(
        len(target_rows) * association_config.state_training_fraction
    ))
    association_start = int(math.floor(
        len(target_rows) * association_config.association_start_fraction
    ))

    model = HMMEmitterModel2(model_config)
    training_updates = []
    for pdw, _label in target_rows[:state_end]:
        training_updates.append(model.process(pdw))

    calibration_scores = []
    for pdw, _label in target_rows[state_end:association_start]:
        calibration_scores.append(model.predictive_evidence(pdw).total_log_evidence)
        training_updates.append(model.process(pdw))
    calibration_scores_array = np.asarray(calibration_scores, dtype=float)
    threshold = float(np.quantile(
        calibration_scores_array,
        1.0 - association_config.calibration_target_recall,
    ))

    training_toa = np.asarray([
        pdw.toa_s for pdw, _label in target_rows[:association_start]
    ])
    training_active = np.asarray([
        update.shadow_active_posterior for update in training_updates
    ])
    gaps_s = np.diff(training_toa)
    active_gap = (
        training_active[1:]
        >= PhaseTimingShadowConfig().active_posterior_threshold
    )
    phase_fit, _periods, _scores = fit_phase_timing_shadow(
        gaps_s[(gaps_s > 0.0) & active_gap],
        PhaseTimingShadowConfig(),
    )

    baseline_arm = Model2SingleTrackAssociationArm(
        copy.deepcopy(model),
        threshold=threshold,
        last_accepted_toa_s=training_toa[-1],
        config=association_config,
    )
    timing_arm = Model2SingleTrackAssociationArm(
        copy.deepcopy(model),
        threshold=threshold,
        last_accepted_toa_s=training_toa[-1],
        timing_fit=phase_fit,
        config=association_config,
    )

    initial_snapshot = model.snapshot()
    rf_centres = np.asarray([
        state["mean_hz"] for state in initial_snapshot["rf_states"]
    ])
    rf_gate_hz = model_config.model1.rf_new_state_gate_hz
    start_toa_s = training_toa[-1]
    end_toa_s = target_rows[-1][0].toa_s
    target = str(args.truth_emitter)

    baseline_decisions = []
    timing_decisions = []
    truth_is_target: list[bool] = []
    candidate_truth_counts: Counter[str] = Counter()
    diagnostic_handle = None
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        for pdw, label in iter_pdws(args.dataset):
            if pdw.toa_s <= start_toa_s:
                continue
            if pdw.toa_s > end_toa_s:
                break
            if float(np.min(np.abs(rf_centres - pdw.frequency_hz))) > rf_gate_hz:
                continue

            baseline = baseline_arm.process(pdw)
            timing = timing_arm.process(pdw)
            is_target = str(label) == target
            baseline_decisions.append(baseline)
            timing_decisions.append(timing)
            truth_is_target.append(is_target)
            candidate_truth_counts[str(label)] += 1
            if diagnostic_handle:
                diagnostic_handle.write(json.dumps({
                    "pdw_id": pdw.pdw_id,
                    "toa_s": pdw.toa_s,
                    "evaluation_truth_label": label,
                    "evaluation_is_target": is_target,
                    "baseline": asdict(baseline),
                    "timing_on": asdict(timing),
                }, allow_nan=False) + "\n")
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    baseline_metrics = confusion(baseline_decisions, truth_is_target)
    timing_metrics = confusion(timing_decisions, truth_is_target)
    target_array = np.asarray(truth_is_target, dtype=bool)
    baseline_accept = np.asarray([
        decision.accepted for decision in baseline_decisions
    ])
    timing_accept = np.asarray([
        decision.accepted for decision in timing_decisions
    ])
    boosts = np.asarray([
        decision.timing_applied_boost for decision in timing_decisions
    ])
    scoreable = np.asarray([
        decision.timing_scoreable for decision in timing_decisions
    ])
    changed = baseline_accept != timing_accept

    result = {
        "status": (
            "controlled one-track Model 2 timing ablation; not a multi-emitter "
            "deinterleaving benchmark"
        ),
        "provenance": run_provenance(args.dataset),
        "input_boundary": {
            "target_truth_used_for_seed_and_evaluation_only": True,
            "truth_available_to_association_arms": False,
            "state_training_target_pulses": state_end,
            "calibration_target_pulses": association_start - state_end,
            "held_out_target_pulses": len(target_rows) - association_start,
            "held_out_start_toa_s": start_toa_s,
            "held_out_end_toa_s": end_toa_s,
            "candidate_selection": (
                "all held-out PDWs within the frozen Model 2 RF discovery gate "
                "of any learned E46 RF state; no AOA, PW, timing or truth gate"
            ),
            "rf_candidate_gate_hz": rf_gate_hz,
            "candidate_pdws": len(baseline_decisions),
            "candidate_truth_counts_evaluation_only": dict(
                candidate_truth_counts.most_common()
            ),
        },
        "calibration": {
            "target_recall_setting": association_config.calibration_target_recall,
            "acceptance_threshold": threshold,
            "score_count": len(calibration_scores),
            "score_minimum": float(calibration_scores_array.min()),
            "score_median": float(np.median(calibration_scores_array)),
            "score_maximum": float(calibration_scores_array.max()),
        },
        "timing_model": {
            "dominant_interval_s": phase_fit.period_s,
            "phase_sigma_s": phase_fit.sigma_s,
            "training_score": phase_fit.training_score,
            "training_active_gaps": phase_fit.training_active_gaps,
            "training_scored_gaps": phase_fit.training_scored_gaps,
            "support_policy": (
                "positive conditional phase LLR only; negative/unscoreable is "
                "neutral; applied support is capped"
            ),
            "weight": association_config.timing_weight,
            "maximum_boost": association_config.timing_max_boost,
            "maximum_intervals": association_config.timing_max_intervals,
        },
        "baseline": baseline_metrics,
        "timing_on": timing_metrics,
        "comparison": {
            "decision_changes": int(changed.sum()),
            "target_rescued": int(np.sum(~baseline_accept & timing_accept & target_array)),
            "target_lost_after_divergence": int(np.sum(baseline_accept & ~timing_accept & target_array)),
            "distractor_promoted": int(np.sum(~baseline_accept & timing_accept & ~target_array)),
            "distractor_removed_after_divergence": int(np.sum(baseline_accept & ~timing_accept & ~target_array)),
            "timing_scoreable_candidates": int(scoreable.sum()),
            "positive_boost_candidates": int(np.sum(boosts > 0.0)),
            "maximum_applied_boost": float(boosts.max(initial=0.0)),
            "mean_applied_boost_all_candidates": float(np.mean(boosts)),
            "mean_applied_boost_target": float(np.mean(boosts[target_array])),
            "mean_applied_boost_distractor": float(np.mean(boosts[~target_array])),
        },
        "checks": {
            "same_candidate_count": len(baseline_decisions) == len(timing_decisions),
            "timing_support_never_negative": bool(np.all(boosts >= 0.0)),
            "timing_support_within_cap": bool(
                np.all(boosts <= association_config.timing_max_boost + 1e-12)
            ),
            "held_out_target_candidate_coverage": (
                int(target_array.sum()) / (len(target_rows) - association_start)
            ),
        },
        "initial_model2_state": initial_snapshot,
        "final_baseline_state": baseline_arm.model.snapshot(),
        "final_timing_on_state": timing_arm.model.snapshot(),
        "config": {
            "association": asdict(association_config),
            "model2": asdict(model_config),
        },
    }
    result["provenance"]["processing_path"] = (
        "truth-isolated E46 Model 2 training -> frozen RF-only candidate envelope "
        "-> parallel one-track association arms with timing off/on -> truth-only "
        "evaluation"
    )
    source_root = Path(__file__).resolve().parent
    result["provenance"]["active_components"] = {
        "state_and_evidence": {
            "function": "HMMEmitterModel2.predictive_evidence",
            "source": "sequential_pdw_tracker/hmm_emitter_model2.py",
        },
        "association": {
            "function": "Model2SingleTrackAssociationArm.process",
            "source": "sequential_pdw_tracker/model2_association.py",
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
            "source": "run_model2_timing_association.py",
        },
    }
    for component in result["provenance"]["active_components"].values():
        component["source_sha256"] = sha256_file(source_root / component["source"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(f"Training target pulses: {association_start}")
    print(f"Held-out RF-plausible candidates: {len(baseline_decisions)}")
    print(
        "Held-out target/distractor candidates: "
        f"{int(target_array.sum())}/{int((~target_array).sum())}"
    )
    print(f"Calibrated Model 2 threshold: {threshold:.3f}")
    print(
        "Baseline precision/recall: "
        f"{baseline_metrics['precision']:.4f}/{baseline_metrics['recall']:.4f}"
    )
    print(
        "Timing-on precision/recall: "
        f"{timing_metrics['precision']:.4f}/{timing_metrics['recall']:.4f}"
    )
    print(
        "Timing decision changes: "
        f"{result['comparison']['decision_changes']} "
        f"(target rescued {result['comparison']['target_rescued']}, "
        f"distractor promoted {result['comparison']['distractor_promoted']})"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
