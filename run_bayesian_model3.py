"""Run the bounded Bayesian Model 3 filter on a controlled Config 101 block."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

from sequential_pdw_tracker.io import iter_pdws
from sequential_pdw_tracker.model3_bayesian import (
    BayesianModel3Config,
    BayesianModel3Tracker,
)
from sequential_pdw_tracker.provenance import run_provenance, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--truth-emitters", nargs="+", default=("46", "38"),
        help="evaluation-only emitters selected outside the tracker",
    )
    parser.add_argument(
        "--limit-selected", type=int, default=250,
        help="maximum selected PDWs in the controlled feasibility block",
    )
    parser.add_argument(
        "--max-global-hypotheses", type=int, default=16,
        help=(
            "global posterior branch cap; 16 is the validated development "
            "profile and 128 reproduces the wider reference"
        ),
    )
    parser.add_argument(
        "--progress-every", type=int, default=1000,
        help="print progress after this many selected PDWs; zero disables it",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/sequential_pdw/hmm_model3/controlled_block.json"),
    )
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if args.limit_selected < 1:
        raise ValueError("limit-selected must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")

    targets = {str(value) for value in args.truth_emitters}
    config = BayesianModel3Config(
        max_global_hypotheses=args.max_global_hypotheses,
    )
    tracker = BayesianModel3Tracker(config)
    selected_counts: Counter[str] = Counter()
    fractional_composition: dict[str, Counter[str]] = defaultdict(Counter)
    expected_source_mass: Counter[str] = Counter()
    last_update = None
    source_pdws = 0
    selected_pdws = 0
    diagnostic_handle = None
    started_at = time.monotonic()
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        for item, label in iter_pdws(args.dataset):
            source_pdws += 1
            truth = str(label)
            if truth not in targets:
                continue
            update = tracker.process(item)
            last_update = update
            selected_pdws += 1
            selected_counts[truth] += 1
            for action, probability in update.association_probabilities.items():
                if action.startswith("track:") or action.startswith("birth:"):
                    track_id = action.split(":", 1)[1]
                    fractional_composition[track_id][truth] += probability
                else:
                    expected_source_mass[action] += probability
            if diagnostic_handle:
                diagnostic_handle.write(json.dumps({
                    "pdw_id": item.pdw_id,
                    "toa_s": item.toa_s,
                    "evaluation_truth_label": label,
                    "posterior": asdict(update),
                }, allow_nan=False) + "\n")
            if (
                args.progress_every
                and selected_pdws % args.progress_every == 0
            ):
                elapsed_wall_s = time.monotonic() - started_at
                print(
                    f"Processed {selected_pdws:,} selected PDWs; "
                    f"{update.retained_global_hypotheses} hypotheses; "
                    f"last retained mass {update.retained_posterior_mass:.9f}; "
                    f"wall {elapsed_wall_s:.1f} s",
                    flush=True,
                )
            if selected_pdws >= args.limit_selected:
                break
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()
    if last_update is None:
        raise ValueError("controlled selection produced no PDWs")
    elapsed_wall_s = time.monotonic() - started_at

    track_rows = []
    for snapshot in tracker.snapshots():
        track_id = snapshot["track_id"]
        counts = fractional_composition[track_id]
        total = sum(counts.values())
        track_rows.append({
            "track_id": track_id,
            "existence_probability": snapshot["existence_probability"],
            "fractional_truth_mass_evaluation_only": dict(counts),
            "fractional_dominant_truth_evaluation_only": (
                counts.most_common(1)[0][0] if counts else None
            ),
            "fractional_purity_evaluation_only": (
                counts.most_common(1)[0][1] / total if total else None
            ),
            "representative_pulse_count": snapshot["representative_pulse_count"],
            "representative_expected_rate_hz": (
                snapshot["representative_expected_rate_hz"]
            ),
        })

    result = {
        "status": (
            "controlled Bayesian association-and-birth feasibility block; "
            "not a full-scene benchmark and not lifecycle acceptance"
        ),
        "provenance": run_provenance(args.dataset),
        "input_boundary": {
            "source_pdws_read": source_pdws,
            "selected_pdws": selected_pdws,
            "selected_truth_counts_evaluation_only": dict(selected_counts),
            "truth_available_to_tracker": False,
            "selection": (
                "requested emitters selected outside inference and interleaved "
                "in original chronological order"
            ),
        },
        "model": tracker.summary(),
        "last_update": asdict(last_update),
        "runtime": {
            "wall_s": elapsed_wall_s,
            "selected_pdws_per_s": selected_pdws / elapsed_wall_s,
        },
        "expected_background_source_mass": dict(expected_source_mass),
        "tracks": track_rows,
        "interpretation_boundary": {
            "map_action_is_diagnostic_only": True,
            "hard_track_assignment_used_for_filter_update": False,
            "confirmation_threshold_present": False,
            "top_k_pruning_is_computational_approximation": True,
            "configured_global_hypothesis_cap": args.max_global_hypotheses,
            "lifecycle_implemented": False,
        },
    }
    result["provenance"]["processing_path"] = (
        "controlled truth selection outside inference -> truth-free marked-Poisson "
        "birth/clutter/existing-track hypotheses -> bounded global posterior"
    )
    root = Path(__file__).resolve().parent
    result["provenance"]["active_components"] = {
        "emitter_mark_model": {
            "function": (
                "HMMEmitterModel2.predictive_evidence / "
                "process_active_features / clone_active"
            ),
            "source": "sequential_pdw_tracker/hmm_emitter_model2.py",
        },
        "multi_target_filter": {
            "function": "BayesianModel3Tracker.process",
            "source": "sequential_pdw_tracker/model3_bayesian.py",
        },
        "runner": {
            "function": "main",
            "source": "run_bayesian_model3.py",
        },
    }
    for component in result["provenance"]["active_components"].values():
        component["source_sha256"] = sha256_file(root / component["source"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Controlled Bayesian block: {selected_pdws} PDWs; "
        f"{last_update.retained_global_hypotheses} retained global hypotheses; "
        f"wall {elapsed_wall_s:.2f} s"
    )
    print(
        "Posterior mass retained at last event: "
        f"{last_update.retained_posterior_mass:.6f}"
    )
    for row in track_rows[:10]:
        print(
            f"{row['track_id']}: existence "
            f"{row['existence_probability']:.4f}; "
            f"fractional evaluation purity "
            f"{row['fractional_purity_evaluation_only']!s}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
