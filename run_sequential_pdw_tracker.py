"""Run the independent sequential probabilistic PDW tracker baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sequential_pdw_tracker.evaluation import evaluate
from sequential_pdw_tracker.io import (
    inspect_dataset,
    iter_pdws,
    iter_truth_isolated_pdws,
)
from sequential_pdw_tracker.provenance import run_provenance
from sequential_pdw_tracker.tracker import SequentialPDWTracker, TrackerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--limit", type=int, default=5000,
        help="PDWs to process; counts selected PDWs in truth-isolation mode (default: 5000)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="zero-based source row, or selected-PDW offset in truth-isolation mode",
    )
    parser.add_argument("--output", type=Path, default=Path("results/sequential_pdw/baseline.json"))
    parser.add_argument("--diagnostics", type=Path, help="optional per-PDW JSONL posterior log")
    parser.add_argument("--max-tracks", type=int, default=64)
    parser.add_argument(
        "--isolate-truth-emitter",
        type=int,
        help=("controlled component test: select only this truth emitter before "
              "inference; truth is not passed to the tracker"),
    )
    args = parser.parse_args()

    info = inspect_dataset(args.dataset)
    config = TrackerConfig(max_tracks=args.max_tracks)
    tracker = SequentialPDWTracker(config)
    decisions = []
    truth = []
    diagnostic_handle = None
    try:
        if args.diagnostics:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_handle = args.diagnostics.open("w")
        source = (
            iter_pdws(args.dataset, limit=args.limit, start=args.start)
            if args.isolate_truth_emitter is None else
            iter_truth_isolated_pdws(
                args.dataset,
                args.isolate_truth_emitter,
                limit=args.limit,
                start=args.start,
            )
        )
        for pdw, label in source:
            decision = tracker.process(pdw)
            decisions.append(decision)
            truth.append(label)
            if diagnostic_handle:
                diagnostic_handle.write(json.dumps(asdict(decision), allow_nan=False) + "\n")
    finally:
        if diagnostic_handle:
            diagnostic_handle.close()

    result = {
        "status": (
            "experimental filtering baseline; not performance accepted"
            if args.isolate_truth_emitter is None else
            "controlled truth-isolated state-estimation test; not a deinterleaving benchmark"
        ),
        "input_selection": {
            "method": ("source_rows" if args.isolate_truth_emitter is None
                       else "evaluation_truth_isolation"),
            "truth_emitter": args.isolate_truth_emitter,
            "start": args.start,
            "limit": args.limit,
            "truth_available_to_tracker": False,
        },
        "dataset_schema": asdict(info),
        "settings": asdict(config),
        "provenance": run_provenance(args.dataset),
        "evaluation": evaluate(decisions, truth),
        "tracks": [asdict(snapshot) for snapshot in tracker.snapshots()],
    }
    result["dataset_schema"]["path"] = str(result["dataset_schema"]["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    evaluation = result["evaluation"]
    if args.isolate_truth_emitter is not None:
        print(f"Controlled input: truth-isolated emitter {args.isolate_truth_emitter}; "
              "truth was not supplied to the tracker")
    print(f"Processed {evaluation['processed_pdws']} PDWs; created {tracker.track_count} tracks")
    print(f"Outcomes: {evaluation['outcomes']}")
    print(f"Labelled-association purity: {evaluation['labelled_association_purity']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
