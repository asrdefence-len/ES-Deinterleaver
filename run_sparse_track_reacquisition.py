"""Evaluate prediction-assisted raw-PDW reacquisition in a sparse window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from run_cross_rf_confidence import circular_mean_deg, group_stats, make_pdws
from testbed.track_aided_pdw_association import (
    TrackAidedPDWAssociator, TrackPrediction,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--seed-window", type=int, default=0)
    parser.add_argument("--seed-blobs", type=int, nargs="+", default=list(range(8, 16)))
    parser.add_argument("--target-window", type=int, default=84)
    parser.add_argument("--evaluation-truth-id", type=int, default=46)
    parser.add_argument("--output", default="results/emitter46_confidence/w84_track_aided.json")
    args = parser.parse_args()

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        seed_start = args.seed_window * args.window_size
        seed_data = np.asarray(handle["data"][seed_start:seed_start + args.window_size], float)
        populations, count = batch_aoa_populations(seed_data[:, names.index("AoA")])
        seed_labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
        seed_indices = np.flatnonzero(np.isin(seed_labels, args.seed_blobs))
        if not len(seed_indices):
            parser.error("Seed blobs contain no PDWs")
        seed_pdws = make_pdws(seed_data, names, seed_indices, seed_start)
        stats = group_stats(seed_data, names, seed_labels)
        modes = tuple(sorted(stats[blob]["frequency_hz"] for blob in args.seed_blobs))
        pw = np.asarray([pdw.pulse_width_s for pdw in seed_pdws])
        aoa = np.asarray([pdw.aoa_deg for pdw in seed_pdws])
        prediction = TrackPrediction(
            emitter_id="E46_CANDIDATE", generated_at_s=max(p.toa_s for p in seed_pdws),
            last_toa_s=max(p.toa_s for p in seed_pdws), frequency_modes_hz=modes,
            aoa_deg=circular_mean_deg(aoa), pulse_width_s=float(np.median(pw)),
            pri_s=85.089e-6, phase_epoch_s=min(p.toa_s for p in seed_pdws),
            aoa_sigma_deg=max(1.5, float(np.std(aoa))),
            pulse_width_sigma_s=max(.08e-6, float(np.std(pw))),
            timing_horizon_s=.3, reacquisition_horizon_s=10.)

        target_start = args.target_window * args.window_size
        target_data = np.asarray(
            handle["data"][target_start:target_start + args.window_size], float)
        target_pdws = make_pdws(target_data, names, np.arange(len(target_data)), target_start)
        gate = TrackAidedPDWAssociator(score_threshold=.8, min_confirm_pulses=5,
                                       confirmation_span_s=.010)
        decisions = [gate.process(pdw, [prediction]) for pdw in target_pdws]

        # Truth is read only after every association decision is complete.
        truth = (np.asarray(handle["labels"][
            target_start:target_start + len(target_data)]).reshape(-1)
            if "labels" in handle else None)

    claimed = [decision for decision in decisions
               if decision.status in {"provisional", "confirmed"}]
    confirmed = tuple(pdw_id for decision in decisions
                      for pdw_id in decision.confirmed_pdw_ids)
    result = {
        "seed_window": args.seed_window,
        "seed_blobs": args.seed_blobs,
        "target_window": args.target_window,
        "truth_is_evaluation_only": True,
        "prediction": {
            "frequency_modes_hz": modes, "aoa_deg": prediction.aoa_deg,
            "pulse_width_s": prediction.pulse_width_s,
            "pulse_width_sigma_s": prediction.pulse_width_sigma_s,
            "timing_horizon_s": prediction.timing_horizon_s,
        },
        "target_pdws": len(target_pdws),
        "gate_claimed_pdws": len(claimed),
        "confirmed_pdw_ids": confirmed,
        "decision_rows": [decision.to_dict() for decision in claimed],
    }
    if truth is not None:
        claimed_local = [decision.pdw_id - target_start - 1 for decision in claimed]
        confirmed_local = [pdw_id - target_start - 1 for pdw_id in confirmed]
        truth_count = int(np.sum(truth == args.evaluation_truth_id))
        claimed_hits = int(np.sum(truth[claimed_local] == args.evaluation_truth_id))
        confirmed_hits = int(np.sum(truth[confirmed_local] == args.evaluation_truth_id))
        result["evaluation"] = {
            "truth_id": args.evaluation_truth_id,
            "truth_pulses_in_window": truth_count,
            "claimed_truth_pulses": claimed_hits,
            "confirmed_truth_pulses": confirmed_hits,
            "claimed_purity": claimed_hits / len(claimed) if claimed else None,
            "claimed_coverage": claimed_hits / truth_count if truth_count else None,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"decision_rows", "prediction"}}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
