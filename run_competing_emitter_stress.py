"""Stress track-aided association with an adversarial similar emitter."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from run_cross_rf_confidence import circular_mean_deg, group_stats, make_pdws
from testbed.pulse_association_pipeline import PulseMeasurement
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
    parser.add_argument("--output", default="results/emitter46_confidence/competing_emitter_stress.json")
    args = parser.parse_args()

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        seed_start = args.seed_window * args.window_size
        seed_data = np.asarray(handle["data"][seed_start:seed_start + args.window_size], float)
        populations, count = batch_aoa_populations(seed_data[:, names.index("AoA")])
        seed_labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
        seed_indices = np.flatnonzero(np.isin(seed_labels, args.seed_blobs))
        seed_pdws = make_pdws(seed_data, names, seed_indices, seed_start)
        stats = group_stats(seed_data, names, seed_labels)
        modes = tuple(sorted(stats[blob]["frequency_hz"] for blob in args.seed_blobs))
        aoa = circular_mean_deg([pdw.aoa_deg for pdw in seed_pdws])
        pw_values = np.asarray([pdw.pulse_width_s for pdw in seed_pdws])
        pw = float(np.median(pw_values))
        pw_sigma = max(.08e-6, float(np.std(pw_values)))
        last = max(pdw.toa_s for pdw in seed_pdws)
        first = min(pdw.toa_s for pdw in seed_pdws)
        emitter46 = TrackPrediction("E46_TRACK", last, last, modes, aoa, pw,
            pri_s=85.089e-6, phase_epoch_s=first,
            pulse_width_sigma_s=pw_sigma)
        competitor = TrackPrediction("COMPETITOR_TRACK", last, last, modes,
            aoa + 2.4, pw + .05e-6, pri_s=87e-6, phase_epoch_s=first,
            pulse_width_sigma_s=pw_sigma)

        target_start = args.target_window * args.window_size
        data = np.asarray(handle["data"][target_start:target_start + args.window_size], float)
        raw = make_pdws(data, names, np.arange(len(data)), target_start)
        target_time = min(pdw.toa_s for pdw in raw) + .0001
        next_id = len(handle["data"]) + 1000
        synthetic, source = [], {}
        for index in range(6):
            pdw = PulseMeasurement(next_id + index,
                target_time + .0005 + index * .00085,
                modes[index % len(modes)] + 5e6,
                pw + .05e-6, aoa + 2.4)
            synthetic.append(pdw)
            source[pdw.pdw_id] = "synthetic_competitor"
        next_id += 6
        for index in range(3):
            pdw = PulseMeasurement(next_id + index,
                target_time + .0007 + index * .0011,
                modes[(index + 2) % len(modes)] + 2e6,
                pw + .025e-6, aoa + 1.2)
            synthetic.append(pdw)
            source[pdw.pdw_id] = "deliberately_ambiguous"

        gate = TrackAidedPDWAssociator(
            score_threshold=.8, ambiguity_margin=.08,
            min_confirm_pulses=5, confirmation_span_s=.010)
        decisions = []
        for pdw in sorted(raw + synthetic, key=lambda item: (item.toa_s, item.pdw_id)):
            decision = gate.process(pdw, [emitter46, competitor])
            decisions.append(decision)

        # Truth is read only after every association decision is complete.
        truth = np.asarray(handle["labels"][
            target_start:target_start + len(data)]).reshape(-1)

    decision_by_id = {decision.pdw_id: decision for decision in decisions}
    pdw_by_id = {pdw.pdw_id: pdw for pdw in raw + synthetic}
    for pdw in raw:
        label = truth[pdw.pdw_id - target_start - 1]
        source[pdw.pdw_id] = ("real_emitter46" if label == args.evaluation_truth_id
                              else "real_background")
    counts = Counter()
    detailed = []
    for pdw in raw + synthetic:
        decision = decision_by_id[pdw.pdw_id]
        outcome = (decision.selected_emitter_id if decision.selected_emitter_id
                   else decision.status)
        counts[(source[pdw.pdw_id], outcome)] += 1
        if decision.status != "unclaimed":
            detailed.append({"source": source[pdw.pdw_id],
                "toa_s": pdw_by_id[pdw.pdw_id].toa_s,
                "frequency_hz": pdw_by_id[pdw.pdw_id].frequency_hz,
                "aoa_deg": pdw_by_id[pdw.pdw_id].aoa_deg,
                "pulse_width_s": pdw_by_id[pdw.pdw_id].pulse_width_s,
                **decision.to_dict()})
    result = {
        "truth_is_evaluation_only": True,
        "stress_design": {
            "shared_rf_modes": True,
            "competitor_aoa_offset_deg": 2.4,
            "competitor_pulse_width_offset_s": .05e-6,
            "ambiguous_pulses_at_midpoint": 3,
        },
        "summary": {f"{source_name} -> {outcome}": count
                    for (source_name, outcome), count in sorted(counts.items())},
        "non_unclaimed_decisions": detailed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
