"""Run a sequential cross-RF association-score experiment.

The first window supplies a measurement-only seed. Later windows are selected
and scored without truth labels; truth is consulted only after each decision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from testbed.association_confidence import AssociationConfidenceTracker
from testbed.cross_rf_train_linker import CrossRFTrainLinker
from testbed.pulse_association_pipeline import FixedPRIPulseAssociation, PulseMeasurement
from testbed.track_aided_pdw_association import (
    TrackAidedPDWAssociator, TrackPrediction,
)


def circular_mean_deg(values):
    angles = np.deg2rad(values)
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * angles)))))


def angular_difference(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def group_stats(data, names, labels):
    rows = {}
    for blob in np.unique(labels[labels >= 0]):
        mask = labels == blob
        rows[int(blob)] = {
            "frequency_hz": float(np.median(data[mask, names.index("Frequency")]) * 1e6),
            "aoa_deg": circular_mean_deg(data[mask, names.index("AoA")]),
            "pulse_width_s": float(np.median(data[mask, names.index("PulseWidth")]) * 1e-6),
        }
    return rows


def select_groups(stats, *, modes, aoa, pw, rf_gate_hz, aoa_gate_deg, pw_gate_s):
    return [blob for blob, row in stats.items()
            if min(abs(row["frequency_hz"] - mode) for mode in modes) <= rf_gate_hz
            and angular_difference(row["aoa_deg"], aoa) <= aoa_gate_deg
            and abs(row["pulse_width_s"] - pw) <= pw_gate_s]


def make_pdws(data, names, indices, offset):
    return [PulseMeasurement(
        offset + int(index) + 1,
        float(data[index, names.index("ToA")] * 1e-6),
        float(data[index, names.index("Frequency")] * 1e6),
        float(data[index, names.index("PulseWidth")] * 1e-6),
        float(data[index, names.index("AoA")])) for index in indices]


def mode_indices(pdws, modes):
    if not pdws:
        return np.empty(0, int)
    rf = np.asarray([pdw.frequency_hz for pdw in pdws])
    return np.argmin(np.abs(rf[:, None] - np.asarray(modes)[None, :]), axis=1)


def transition_model(pdws, modes):
    count = np.ones((len(modes), len(modes)), float)
    ordered = sorted(pdws, key=lambda pdw: (pdw.toa_s, pdw.pdw_id))
    sequence = mode_indices(ordered, modes)
    for left, right in zip(sequence, sequence[1:]):
        count[left, right] += 1.
    return count / count.sum(axis=1, keepdims=True)


def transition_score(pdws, modes, model, previous_mode):
    ordered = sorted(pdws, key=lambda pdw: (pdw.toa_s, pdw.pdw_id))
    sequence = mode_indices(ordered, modes)
    if not len(sequence):
        return None, previous_mode
    augmented = sequence if previous_mode is None else np.r_[previous_mode, sequence]
    probabilities = [model[left, right] for left, right in zip(augmented, augmented[1:])]
    if probabilities:
        geometric = math.exp(float(np.mean(np.log(np.maximum(probabilities, 1e-12)))))
        score = min(1., geometric * len(modes))
    else:
        score = 1.
    return score, int(sequence[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--seed-window", type=int, default=0)
    parser.add_argument("--seed-blobs", type=int, nargs="+", default=list(range(8, 16)))
    parser.add_argument("--tolerance-us", type=float, default=5.)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output", default="results/emitter46_confidence")
    args = parser.parse_args()
    if args.window_size < 8 or args.seed_window < 0 or args.tolerance_us <= 0:
        parser.error("Invalid window configuration")

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        total = len(handle["data"])
        seed_start = args.seed_window * args.window_size
        seed_data = np.asarray(handle["data"][seed_start:seed_start + args.window_size], float)
        populations, count = batch_aoa_populations(seed_data[:, names.index("AoA")])
        seed_labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
        seed_indices = np.flatnonzero(np.isin(seed_labels, args.seed_blobs))
        if not len(seed_indices):
            parser.error("Seed blobs contain no PDWs")
        seed_pdws = make_pdws(seed_data, names, seed_indices, seed_start)
        seed_association = FixedPRIPulseAssociation(
            tolerance_s=args.tolerance_us * 1e-6, min_pulses=20)
        seed_trains, _, _ = seed_association.process(
            seed_pdws, seed_labels[seed_indices], f"W{args.seed_window}")
        families = CrossRFTrainLinker().link(seed_trains)
        if not families:
            parser.error("No seed pulse-train family was discovered")
        seed_family = max(families, key=lambda family: len(family.train_ids))
        seed_stats = group_stats(seed_data, names, seed_labels)
        modes = sorted(seed_stats[blob]["frequency_hz"] for blob in args.seed_blobs
                       if blob in seed_stats)
        if len(modes) != len(args.seed_blobs):
            parser.error("One or more seed blobs were not found")
        seed_aoa = circular_mean_deg([pdw.aoa_deg for pdw in seed_pdws])
        seed_pw = float(np.median([pdw.pulse_width_s for pdw in seed_pdws]))
        model = transition_model(seed_pdws, modes)
        confidence = AssociationConfidenceTracker(
            expected_rf_hz=modes, expected_aoa_deg=seed_aoa,
            expected_pw_s=seed_pw, expected_pri_s=seed_family.base_pri_s,
            timing_gate_s=args.tolerance_us * 1e-6)
        association = FixedPRIPulseAssociation(
            tolerance_s=args.tolerance_us * 1e-6, min_pulses=6)
        raw_prediction = TrackPrediction(
            emitter_id="E46_CANDIDATE", generated_at_s=max(p.toa_s for p in seed_pdws),
            last_toa_s=max(p.toa_s for p in seed_pdws),
            frequency_modes_hz=tuple(modes), aoa_deg=seed_aoa,
            pulse_width_s=seed_pw, pri_s=seed_family.base_pri_s,
            phase_epoch_s=min(p.toa_s for p in seed_pdws),
            pulse_width_sigma_s=max(.08e-6, float(np.std(
                [p.pulse_width_s for p in seed_pdws]))),
            timing_horizon_s=.3, reacquisition_horizon_s=10.)
        raw_gate = TrackAidedPDWAssociator(
            score_threshold=.8, min_confirm_pulses=5, confirmation_span_s=.010)

        n_windows = math.ceil(total / args.window_size)
        if args.max_windows is not None:
            n_windows = min(n_windows, args.max_windows)
        evidence_rows, evaluation_rows = [], []
        previous_mode = None
        origin_s = float(seed_data[0, names.index("ToA")] * 1e-6)
        total_selected = total_hits = total_truth = total_associated = 0
        for window in range(n_windows):
            start = window * args.window_size
            data = np.asarray(handle["data"][start:start + args.window_size], float)
            if not len(data):
                break
            truth = (np.asarray(handle["labels"][start:start + len(data)]).reshape(-1)
                     if "labels" in handle else None)
            populations, count = batch_aoa_populations(data[:, names.index("AoA")])
            labels = group_window(data, names, populations, count, 8, .20, 6., 8)
            stats = group_stats(data, names, labels)
            chosen = (list(args.seed_blobs) if window == args.seed_window else
                select_groups(stats, modes=modes, aoa=seed_aoa, pw=seed_pw,
                    rf_gate_hz=40e6, aoa_gate_deg=3., pw_gate_s=.12e-6))
            selected_indices = np.flatnonzero(np.isin(labels, chosen))
            pdws = make_pdws(data, names, selected_indices, start)
            track_aided_ids = ()
            # Conservative first integration: invoke the raw-PDW maintenance
            # path only when discovery/group selection has no observation.
            if not pdws and window != args.seed_window:
                all_pdws = make_pdws(data, names, np.arange(len(data)), start)
                decisions = [raw_gate.process(pdw, [raw_prediction]) for pdw in all_pdws]
                track_aided_ids = tuple(pdw_id for decision in decisions
                                        for pdw_id in decision.confirmed_pdw_ids)
                if track_aided_ids:
                    selected_indices = np.asarray(
                        [pdw_id - start - 1 for pdw_id in track_aided_ids], int)
                    pdws = make_pdws(data, names, selected_indices, start)
            trains, _, assignments = association.process(
                pdws, labels[selected_indices], f"W{window}")
            rf_transition, previous_mode = transition_score(
                pdws, modes, model, previous_mode)
            time_s = float(data[0, names.index("ToA")] * 1e-6)
            end_s = float(data[-1, names.index("ToA")] * 1e-6)
            evidence = confidence.update(window_id=f"W{window}",
                time_s=time_s, window_end_s=end_s, pdws=pdws, trains=trains,
                candidate_groups=len(chosen), rf_transition_score=rf_transition)
            row = evidence.to_dict()
            row["time_from_start_s"] = time_s - origin_s
            row["window_end_from_start_s"] = end_s - origin_s
            row["selected_blob_ids"] = chosen
            row["train_count"] = len(trains)
            row["track_aided_pulses"] = len(track_aided_ids)
            evidence_rows.append(row)

            evaluation = {"window_id": f"W{window}"}
            if truth is not None:
                selected_truth = truth[selected_indices]
                hits = int(np.sum(selected_truth == 46))
                true_count = int(np.sum(truth == 46))
                associated_ids = {pdw_id for pdw_id, train_id in assignments.items()
                                  if train_id is not None}
                associated_local = [pdw_id - start - 1 for pdw_id in associated_ids]
                associated_hits = int(np.sum(truth[associated_local] == 46)) if associated_local else 0
                evaluation.update(selected_pulses=len(pdws), selected_truth_46=hits,
                    truth_46_in_window=true_count, associated_pulses=len(associated_ids),
                    associated_truth_46=associated_hits)
                total_selected += len(pdws)
                total_hits += hits
                total_truth += true_count
                total_associated += len(associated_ids)
            evaluation_rows.append(evaluation)

    result = {
        "score_type": "UNCALIBRATED_ASSOCIATION_SCORE",
        "score_components": {"timing": .40, "rf_behavior": .25,
                             "signal_similarity": .20, "continuity": .15},
        "truth_is_evaluation_only": True,
        "seed": {"window": args.seed_window, "blob_ids": args.seed_blobs,
                 "frequency_modes_hz": modes, "aoa_deg": seed_aoa,
                 "pulse_width_s": seed_pw, "base_pri_s": seed_family.base_pri_s},
        "windows": evidence_rows,
        "evaluation_windows": evaluation_rows,
        "evaluation_summary": {
            "selected_pulses": total_selected,
            "selected_truth_46": total_hits,
            "truth_46_total": total_truth,
            "associated_pulses": total_associated,
            "selection_purity": total_hits / total_selected if total_selected else None,
            "selection_coverage": total_hits / total_truth if total_truth else None,
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "association_confidence.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["evaluation_summary"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
