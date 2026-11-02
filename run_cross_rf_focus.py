"""Focused cross-RF pulse-train family experiment on selected candidate blobs.

Blob selection defines the controlled experiment.  Truth labels are read only
after train discovery and cross-RF linking to score the retained hypotheses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from testbed.cross_rf_train_linker import CrossRFTrainLinker
from testbed.pulse_association_pipeline import FixedPRIPulseAssociation, PulseMeasurement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--blobs", type=int, nargs="+", required=True)
    parser.add_argument("--tolerance-us", type=float, default=5.,
                        help="pulse-to-lattice association tolerance")
    parser.add_argument("--output", default="results/cross_rf_focus")
    args = parser.parse_args()
    if args.window < 0 or args.window_size < 8:
        parser.error("Invalid window")

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        start = args.window * args.window_size
        stop = min(start + args.window_size, len(handle["data"]))
        data = np.asarray(handle["data"][start:stop], float)
        truth = (np.asarray(handle["labels"][start:stop]).reshape(-1)
                 if "labels" in handle else None)
    populations, count = batch_aoa_populations(data[:, names.index("AoA")])
    blob_ids = group_window(data, names, populations, count, 8, .20, 6., 8)
    selected = np.flatnonzero(np.isin(blob_ids, args.blobs))
    if not len(selected):
        parser.error("No PDWs in the selected blobs")
    pdws = [PulseMeasurement(start + int(index) + 1,
        float(data[index, names.index("ToA")] * 1e-6),
        float(data[index, names.index("Frequency")] * 1e6),
        float(data[index, names.index("PulseWidth")] * 1e-6),
        float(data[index, names.index("AoA")])) for index in selected]
    association = FixedPRIPulseAssociation(tolerance_s=args.tolerance_us * 1e-6)
    trains, unexplained, assignments = association.process(
        pdws, blob_ids[selected], f"W{args.window}")
    families = CrossRFTrainLinker().link(trains)
    truth_by_pdw = ({pdw.pdw_id: truth[index].item()
                     for pdw, index in zip(pdws, selected)} if truth is not None else {})
    train_by_id = {train.observation.train_id: train for train in trains}

    family_rows = []
    for family in families:
        pdw_ids = tuple(pdw_id for train_id in family.train_ids
                        for pdw_id in train_by_id[train_id].observation.pdw_ids)
        counts = Counter(truth_by_pdw[pdw_id] for pdw_id in pdw_ids) if truth_by_pdw else Counter()
        row = asdict(family)
        row.update(pdw_count=len(pdw_ids), truth_counts_eval=dict(counts),
                   all_trains_phase_aligned=family.phase_cluster_count == 1,
                   interpretation=("COMMON_LATTICE_CANDIDATE"
                       if family.phase_cluster_count == 1 else
                       "COMMON_PRI_FAMILY_MULTIPLE_PHASES"))
        family_rows.append(row)

    train_rows = []
    for train in trains:
        observation = train.observation
        toa = np.asarray(observation.toa_sequence_s, float)
        residual = ((toa - train.phase_epoch_s + observation.pri_s / 2.)
                    % observation.pri_s - observation.pri_s / 2.)
        train_rows.append({
            "train_id": observation.train_id,
            "blob_id": train.blob_id,
            "frequency_hz": observation.frequency_hz,
            "pri_s": observation.pri_s,
            "phase_epoch_s": train.phase_epoch_s,
            "pulse_count": observation.pulse_count,
            "occupied_slots": train.occupied_slots,
            "span_slots": train.span_slots,
            "mean_residual_s": float(np.mean(residual)),
            "mean_absolute_residual_s": float(np.mean(np.abs(residual))),
            "rms_residual_s": float(np.sqrt(np.mean(residual ** 2))),
            "pdw_ids": list(observation.pdw_ids),
        })

    result = {
        "window": args.window,
        "selected_blobs": args.blobs,
        "selected_pdws": len(pdws),
        "candidate_trains": len(trains),
        "associated_pdws": sum(value is not None for value in assignments.values()),
        "unexplained_pdws": len(unexplained),
        "train_metrics": train_rows,
        "families": family_rows,
        "truth_is_evaluation_only": True,
        "physical_identity_decided": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross_rf_families.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
