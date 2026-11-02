"""Run one measurement-selected blob track across the full Config 101 record.

The seed blobs define the development signal without exposing truth to the
tracker. Every later window is screened using the seed RF/AOA/PW envelope,
then bounded gap-PRI blob summaries are sent to the persistent tracker.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_multiscale_rf import group_window as multiscale_group_window
from run_cross_rf_confidence import circular_mean_deg, group_stats, select_groups
from testbed.blob_observation import summarize_blob_trains, summarize_blob_pri
from testbed.pipeline_provenance import run_manifest
from testbed.blob_tracker import BlobEmitterTracker
from testbed.pulse_association_pipeline import FixedPRIPulseAssociation, PulseMeasurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--seed-window", type=int, default=0)
    parser.add_argument("--seed-blobs", type=int, nargs="+", default=None)
    parser.add_argument("--evaluation-truth-id", type=int, default=46)
    parser.add_argument("--output", default="results/long_blob_tracker_e46.json")
    parser.add_argument("--pipeline", choices=("integrated", "legacy"), default="integrated")
    args = parser.parse_args()
    if args.seed_blobs is None:
        if args.seed_window != 0 or args.window_size != 10000:
            parser.error("Supply explicit seed blobs when changing the seed window or window size")
        args.seed_blobs = list(range(10, 34) if args.pipeline == "integrated" else range(8, 16))
    def group_window(data, names, populations, count, *legacy_parameters):
        return (rf_pw_groups(data, names, populations, count)
            if args.pipeline == "integrated" else
            multiscale_group_window(data, names, populations, count, *legacy_parameters))
    if args.window_size < 8 or args.seed_window < 0:
        parser.error("Invalid window configuration")

    with h5py.File(args.file) as handle:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in handle["metadata/feature_names"][:]]
        data_all = handle["data"]
        truth_all = handle.get("labels")
        total = len(data_all)
        toa_col = names.index("ToA")
        rf_col = names.index("Frequency")
        pw_col = names.index("PulseWidth")
        aoa_col = names.index("AoA")
        origin_s = float(data_all[0, toa_col] * 1e-6)
        recording_end_s = float(data_all[-1, toa_col] * 1e-6)

        seed_start = args.seed_window * args.window_size
        seed_data = np.asarray(data_all[seed_start:seed_start + args.window_size], float)
        populations, count = batch_aoa_populations(seed_data[:, aoa_col])
        seed_labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
        seed_mask = np.isin(seed_labels, args.seed_blobs)
        if not np.any(seed_mask):
            parser.error("Seed blobs contain no PDWs")
        seed_stats = group_stats(seed_data, names, seed_labels)
        modes = sorted(seed_stats[blob]["frequency_hz"] for blob in args.seed_blobs
                       if blob in seed_stats)
        if len(modes) != len(args.seed_blobs):
            parser.error("One or more seed blobs were not found")
        seed_aoa = circular_mean_deg(seed_data[seed_mask, aoa_col])
        seed_pw_s = float(np.median(seed_data[seed_mask, pw_col]) * 1e-6)
        seed_pw_modes = sorted({seed_stats[blob]["pulse_width_s"] for blob in args.seed_blobs})

        tracker = BlobEmitterTracker(max_gap_s=10.,
            max_tracks=16, beam_width=16)
        manifest = run_manifest(args.file, grouping="hdbscan" if args.pipeline == "integrated" else "multiscale",
            measurement_mode="histogram" if args.pipeline == "integrated" else "train",
            settings=vars(args), tracker=tracker)
        association = FixedPRIPulseAssociation(
            tolerance_s=5. * 1e-6, min_pulses=6)
        primary_track_id = None
        monitor = []
        evaluation = []
        windows = math.ceil(total / args.window_size)

        for window in range(windows):
            start = window * args.window_size
            stop = min(start + args.window_size, total)
            data = np.asarray(data_all[start:stop], float)
            if len(data) < 8:
                continue
            populations, count = batch_aoa_populations(data[:, aoa_col])
            labels = group_window(data, names, populations, count, 8, .20, 6., 8)
            stats = group_stats(data, names, labels)
            chosen = ([] if window < args.seed_window else
                list(args.seed_blobs) if window == args.seed_window else
                sorted({blob for pw in (seed_pw_modes if args.pipeline == "integrated" else [seed_pw_s])
                    for blob in select_groups(stats, modes=modes, aoa=seed_aoa, pw=pw,
                        rf_gate_hz=40e6, aoa_gate_deg=3., pw_gate_s=.12e-6)}))
            selected = np.isin(labels, chosen)
            observations = ()
            unexplained_count = 0
            last_association_confidence = None
            primary_updates = 0
            if np.any(selected):
                selected_indices = np.flatnonzero(selected)
                selected_ids = start + selected_indices
                selected_data = data[selected]
                selected_labels = labels[selected]
                pdws = tuple(PulseMeasurement(int(pdw_id),
                    float(row[toa_col] * 1e-6), float(row[rf_col] * 1e6),
                    float(row[pw_col] * 1e-6), float(row[aoa_col]))
                    for pdw_id, row in zip(selected_ids, selected_data))
                if args.pipeline == "integrated":
                    observations = summarize_blob_pri(selected_data, names, selected_labels, f"W{window}")
                else:
                    trains, unexplained, _ = association.process(pdws, selected_labels, f"W{window}")
                    unexplained_count = len(unexplained)
                    observations = summarize_blob_trains(selected_data, names,
                        selected_labels, selected_ids, f"W{window}", trains)
                for observation in observations:
                    tracker.process(observation)
                    best = tracker.hypotheses[0]
                    assigned = dict(best.recent_assignments).get(observation.observation_id)
                    if primary_track_id is not None and assigned == primary_track_id:
                        primary_updates += 1
                        last_association_confidence = tracker.association_metrics(
                            observation.observation_id)["best_confidence"]

            if primary_track_id is None and window >= args.seed_window and tracker.hypotheses:
                best_tracks = tracker.hypotheses[0].tracks
                if best_tracks:
                    primary_track_id = max(best_tracks,
                        key=lambda track: (track.updates, len(track.rf_modes))).track_id
                    primary_updates = next(track.updates for track in best_tracks
                                           if track.track_id == primary_track_id)

            end_s = float(data[-1, toa_col] * 1e-6)
            for sample_time_s, sample_tracks in tracker.advance_to(end_s):
                current = next((item for item in sample_tracks
                    if item.track_id == primary_track_id), None)
                state = (None if current is None else
                         tracker.track_monitor(current, sample_time_s))
                if state is not None:
                    confidence = state["track_confidence"]
                    state["status"] = ("CONFIRMED" if confidence >= .8 else
                        "COASTING" if confidence >= .3 else "LOST")
                monitor.append({"window": None,
                    "time_from_start_s": sample_time_s - origin_s,
                    "window_start_from_start_s": sample_time_s - origin_s,
                    "candidate_blob_count": 0, "measurement_count": 0,
                    "measurement_pulse_count": 0,
                    "unexplained_pulse_count": 0, "primary_updates": 0,
                    "last_association_confidence": None, "track": state})
            tracker.complete_window(f"W{window}", end_s)
            track = None
            for hypothesis in tracker.hypotheses:
                track = next((item for item in hypothesis.tracks
                              if item.track_id == primary_track_id), None)
                if track is not None:
                    break
            state = tracker.track_monitor(track, end_s) if track is not None else None
            if state is not None:
                confidence = state["track_confidence"]
                state["status"] = ("CONFIRMED" if confidence >= .8 else
                                   "COASTING" if confidence >= .3 else "LOST")
            monitor.append({"window": window,
                "time_from_start_s": end_s - origin_s,
                "window_start_from_start_s": float(data[0, toa_col] * 1e-6) - origin_s,
                "candidate_blob_count": len(chosen),
                "measurement_count": len(observations),
                "measurement_pulse_count": sum(obs.pulse_count for obs in observations),
                "unexplained_pulse_count": unexplained_count,
                "primary_updates": primary_updates,
                "last_association_confidence": last_association_confidence,
                "track": state})

            if truth_all is not None:
                truth = np.asarray(truth_all[start:stop]).reshape(-1)
                selected_truth = truth[selected]
                evaluation.append({"window": window,
                    "truth_pulses": int(np.sum(truth == args.evaluation_truth_id)),
                    "selected_pulses": int(np.sum(selected)),
                    "selected_truth_pulses": int(np.sum(
                        selected_truth == args.evaluation_truth_id))})

    result = {"run_manifest": manifest, "truth_is_evaluation_only": True,
        "duration_s": recording_end_s - origin_s,
        "recording_windows": windows,
        "primary_track_id": primary_track_id,
        "seed": {"window": args.seed_window, "blob_ids": args.seed_blobs,
            "frequency_modes_hz": modes, "aoa_deg": seed_aoa,
            "pulse_width_s": seed_pw_s},
        "monitor": monitor, "evaluation": evaluation}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"duration_s": result["duration_s"],
        "recording_windows": windows, "monitor_samples": len(monitor),
        "primary_track_id": primary_track_id,
        "final_status": monitor[-1]["track"]["status"] if monitor[-1]["track"] else None,
        "updates": sum(row["primary_updates"] for row in monitor)}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
