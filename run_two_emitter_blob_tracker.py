"""Track selected Config 101 emitter families through the shared blob pipeline."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from contextlib import ExitStack
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


SCENARIOS = {
    "e46_only": {
        "E46": {"seed_window": 0, "blobs": tuple(range(8, 16)), "truth_id": 46},
    },
    "separated": {
        "E46": {"seed_window": 0, "blobs": tuple(range(8, 16)), "truth_id": 46},
        "E33": {"seed_window": 0, "blobs": (62,), "truth_id": 33},
    },
    "moderate": {
        "E46": {"seed_window": 0, "blobs": tuple(range(8, 16)), "truth_id": 46},
        "E38": {"seed_window": 3, "blobs": (6,), "truth_id": 38},
    },
}


def status(state):
    confidence = state["track_confidence"]
    return "CONFIRMED" if confidence >= .8 else "COASTING" if confidence >= .3 else "LOST"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="separated")
    parser.add_argument("--output", help="Result path; E46-only defaults to its own versioned folder")
    parser.add_argument("--consistency-log", help="Stream branch-labelled Kalman diagnostics to JSONL")
    parser.add_argument("--pipeline", choices=("integrated", "legacy"), default="integrated")
    args = parser.parse_args()
    if args.output is None:
        args.output = ("results/e46_only/v1/e46_only.json" if args.scenario == "e46_only"
                       else "results/two_emitter_blob_tracker.json")
    definitions = {name: dict(values) for name, values in SCENARIOS[args.scenario].items()}
    if args.pipeline == "integrated":
        definitions["E46"]["blobs"] = tuple(range(10, 34))
        if "E38" in definitions:
            definitions["E38"]["blobs"] = tuple(range(14, 26))
        if "E33" in definitions:
            definitions["E33"]["blobs"] = (128, 129)
    def group_window(data, names, populations, count, *legacy_parameters):
        return (rf_pw_groups(data, names, populations, count)
            if args.pipeline == "integrated" else
            multiscale_group_window(data, names, populations, count, *legacy_parameters))
    if args.window_size < 8:
        parser.error("Window size must be at least eight")

    with h5py.File(args.file) as handle, ExitStack() as diagnostics_stack:
        sink = None
        if args.consistency_log:
            log_path = Path(args.consistency_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = diagnostics_stack.enter_context(log_path.open("w"))
            def sink(event):
                log_handle.write(json.dumps(event, allow_nan=False) + "\n")
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in handle["metadata/feature_names"][:]]
        data_all, truth_all = handle["data"], handle.get("labels")
        total = len(data_all)
        columns = {name: names.index(name) for name in
                   ("ToA", "Frequency", "PulseWidth", "AoA")}
        origin_s = float(data_all[0, columns["ToA"]] * 1e-6)
        recording_end_s = float(data_all[-1, columns["ToA"]] * 1e-6)

        families = {}
        for name, definition in definitions.items():
            seed_start = definition["seed_window"] * args.window_size
            seed_data = np.asarray(data_all[seed_start:seed_start + args.window_size], float)
            populations, count = batch_aoa_populations(seed_data[:, columns["AoA"]])
            seed_labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
            seed_stats = group_stats(seed_data, names, seed_labels)
            mask = np.isin(seed_labels, definition["blobs"])
            if not np.any(mask):
                parser.error(f"{name} seed blobs were not found")
            modes = [seed_stats[blob]["frequency_hz"] for blob in definition["blobs"]
                     if blob in seed_stats]
            if len(modes) != len(definition["blobs"]):
                parser.error(f"One or more {name} seed blobs were not found")
            families[name] = {**definition, "frequency_modes_hz": modes,
                "pulse_width_modes_s": sorted({seed_stats[blob]["pulse_width_s"]
                    for blob in definition["blobs"]}),
                "aoa_deg": circular_mean_deg(seed_data[mask, columns["AoA"]]),
                "pulse_width_s": float(np.median(
                    seed_data[mask, columns["PulseWidth"]]) * 1e-6)}

        tracker = BlobEmitterTracker(max_gap_s=10.,
            max_tracks=16, beam_width=32, consolidation_min_updates=2,
            consolidation_gate_sigma=4., consistency_sink=sink)
        manifest = run_manifest(args.file, grouping="hdbscan" if args.pipeline == "integrated" else "multiscale",
            measurement_mode="histogram" if args.pipeline == "integrated" else "train",
            settings=vars(args), tracker=tracker)
        association = FixedPRIPulseAssociation(
            tolerance_s=5. * 1e-6, min_pulses=6)
        track_ids = {}
        monitor, events, evaluation = [], [], []
        windows = math.ceil(total / args.window_size)

        for window in range(windows):
            start, stop = window * args.window_size, min((window + 1) * args.window_size, total)
            data = np.asarray(data_all[start:stop], float)
            if len(data) < 8:
                continue
            populations, count = batch_aoa_populations(data[:, columns["AoA"]])
            labels = group_window(data, names, populations, count, 8, .20, 6., 8)
            stats = group_stats(data, names, labels)
            chosen_by_family = {}
            for name, family in families.items():
                chosen_by_family[name] = ([] if window < family["seed_window"] else
                    list(family["blobs"]) if window == family["seed_window"] else
                    sorted({blob for pw in (family["pulse_width_modes_s"]
                            if args.pipeline == "integrated" else [family["pulse_width_s"]])
                        for blob in select_groups(stats, modes=family["frequency_modes_hz"],
                            aoa=family["aoa_deg"], pw=pw,
                            rf_gate_hz=40e6, aoa_gate_deg=3., pw_gate_s=.12e-6)}))
            chosen = sorted({blob for values in chosen_by_family.values() for blob in values})
            selected = np.isin(labels, chosen)
            window_observations = []
            observation_family = {}
            train_truth_counts = {}
            if np.any(selected):
                selected_indices = np.flatnonzero(selected)
                selected_ids = start + selected_indices
                selected_data, selected_labels = data[selected], labels[selected]
                pdws = tuple(PulseMeasurement(int(pdw_id),
                    float(row[columns["ToA"]] * 1e-6),
                    float(row[columns["Frequency"]] * 1e6),
                    float(row[columns["PulseWidth"]] * 1e-6),
                    float(row[columns["AoA"]]))
                    for pdw_id, row in zip(selected_ids, selected_data))
                if args.pipeline == "integrated":
                    trains = ()
                    window_observations = list(summarize_blob_pri(selected_data, names,
                        selected_labels, f"W{window}"))
                else:
                    trains, _, _ = association.process(pdws, selected_labels, f"W{window}")
                    window_observations = list(summarize_blob_trains(selected_data, names,
                        selected_labels, selected_ids, f"W{window}", trains))
                for observation in window_observations:
                    candidates = [name for name, blob_ids in chosen_by_family.items()
                                  if observation.blob_id in blob_ids]
                    observation_family[observation.observation_id] = (
                        candidates[0] if len(candidates) == 1 else "OVERLAP")
                if truth_all is not None:
                    truth = np.asarray(truth_all[start:stop]).reshape(-1)
                    selected_lookup = {int(pdw_id): int(index)
                        for pdw_id, index in zip(selected_ids, selected_indices)}
                    if args.pipeline == "integrated":
                        for observation in window_observations:
                            indices = np.flatnonzero(labels == observation.blob_id)
                            train_truth_counts[observation.observation_id] = dict(Counter(map(int, truth[indices])))
                    for train in trains:
                        indices = [selected_lookup[int(pdw_id)]
                                   for pdw_id in train.observation.pdw_ids]
                        train_truth_counts[train.observation.train_id] = dict(
                            Counter(map(int, truth[indices])))

            latest_confidence = {name: None for name in families}
            updates = {name: 0 for name in families}
            for observation in sorted(window_observations,
                                      key=lambda item: (item.start_s, item.observation_id)):
                pre_costs = {name: tracker.candidate_cost(observation, track_id)
                             for name, track_id in track_ids.items()}
                tracker.process(observation)
                best = tracker.hypotheses[0]
                assigned = dict(best.recent_assignments).get(observation.observation_id)
                metrics = tracker.association_metrics(observation.observation_id)
                family_name = observation_family[observation.observation_id]
                events.append({"window": window,
                    "time_from_start_s": observation.end_s - origin_s,
                    "observation_id": observation.observation_id,
                    "blob_id": observation.blob_id,
                    "source_family": family_name, "assigned_track_id": assigned,
                    "best_confidence": metrics["best_confidence"],
                    "pre_update_candidate_costs": pre_costs,
                    "pulse_count": observation.pulse_count,
                    "measurement": {"aoa_deg": observation.aoa_deg,
                        "pulse_width_s": observation.pw_median_s,
                        "frequency_hz": observation.frequency_median_hz,
                        "pri_s": observation.pri_s,
                        "pri_sigma_s": observation.pri_sigma_s,
                        "pri_histogram": (None if observation.pri_histogram is None else asdict(observation.pri_histogram)),
                        "timing_rms_residual_s": observation.timing_rms_residual_s},
                    "truth_counts_eval": train_truth_counts.get(observation.observation_id, {})})
                for name, track_id in track_ids.items():
                    if assigned == track_id:
                        updates[name] += 1
                        latest_confidence[name] = metrics["best_confidence"]

            mapping = dict(tracker.hypotheses[0].recent_assignments)
            for name, family in families.items():
                if name not in track_ids and window == family["seed_window"]:
                    assigned = [mapping.get(obs.observation_id) for obs in window_observations
                                if observation_family[obs.observation_id] == name]
                    assigned = [value for value in assigned if value is not None]
                    if not assigned:
                        parser.error(f"No seed track was formed for {name}")
                    track_ids[name] = Counter(assigned).most_common(1)[0][0]
                    updates[name] = sum(value == track_ids[name] for value in assigned)

            end_s = float(data[-1, columns["ToA"]] * 1e-6)
            for sample_time_s, sample_tracks in tracker.advance_to(end_s):
                sample_states = {}
                for name, track_id in track_ids.items():
                    track = next((item for item in sample_tracks
                                  if item.track_id == track_id), None)
                    if track is not None:
                        state = tracker.track_monitor(track, sample_time_s)
                        state["status"] = status(state)
                        sample_states[name] = state
                monitor.append({"window": None,
                    "time_from_start_s": sample_time_s - origin_s,
                    "measurement_counts": {name: 0 for name in families},
                    "track_updates": {name: 0 for name in families},
                    "latest_association_confidence": {
                        name: None for name in families},
                    "tracks": sample_states})
            tracker.complete_window(f"W{window}", end_s)
            states = {}
            for name, track_id in track_ids.items():
                track = next((item for hypothesis in tracker.hypotheses
                    for item in hypothesis.tracks if item.track_id == track_id), None)
                if track is not None:
                    state = tracker.track_monitor(track, end_s)
                    state["status"] = status(state)
                    states[name] = state
            monitor.append({"window": window,
                "time_from_start_s": end_s - origin_s,
                "measurement_counts": {name: sum(
                    observation_family[obs.observation_id] == name
                    for obs in window_observations) for name in families},
                "track_updates": updates,
                "latest_association_confidence": latest_confidence,
                "tracks": states})

            if truth_all is not None:
                truth = np.asarray(truth_all[start:stop]).reshape(-1)
                row = {"window": window}
                for name, family in families.items():
                    mask = np.isin(labels, chosen_by_family[name])
                    row[name] = {"truth_pulses": int(np.sum(truth == family["truth_id"])),
                        "selected_pulses": int(np.sum(mask)),
                        "selected_truth_pulses": int(np.sum(truth[mask] == family["truth_id"]))}
                evaluation.append(row)

    duration_s = recording_end_s - origin_s

    scores = {}
    for name, family in families.items():
        truth_pulses = sum(row[name]["truth_pulses"] for row in evaluation)
        selected = sum(row[name]["selected_pulses"] for row in evaluation)
        hits = sum(row[name]["selected_truth_pulses"] for row in evaluation)
        scores[name] = {"truth_pulses": truth_pulses, "selected_pulses": selected,
            "selected_truth_pulses": hits,
            "selection_purity": hits / selected if selected else None,
            "selection_coverage": hits / truth_pulses if truth_pulses else None}
    aliases = tracker.consolidated_aliases
    def canonical(track_id):
        seen = set()
        while track_id in aliases and track_id not in seen:
            seen.add(track_id)
            track_id = aliases[track_id]
        return track_id
    for event in events:
        event["canonical_track_id"] = canonical(event["assigned_track_id"])
    cross_assignments = {f"{source}_to_{target}": 0
        for source in families for target in families if source != target}
    for event in events:
        counts = {int(key): value for key, value in event["truth_counts_eval"].items()}
        if not counts:
            continue
        truth_id = max(counts, key=counts.get)
        source = next((name for name, family in families.items()
                       if family["truth_id"] == truth_id), None)
        if source is not None:
            for target, target_track_id in track_ids.items():
                if target != source and event["canonical_track_id"] == target_track_id:
                    cross_assignments[f"{source}_to_{target}"] += 1

    canonical_retention = {name: {"retained": sum(
        event["source_family"] == name
        and event["canonical_track_id"] == track_ids[name] for event in events),
        "total": sum(event["source_family"] == name for event in events)}
        for name in families}

    result = {"run_manifest": manifest, "truth_is_evaluation_only": True, "scenario": args.scenario,
        "duration_s": duration_s,
        "track_ids": track_ids, "families": families, "monitor": monitor,
        "events": events, "evaluation": evaluation, "evaluation_summary": scores,
        "cross_assigned_measurements": cross_assignments,
        "canonical_primary_retention": canonical_retention,
        "final_tracker_snapshot": tracker.snapshot()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"duration_s": duration_s, "track_ids": track_ids,
        "evaluation_summary": scores,
        "canonical_primary_retention": canonical_retention,
        "cross_assigned_measurements": cross_assignments}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
