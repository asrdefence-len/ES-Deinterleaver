"""Supported AOA + RF/PW HDBSCAN → local gap-PRI summary → Kalman tracker.

One summary per frozen blob is the default. Legacy multiscale grouping and
fixed-train extraction require explicit flags and are marked in the manifest.
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_multiscale_rf import group_window
from testbed.blob_observation import summarize_blobs, summarize_blob_trains, summarize_blob_pri
from testbed.pipeline_provenance import run_manifest
from testbed.blob_tracker import BlobEmitterTracker
from testbed.pulse_association_pipeline import FixedPRIPulseAssociation, PulseMeasurement


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--windows", type=int, default=1, help="0 means all remaining windows")
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--blobs", nargs="+", type=int)
    ap.add_argument("--grouping", choices=("multiscale", "hdbscan"), default="hdbscan")
    ap.add_argument("--measurement-mode", choices=("histogram", "train", "blob"), default="histogram",
                    help="histogram exports local gap mean, spread and up to two humps")
    ap.add_argument("--pri-bin-us", type=float, default=2.)
    ap.add_argument("--pri-min-gap-us", type=float, default=10.)
    ap.add_argument("--pri-max-gap-us", type=float, default=400.)
    ap.add_argument("--pri-neighbourhood", type=float, default=.15)
    ap.add_argument("--max-gap-s", type=float, default=.3)
    ap.add_argument("--consistency-log", action="store_true")
    ap.add_argument("--toa-scale", type=float, default=1e-6)
    ap.add_argument("--frequency-scale", type=float, default=1e6)
    ap.add_argument("--pw-scale", type=float, default=1e-6)
    ap.add_argument("--beam-width", type=int, default=16)
    ap.add_argument("--max-tracks", type=int, default=64)
    ap.add_argument("--max-modes", type=int, default=8)
    ap.add_argument("--ambiguity-margin", type=float, default=2.)
    ap.add_argument("--output", default="results/blob_tracker")
    a = ap.parse_args()
    if a.window < 0 or a.windows < 0 or a.window_size < 8:
        ap.error("Invalid window selection")
    if a.blobs is not None and a.windows != 1:
        ap.error("Blob IDs are window-local; select --blobs for one window only")
    if not all(np.isfinite(v) and v > 0 for v in (a.toa_scale, a.frequency_scale, a.pw_scale)):
        ap.error("Unit scales must be finite and positive")
    if (not 0 <= a.pri_min_gap_us < a.pri_max_gap_us or a.pri_bin_us <= 0
            or not 0 < a.pri_neighbourhood < 1):
        ap.error("Invalid PRI histogram settings")
    tracker = BlobEmitterTracker(beam_width=a.beam_width, max_tracks=a.max_tracks,
        max_modes=a.max_modes, ambiguity_margin=a.ambiguity_margin, max_gap_s=a.max_gap_s)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_manifest(a.file, grouping=a.grouping,
        measurement_mode=a.measurement_mode, settings=vars(a), tracker=tracker)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Pipeline:", json.dumps(manifest["pipeline"]), flush=True)
    observations_rows, decisions, evaluations, windows, monitor_rows = [], [], [], [], []
    with h5py.File(a.file) as h, (out / "hypotheses.jsonl").open("w") as history, \
            (out / "consistency.jsonl").open("w") as consistency:
        if a.consistency_log:
            tracker.consistency_sink = lambda event: consistency.write(json.dumps(event, allow_nan=False) + "\n")
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in h["metadata/feature_names"][:]]
        if not {"ToA", "Frequency", "PulseWidth", "AoA"}.issubset(names):
            ap.error("Missing required observation features")
        limit = (len(h["data"]) + a.window_size - 1) // a.window_size
        if a.windows:
            limit = min(limit, a.window + a.windows)
        for wi in range(a.window, limit):
            start = wi * a.window_size
            stop = min(start + a.window_size, len(h["data"]))
            if stop - start < 8:
                continue
            x = np.asarray(h["data"][start:stop], float)
            populations, npop = batch_aoa_populations(x[:, names.index("AoA")])
            original = (group_window(x, names, populations, npop, 8, .20, 6., 8)
                if a.grouping == "multiscale" else rf_pw_groups(x, names, populations, npop))
            immutable_membership = original.copy()
            selected = np.isin(original, a.blobs) if a.blobs is not None else original >= 0
            if not np.any(selected) and a.blobs is not None:
                ap.error(f"No selected blobs in window {wi}")
            # This is grouping-stage feature export, not tracker PDW processing.
            selected_data = x[selected]
            selected_labels = original[selected]
            selected_ids = np.arange(start, stop, dtype=int)[selected]
            selected_window_indices = np.flatnonzero(selected)
            evaluation_indices = {}
            timing_unexplained = 0
            if a.measurement_mode == "train":
                columns = {name: names.index(name) for name in
                           ("ToA", "Frequency", "PulseWidth", "AoA")}
                pdws = tuple(PulseMeasurement(int(pdw_id),
                    float(row[columns["ToA"]] * a.toa_scale),
                    float(row[columns["Frequency"]] * a.frequency_scale),
                    float(row[columns["PulseWidth"]] * a.pw_scale),
                    float(row[columns["AoA"]]))
                    for pdw_id, row in zip(selected_ids, selected_data))
                trains, unexplained, _ = FixedPRIPulseAssociation(
                    tolerance_s=5. * 1e-6).process(
                    pdws, selected_labels, f"W{wi}")
                timing_unexplained = len(unexplained)
                observations = summarize_blob_trains(selected_data, names,
                    selected_labels, selected_ids, f"W{wi}", trains,
                    toa_scale=a.toa_scale, frequency_scale=a.frequency_scale,
                    pw_scale=a.pw_scale)
                selected_lookup = {int(pdw_id): int(index)
                    for pdw_id, index in zip(selected_ids, selected_window_indices)}
                train_by_id = {train.observation.train_id: train for train in trains}
                for obs in observations:
                    if obs.train_id is None:
                        evaluation_indices[obs.observation_id] = np.flatnonzero(
                            original == obs.blob_id)
                    else:
                        evaluation_indices[obs.observation_id] = np.asarray([
                            selected_lookup[int(pdw_id)] for pdw_id in
                            train_by_id[obs.train_id].observation.pdw_ids], int)
            else:
                summarizer = summarize_blob_pri if a.measurement_mode == "histogram" else summarize_blobs
                pri_settings = ({"bin_width_s": a.pri_bin_us * 1e-6,
                    "min_gap_s": a.pri_min_gap_us * 1e-6,
                    "max_gap_s": a.pri_max_gap_us * 1e-6,
                    "neighbourhood": a.pri_neighbourhood}
                    if a.measurement_mode == "histogram" else {})
                observations = summarizer(selected_data, names,
                    selected_labels, f"W{wi}", toa_scale=a.toa_scale,
                    frequency_scale=a.frequency_scale, pw_scale=a.pw_scale, **pri_settings)
                evaluation_indices = {obs.observation_id: np.flatnonzero(
                    original == obs.blob_id) for obs in observations}
            for obs in observations:
                options = tracker.process(obs)
                metrics = tracker.association_metrics(obs.observation_id)
                best = tracker.hypotheses[0]
                best_track_id = dict(best.recent_assignments).get(obs.observation_id)
                best_track = next((track for track in best.tracks
                    if track.track_id == best_track_id), None)
                monitor_rows.append({"sequence": len(monitor_rows) + 1,
                    "window": wi, "observation_id": obs.observation_id,
                    "blob_id": obs.blob_id, "start_s": obs.start_s,
                    "end_s": obs.end_s, "pulse_count": obs.pulse_count,
                    "measurement": {"pri_s": obs.pri_s,
                        "pri_sigma_s": obs.pri_sigma_s,
                        "pri_histogram": (None if obs.pri_histogram is None else asdict(obs.pri_histogram)),
                        "timing_rms_residual_s": obs.timing_rms_residual_s,
                        "timing_pulse_count": obs.timing_pulse_count},
                    "status": options["status"],
                    "candidate_track_ids": options["candidate_track_ids"],
                    "best_hypothesis_track_id": best_track_id,
                    "best_confidence": metrics["best_confidence"],
                    "relative_association_confidence":
                        metrics["relative_association_confidence"],
                    "track": (None if best_track is None else
                              tracker.track_monitor(best_track, obs.end_s))})
                observations_rows.append(dict(window=wi, **asdict(obs)))
            if not np.array_equal(original, immutable_membership):
                raise RuntimeError("Original blob membership changed")
            window_end_s = float(x[-1, names.index("ToA")] * a.toa_scale)
            tracker.advance_to(window_end_s)
            tracker.complete_window(f"W{wi}", window_end_s)
            snapshot = tracker.snapshot()
            plausible = tracker.plausible_hypotheses()
            observation_ids = {o.observation_id for o in observations}
            partitions = []
            for hypothesis in plausible:
                mapping = dict(hypothesis.recent_assignments)
                assignments = [mapping.get(oid) for oid in observation_ids]
                # A whole-window partition can only be reported while all of
                # its observations remain in the bounded association history.
                if observation_ids.issubset(mapping) and None not in assignments:
                    partitions.append(len(set(assignments)))
            for obs in observations:
                options = tracker.association_options(obs.observation_id)
                metrics = tracker.association_metrics(obs.observation_id)
                decisions.append(dict(window=wi, observation_id=obs.observation_id,
                    blob=obs.blob_id, pulse_count=obs.pulse_count, status=options["status"],
                    candidate_track_ids=json.dumps(options["candidate_track_ids"]),
                    best_confidence=metrics["best_confidence"],
                    relative_association_confidence=json.dumps(
                        metrics["relative_association_confidence"], sort_keys=True),
                    confidence_is_calibrated_probability=False))
                print(f" W{wi} B{obs.blob_id}: {obs.pulse_count} PDWs, "
                      f"RF median={obs.frequency_median_hz / a.frequency_scale:.3f}, "
                      f"AOA={obs.aoa_deg:.3f}, {options}, "
                      f"confidence={metrics['best_confidence']:.3f}", flush=True)
            row = dict(window=wi,
                selected_blobs=len({o.blob_id for o in observations}),
                tracker_measurements=len(observations),
                selected_pdws=int(np.count_nonzero(selected)),
                measurement_pdws=sum(o.pulse_count for o in observations),
                measurement_mode=a.measurement_mode,
                timing_unexplained_pdws=timing_unexplained,
                membership_changed=False,
                retained_hypotheses=len(tracker.hypotheses),
                plausible_hypotheses=len(plausible),
                retained_track_counts_for_window=json.dumps(sorted(set(partitions))),
                one_track_for_all_selected_blobs_retained=1 in partitions)
            # Truth is used only after all whole-blob association decisions.
            if "labels" in h:
                truth = np.asarray(h["labels"][start:stop]).reshape(-1)
                for obs in observations:
                    indices = evaluation_indices[obs.observation_id]
                    counts = Counter(truth[indices].tolist())
                    evaluations.append(dict(window=wi, observation_id=obs.observation_id,
                        truth_counts_eval=json.dumps(dict(counts), sort_keys=True),
                        blob_purity_pct=100 * max(counts.values()) / len(indices)))
                row["true_emitters_in_selected_pdws_eval"] = len(np.unique(truth[selected]))
            windows.append(row)
            history.write(json.dumps(dict(window=wi, **snapshot)) + "\n")
            history.flush()
            with (out / "blob_monitor.jsonl").open("w") as monitor_file:
                for monitor_row in monitor_rows:
                    monitor_file.write(json.dumps(monitor_row) + "\n")
            write_csv(out / "blob_observations.csv", observations_rows, ["observation_id"])
            write_csv(out / "blob_associations.csv", decisions, ["observation_id", "status"])
            write_csv(out / "blob_evaluation.csv", evaluations, ["observation_id", "truth_counts_eval"])
            write_csv(out / "windows.csv", windows, ["window", "selected_blobs"])
            print(f"Window {wi}: {row}", flush=True)
    if not windows:
        ap.error("No available windows")
    (out / "tracker_snapshot.json").write_text(json.dumps(tracker.snapshot(), indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    print(f"Saved {out}; original blob membership is unchanged.")
    print("One-emitter compatibility is a hypothesis, not an established identity; costs are not probabilities.")


if __name__ == "__main__":
    main()
