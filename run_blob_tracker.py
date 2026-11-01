"""Track fixed-membership blob observations, without pulse-train discovery.

python3 run_blob_tracker.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/blob_tracker_focus

The grouping adapter summarizes each original blob ONCE. The tracker sees
only those bounded summaries. It cannot access or reassign individual PDWs.
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
from testbed.blob_observation import summarize_blobs
from testbed.blob_tracker import BlobEmitterTracker


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
    ap.add_argument("--grouping", choices=("multiscale", "hdbscan"), default="multiscale")
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
    tracker = BlobEmitterTracker(beam_width=a.beam_width, max_tracks=a.max_tracks,
        max_modes=a.max_modes, ambiguity_margin=a.ambiguity_margin)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    observations_rows, decisions, evaluations, windows = [], [], [], []
    with h5py.File(a.file) as h, (out / "hypotheses.jsonl").open("w") as history:
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
            if not np.any(selected):
                ap.error(f"No selected blobs in window {wi}")
            # This is grouping-stage feature export, not tracker processing.
            observations = summarize_blobs(x[selected], names, original[selected], f"W{wi}",
                toa_scale=a.toa_scale, frequency_scale=a.frequency_scale, pw_scale=a.pw_scale)
            for obs in observations:
                tracker.process(obs)
                observations_rows.append(dict(window=wi, **asdict(obs)))
            if not np.array_equal(original, immutable_membership):
                raise RuntimeError("Original blob membership changed")
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
                decisions.append(dict(window=wi, observation_id=obs.observation_id,
                    blob=obs.blob_id, pulse_count=obs.pulse_count, status=options["status"],
                    candidate_track_ids=json.dumps(options["candidate_track_ids"])))
                print(f" W{wi} B{obs.blob_id}: {obs.pulse_count} PDWs, "
                      f"RF median={obs.frequency_median_hz / a.frequency_scale:.3f}, "
                      f"AOA={obs.aoa_deg:.3f}, {options}", flush=True)
            row = dict(window=wi, selected_blobs=len(observations),
                selected_pdws=sum(o.pulse_count for o in observations),
                membership_changed=False,
                retained_hypotheses=len(tracker.hypotheses),
                plausible_hypotheses=len(plausible),
                retained_track_counts_for_window=json.dumps(sorted(set(partitions))),
                one_track_for_all_selected_blobs_retained=1 in partitions)
            # Truth is used only after all whole-blob association decisions.
            if "labels" in h:
                truth = np.asarray(h["labels"][start:stop]).reshape(-1)
                for obs in observations:
                    counts = Counter(truth[original == obs.blob_id].tolist())
                    evaluations.append(dict(window=wi, observation_id=obs.observation_id,
                        truth_counts_eval=json.dumps(dict(counts), sort_keys=True),
                        blob_purity_pct=100 * max(counts.values()) / obs.pulse_count))
                row["true_emitters_in_selected_pdws_eval"] = len(np.unique(truth[selected]))
            windows.append(row)
            history.write(json.dumps(dict(window=wi, **snapshot)) + "\n")
            history.flush()
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
