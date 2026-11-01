"""Run the first integrated pulse-association / emitter-tracking baseline.

python3 run_pulse_association.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/pulse_association_focus

Turing defaults: raw TOA and PW in microseconds, RF in MHz, AOA in degrees.
Override scale arguments for other units. Truth is read only after inference.
No OT or frequency-density splitting is involved.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_multiscale_rf import group_window
from testbed.pulse_association_pipeline import (
    FixedPRIPulseAssociation, PulseAssociationTrackingPipeline, PulseMeasurement,
)


def write_csv(path, rows, fallback_fields):
    fields = list(rows[0]) if rows else fallback_fields
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--blobs", type=int, nargs="+", help="Window-local blob IDs; omit to process all blobs")
    ap.add_argument("--grouping", choices=("multiscale", "hdbscan"), default="multiscale")
    ap.add_argument("--toa-scale", type=float, default=1e-6, help="Seconds per raw TOA unit")
    ap.add_argument("--frequency-scale", type=float, default=1e6, help="Hz per raw RF unit")
    ap.add_argument("--pw-scale", type=float, default=1e-6, help="Seconds per raw PW unit")
    ap.add_argument("--pri-min-us", type=float, default=20.)
    ap.add_argument("--pri-max-us", type=float, default=200.)
    ap.add_argument("--tolerance-us", type=float, default=2.)
    ap.add_argument("--bin-width-us", type=float, default=.1)
    ap.add_argument("--min-pulses", type=int, default=20)
    ap.add_argument("--max-trains", type=int, default=4)
    ap.add_argument("--output", default="results/pulse_association")
    a = ap.parse_args()
    if (a.window < 0 or a.windows < 1 or a.window_size < 8
            or not all(np.isfinite(v) and v > 0 for v in (a.toa_scale, a.frequency_scale, a.pw_scale))):
        ap.error("Invalid windows or unit scales")
    if a.blobs is not None and a.windows > 1:
        ap.error("Blob IDs are window-local. Use --blobs with one window, or omit it for multiwindow tracking.")
    association = FixedPRIPulseAssociation(
        pri_min_s=a.pri_min_us * 1e-6, pri_max_s=a.pri_max_us * 1e-6,
        tolerance_s=a.tolerance_us * 1e-6, bin_width_s=a.bin_width_us * 1e-6,
        min_pulses=a.min_pulses, max_trains_per_blob=a.max_trains)
    tracker = PulseAssociationTrackingPipeline(association=association)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    pulse_rows, train_rows, eval_rows, window_rows = [], [], [], []
    with h5py.File(a.file) as h:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in h["metadata/feature_names"][:]]
        required = {"ToA", "Frequency", "PulseWidth", "AoA"}
        if not required.issubset(names):
            ap.error(f"Missing required features: {sorted(required - set(names))}")
        for wi in range(a.window, a.window + a.windows):
            start = wi * a.window_size
            stop = min(start + a.window_size, len(h["data"]))
            if stop <= start:
                break
            x = np.asarray(h["data"][start:stop], float)
            populations, npop = batch_aoa_populations(x[:, names.index("AoA")])
            blobs = (group_window(x, names, populations, npop, 8, .20, 6., 8)
                     if a.grouping == "multiscale" else rf_pw_groups(x, names, populations, npop))
            selected = np.flatnonzero(np.isin(blobs, a.blobs)) if a.blobs is not None else np.arange(len(x))
            if not len(selected):
                ap.error(f"No selected blobs in window {wi}")
            pdws = [PulseMeasurement(
                pdw_id=start + int(i) + 1,
                toa_s=float(x[i, names.index("ToA")] * a.toa_scale),
                frequency_hz=float(x[i, names.index("Frequency")] * a.frequency_scale),
                pulse_width_s=float(x[i, names.index("PulseWidth")] * a.pw_scale),
                aoa_deg=float(x[i, names.index("AoA")])) for i in selected]
            result = tracker.process_window(pdws, blobs[selected], f"W{wi}")
            # Evaluation starts here. Neither component has access to labels
            # or the file's transmitter configurations.
            truth = np.asarray(h["labels"][start:stop]).reshape(-1) if "labels" in h else None
            truth_by_id = {p.pdw_id: truth[i].item() for p, i in zip(pdws, selected)} if truth is not None else None
            assigned = sum(v is not None for v in result.pulse_train_assignments.values())
            row = dict(window=wi, window_pdws=len(x), selected_pdws=len(pdws),
                candidate_trains=len(result.trains), associated_pdws=assigned,
                unexplained_pdws=len(result.unexplained_pdw_ids),
                association_coverage_pct=100 * assigned / len(pdws),
                provisional_emitter_hypotheses=len(tracker.identity_manager.tracks),
                pending_train_observations=len(tracker.identity_manager.pending))
            correct = 0
            for tr in result.trains:
                obs = tr.observation
                eid = result.train_emitter_assignments[obs.train_id]
                train_rows.append(dict(window=wi, blob=tr.blob_id, train_id=obs.train_id,
                    provisional_emitter_id=eid, timing_model="FIXED_CANDIDATE",
                    pri_us=obs.pri_s / 1e-6, phase_epoch_s=tr.phase_epoch_s,
                    rms_residual_us=tr.rms_residual_s / 1e-6,
                    pdws=obs.pulse_count, occupied_slots=tr.occupied_slots, span_slots=tr.span_slots,
                    start_s=obs.start_toa_s, end_s=obs.end_toa_s,
                    frequency_hz=obs.frequency_hz, aoa_deg=obs.aoa_deg,
                    pulse_width_s=obs.pulse_width_s,
                    pdw_ids=json.dumps(obs.pdw_ids)))
                if truth_by_id is not None:
                    counts = Counter(truth_by_id[i] for i in obs.pdw_ids)
                    correct += max(counts.values())
                    eval_rows.append(dict(window=wi, train_id=obs.train_id,
                        truth_counts_eval=json.dumps(dict(counts), sort_keys=True),
                        purity_pct=100 * max(counts.values()) / obs.pulse_count))
                print(f" W{wi} B{tr.blob_id}: {obs.train_id} PRI={obs.pri_s / 1e-6:.4f} us "
                      f"matched={obs.pulse_count} -> provisional {eid}", flush=True)
            if truth_by_id is not None:
                row.update(train_purity_pct=100 * correct / assigned if assigned else None,
                           true_emitters_in_selected_pdws=len(set(truth_by_id.values())))
            window_rows.append(row)
            for p, blob in zip(pdws, blobs[selected]):
                train_id = result.pulse_train_assignments[p.pdw_id]
                eid = result.provisional_emitter_assignments[p.pdw_id]
                status = ("unexplained" if train_id is None else
                          "emitter_identity_pending" if eid is None else "provisional_emitter_association")
                pulse_rows.append(dict(window=wi, pdw_id=p.pdw_id, blob=int(blob),
                    pulse_train_id=train_id, provisional_emitter_id=eid, status=status))
            print(f"Window {wi}: {row}", flush=True)
            write_csv(out / "pulse_assignments.csv", pulse_rows,
                      ["window", "pdw_id", "blob", "pulse_train_id", "provisional_emitter_id", "status"])
            write_csv(out / "trains.csv", train_rows, ["window", "blob", "train_id"])
            write_csv(out / "train_evaluation.csv", eval_rows, ["window", "train_id", "truth_counts_eval", "purity_pct"])
            write_csv(out / "windows.csv", window_rows, ["window", "selected_pdws"])
    if not window_rows:
        ap.error("No available windows")
    (out / "identity_snapshot.json").write_text(json.dumps(tracker.identity_manager.snapshot(), indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    print(f"Saved {out}. Pulse-train membership and emitter IDs are provisional, not truth.")
    print("Fixed PRI only; harmonic ambiguity and frequency-agile emitter identity remain unresolved capabilities.")


if __name__ == "__main__":
    main()
