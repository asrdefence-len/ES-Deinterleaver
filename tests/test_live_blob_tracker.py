import json
import sys

import h5py
import numpy as np

from run_live_blob_tracker import main


def make_stream(path):
    rows, labels = [], []
    for pulse in range(160):
        t = pulse * 5_000.  # raw microseconds
        rows.append([t, 9400. + (pulse % 3), .30, 10. + .05 * (pulse % 2)])
        labels.append(1)
        rows.append([t + 2_000., 10_200. + (pulse % 3), .45, 40. + .05 * (pulse % 2)])
        labels.append(2)
    order = np.argsort(np.asarray(rows)[:, 0], kind="stable")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.asarray(rows)[order])
        handle.create_dataset("labels", data=np.asarray(labels)[order])
        metadata = handle.create_group("metadata")
        metadata.create_dataset("feature_names",
            data=np.array([b"ToA", b"Frequency", b"PulseWidth", b"AoA"]))


def test_headless_live_runner_writes_track_history_and_summary(tmp_path, monkeypatch):
    recording, output = tmp_path / "stream.h5", tmp_path / "results"
    make_stream(recording)
    monkeypatch.setattr(sys, "argv", ["run_live_blob_tracker.py", str(recording),
        "--window-size", "80", "--playback-rate", "0", "--no-live",
        "--output", str(output)])
    main()
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["pdws"] == 320
    assert summary["observations"] > 0
    assert summary["unique_track_ids"] >= 2
    assert (output / "live_track_history.csv").exists()
    assert (output / "tracker_snapshot.json").exists()
