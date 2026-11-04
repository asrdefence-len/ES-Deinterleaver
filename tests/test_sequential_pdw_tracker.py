from dataclasses import asdict

import h5py
import numpy as np
import pytest

from sequential_pdw_tracker.evaluation import evaluate
from sequential_pdw_tracker.io import (
    inspect_dataset,
    iter_pdws,
    iter_truth_isolated_pdws,
)
from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.tracker import SequentialPDWTracker, TrackerConfig


def pdw(index, toa_us, frequency_mhz, pulse_width_us, aoa_deg):
    return PDW(index, toa_us * 1e-6, frequency_mhz * 1e6,
               pulse_width_us * 1e-6, aoa_deg, -70.0)


def test_hdf5_reader_converts_units_and_keeps_truth_separate(tmp_path):
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.array([
            [10.0, 1000.0, 2.0, 359.0, -80.0],
            [11.0, 1010.0, 2.1, 1.0, -79.0],
        ]))
        handle.create_dataset("labels", data=np.array([[7], [8]]))
        metadata = handle.create_group("metadata")
        metadata.create_dataset("feature_names", data=np.array(
            [b"ToA", b"Frequency", b"PulseWidth", b"AoA", b"Amplitude"]))
    info = inspect_dataset(path)
    rows = list(iter_pdws(path))
    assert info.labels_present and info.total_pdws == 2
    assert rows[0][0].toa_s == pytest.approx(10e-6)
    assert rows[0][0].frequency_hz == 1e9
    assert rows[0][1] == 7
    assert "truth" not in asdict(rows[0][0])


def test_truth_isolation_preserves_source_ids_and_never_enters_pdw(tmp_path):
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.array([
            [10.0, 1000.0, 2.0, 10.0, -80.0],
            [11.0, 2000.0, 3.0, 20.0, -79.0],
            [12.0, 1100.0, 2.0, 10.0, -78.0],
            [13.0, 1200.0, 2.0, 10.0, -77.0],
        ]))
        handle.create_dataset("labels", data=np.array([[46], [7], [46], [46]]))
        metadata = handle.create_group("metadata")
        metadata.create_dataset("feature_names", data=np.array(
            [b"ToA", b"Frequency", b"PulseWidth", b"AoA", b"Amplitude"]))
    rows = list(iter_truth_isolated_pdws(path, 46, start=1, limit=2))
    assert [item.pdw_id for item, _ in rows] == [2, 3]
    assert [label for _, label in rows] == [46, 46]
    assert all("truth" not in asdict(item) for item, _ in rows)


def test_probabilities_are_normalized_and_include_new_and_clutter():
    tracker = SequentialPDWTracker()
    first = tracker.process(pdw(0, 0, 1000, 1, 10))
    second = tracker.process(pdw(1, 100, 1000, 1, 10))
    assert first.outcome == "birth"
    assert set(("new", "clutter")) <= set(second.posterior)
    assert abs(sum(second.posterior.values()) - 1.0) < 1e-12


def test_interleaved_emitters_and_frequency_hop_remain_separate():
    tracker = SequentialPDWTracker()
    sequence = [
        pdw(0, 0, 1000, 1.0, 10),
        pdw(1, 20, 5000, 3.0, 100),
        pdw(2, 100, 1100, 1.0, 10.2),
        pdw(3, 120, 5000, 3.0, 99.8),
        pdw(4, 200, 1000, 1.0, 10.1),
        pdw(5, 220, 5000, 3.0, 100.1),
    ]
    decisions = [tracker.process(item) for item in sequence]
    assert decisions[0].track_id == decisions[2].track_id == decisions[4].track_id
    assert decisions[1].track_id == decisions[3].track_id == decisions[5].track_id
    assert decisions[0].track_id != decisions[1].track_id
    first = next(snapshot for snapshot in tracker.snapshots()
                 if snapshot.track_id == decisions[0].track_id)
    assert len(first.rf_modes_hz) == 2


def test_default_hop_hypothesis_outranks_birth_when_other_features_agree():
    tracker = SequentialPDWTracker()
    first = tracker.process(pdw(0, 0, 1000, 1.0, 10.0))
    hopped = tracker.process(pdw(1, 100, 1100, 1.0, 10.0))
    assert first.outcome == "birth"
    assert hopped.outcome == "assigned"
    assert hopped.track_id == first.track_id
    assert hopped.posterior[first.track_id] > hopped.posterior["new"]
    assert len(tracker.snapshots()[0].rf_modes_hz) == 2


def test_unseen_mode_mass_does_not_shrink_as_modes_are_learned():
    tracker = SequentialPDWTracker()
    decisions = [tracker.process(pdw(
        index, index * 100, 1000 + index * 100, 1.0, 10.0,
    )) for index in range(8)]
    assert decisions[0].outcome == "birth"
    assert all(decision.outcome == "assigned" for decision in decisions[1:])
    assert len({decision.track_id for decision in decisions}) == 1
    assert len(tracker.snapshots()[0].rf_modes_hz) == 8


def test_pulse_width_modes_are_learned_without_splitting_identity():
    tracker = SequentialPDWTracker()
    widths = (0.10, 0.27, 0.43) * 4
    decisions = [tracker.process(pdw(
        index, index * 100, 1000, width, 10.0,
    )) for index, width in enumerate(widths)]
    assert decisions[0].outcome == "birth"
    assert all(decision.outcome == "assigned" for decision in decisions[1:])
    assert len({decision.track_id for decision in decisions}) == 1
    assert len(tracker.snapshots()[0].pulse_width_modes_s) == 3


def test_timing_stays_neutral_until_multiple_gaps_support_pri():
    tracker = SequentialPDWTracker()
    decisions = [
        tracker.process(pdw(0, 0, 1000, 1.0, 10.0)),
        tracker.process(pdw(1, 100, 1000, 1.0, 10.0)),
        # A single 100 us gap must not make this later valid pulse impossible.
        tracker.process(pdw(2, 350, 1000, 1.0, 10.0)),
    ]
    assert [decision.outcome for decision in decisions] == ["birth", "assigned", "assigned"]
    assert len({decision.track_id for decision in decisions}) == 1
    assert tracker.snapshots()[0].pri_support == 2


def test_evaluation_keeps_missing_tracks_and_all_fragments_visible():
    tracker = SequentialPDWTracker(TrackerConfig(max_tracks=1))
    decisions = [
        tracker.process(pdw(0, 0, 1000, 1, 10)),
        tracker.process(pdw(1, 10, 9000, 9, 170)),
    ]
    result = evaluate(decisions, [46, 38])
    assert result["truth_summary"]["46"]["coverage"] == 1.0
    assert result["truth_summary"]["38"]["coverage"] == 0.0
    assert result["truth_summary"]["38"]["primary_track"] is None
    assert result["truth_summary"]["38"]["fragments"] == {}
