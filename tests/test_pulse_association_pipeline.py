import numpy as np
import pytest

from testbed.emitter_track_manager import EmitterTrackManager
from testbed.pulse_association_pipeline import (
    FixedPRIPulseAssociation, PulseAssociationTrackingPipeline, PulseMeasurement,
)
from testbed.pulse_train_observation import PulseTrainObservation


def train(first_id=1, start_s=15.6, pri_us=85.1, count=120, aoa=45.):
    rng = np.random.default_rng(123)
    return [PulseMeasurement(first_id + i,
        start_s + (i * pri_us + rng.normal(0, .02)) * 1e-6,
        9.4e9, 8e-6, aoa) for i in range(count)]


def test_fixed_train_adapter_preserves_units_and_pdw_references():
    pdws = train()
    item = FixedPRIPulseAssociation()
    trains, unexplained, assignments = item.process(pdws, np.zeros(len(pdws), int), "W0")
    assert len(trains) == 1
    obs = trains[0].observation
    assert obs.pri_s == pytest.approx(85.1e-6, abs=.05e-6)
    assert obs.start_toa_s == pdws[0].toa_s
    assert obs.frequency_hz == 9.4e9
    assert obs.pulse_width_s == 8e-6
    assert set(obs.pdw_ids) == {p.pdw_id for p in pdws}
    assert not unexplained
    assert all(v == obs.train_id for v in assignments.values())
    assert obs.association_confidence is None
    assert obs.missed_pulse_count is None
    assert not hasattr(obs, "emitter_id")


def test_unexplained_pdws_remain_in_accounting():
    pdws = train()
    pdws.append(PulseMeasurement(999, 15.6 + (20 * 85.1 + 31) * 1e-6, 9.4e9, 8e-6, 45))
    result = PulseAssociationTrackingPipeline().process_window(pdws, np.zeros(len(pdws), int), "W0")
    assert 999 in result.unexplained_pdw_ids
    assert result.pulse_train_assignments[999] is None
    assert result.provisional_emitter_assignments[999] is None
    members = [i for tr in result.trains for i in tr.observation.pdw_ids]
    assert len(members) == len(set(members))
    assert set(members) | set(result.unexplained_pdw_ids) == {p.pdw_id for p in pdws}


def test_tracker_identity_persists_across_disjoint_windows_and_pri_change():
    pipeline = PulseAssociationTrackingPipeline()
    a = train()
    b = train(first_id=1001, start_s=15.63, pri_us=70.)
    first = pipeline.process_window(a, np.zeros(len(a), int), "W0")
    second = pipeline.process_window(b, np.zeros(len(b), int), "W1")
    assert set(first.provisional_emitter_assignments.values()) == {"E1"}
    assert set(second.provisional_emitter_assignments.values()) == {"E1"}
    assert len(pipeline.identity_manager.tracks) == 1
    assert len(pipeline.identity_manager.tracks["E1"].observations) == 2


def test_overlapping_blob_trains_are_not_forced_into_one_emitter():
    a = train()
    b = train(first_id=1001, start_s=15.60002)
    pdws = a + b
    result = PulseAssociationTrackingPipeline().process_window(
        pdws, np.r_[np.zeros(len(a), int), np.ones(len(b), int)], "W0")
    assert len(result.trains) == 2
    assert set(result.train_emitter_assignments.values()) == {"E1", "E2"}


def test_replayed_pdw_and_out_of_order_windows_are_rejected():
    pdws = train()
    pipeline = PulseAssociationTrackingPipeline()
    pipeline.process_window(pdws, np.zeros(len(pdws), int), "W0")
    with pytest.raises(ValueError, match="already processed"):
        pipeline.process_window(pdws, np.zeros(len(pdws), int), "W1")
    old = train(first_id=2001, start_s=15.5)
    with pytest.raises(ValueError, match="chronological"):
        pipeline.process_window(old, np.zeros(len(old), int), "old")


def test_negative_group_labels_and_sparse_data_are_preserved():
    pdws = train(count=10)
    result = PulseAssociationTrackingPipeline().process_window(pdws, np.full(10, -1), "W0")
    assert not result.trains
    assert len(result.unexplained_pdw_ids) == 10
    assert set(result.pulse_train_assignments.values()) == {None}


def test_shuffled_input_preserves_membership():
    pdws = train()
    stage = FixedPRIPulseAssociation()
    a, _, _ = stage.process(pdws, np.zeros(len(pdws), int), "W0")
    b, _, _ = stage.process(pdws[::-1], np.zeros(len(pdws), int), "W1")
    assert set(a[0].observation.pdw_ids) == set(b[0].observation.pdw_ids)


def test_identity_manager_uses_circular_bearing_distance():
    def obs(name, start, aoa):
        return PulseTrainObservation(schema_version=1, train_id=name,
            start_toa_s=start, end_toa_s=start + .01, pri_s=85e-6, pri_sigma_s=None,
            frequency_hz=9.4e9, frequency_sigma_hz=None, aoa_deg=aoa, aoa_sigma_deg=None,
            pulse_width_s=8e-6, pulse_width_sigma_s=None, pulse_count=1, pdw_ids=(1,))
    manager = EmitterTrackManager(aoa_gate_deg=3.)
    assert manager.process(obs("a", 0., 179.)) == "E1"
    assert manager.process(obs("b", .02, -179.)) == "E1"


def test_invalid_inputs_fail_before_association():
    pdws = train(count=10)
    stage = FixedPRIPulseAssociation()
    with pytest.raises(ValueError, match="integer blob"):
        stage.process(pdws, np.zeros(9, int), "W0")
    with pytest.raises(ValueError, match="Duplicate PDW"):
        stage.process(pdws + pdws, np.zeros(20, int), "W0")
    with pytest.raises(ValueError, match="finite measurements"):
        stage.process([PulseMeasurement(1, float("nan"), 9e9, 8e-6, 0)], np.array([0]), "W0")
