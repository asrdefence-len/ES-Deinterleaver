from dataclasses import asdict, replace

import numpy as np
import pytest

from testbed.blob_observation import BlobObservation, summarize_blobs, summarize_blob_trains
from testbed.blob_tracker import BlobEmitterTracker
from testbed.pulse_association_pipeline import AssociatedTrain
from testbed.pulse_train_observation import PulseTrainObservation


def obs(name, frequency=9.4e9, aoa=30., start=0., pw=.3e-6):
    return BlobObservation(name, name.split(":")[0], 0, start, start + .1,
        100, aoa, .5, frequency - 1e6, frequency, frequency + 1e6,
        pw - .05e-6, pw, pw + .05e-6)


def test_overlapping_rf_modes_retain_same_and_different_emitter_hypotheses():
    tracker = BlobEmitterTracker()
    observations = [obs("W0:B9", 10.2e9, start=0.),
                    obs("W0:B12", 10.8e9, start=.001),
                    obs("W0:B11", 10.6e9, start=.002)]
    for o in observations:
        tracker.process(o)
    counts = {len({dict(h.recent_assignments)[o.observation_id] for o in observations})
              for h in tracker.plausible_hypotheses()}
    assert counts == {1, 2}
    retained_counts = {len({dict(h.recent_assignments)[o.observation_id]
                            for o in observations}) for h in tracker.hypotheses}
    assert retained_counts == {1, 2, 3}
    assert tracker.association_options("W0:B11")["status"] == "AMBIGUOUS"
    assert any(len(t.rf_modes) == 3 for h in tracker.hypotheses for t in h.tracks)


def test_known_mode_continuity_keeps_identity_across_windows():
    tracker = BlobEmitterTracker()
    tracker.process(obs("W0:B1"))
    options = tracker.process(obs("W1:B9", start=.2))
    assert options == {"status": "PROVISIONAL", "candidate_track_ids": ["E1"]}


def test_different_bearings_do_not_share_track():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a", aoa=10))
    options = tracker.process(obs("b", aoa=50, start=.01))
    assert options["candidate_track_ids"] == ["E2"]


def test_circular_bearing_continuity():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a", aoa=179.8))
    assert "E1" in tracker.process(obs("b", aoa=-179.8, start=.2))["candidate_track_ids"]


def test_capacity_does_not_force_incompatible_association():
    tracker = BlobEmitterTracker(max_tracks=1)
    tracker.process(obs("a", aoa=10))
    assert tracker.process(obs("b", aoa=50, start=.01))["status"] == "UNRESOLVED_CAPACITY"


def test_rf_mode_memory_limit_does_not_force_a_new_emitter():
    tracker = BlobEmitterTracker(max_modes=2)
    for i, rf in enumerate((9.4e9, 10.2e9, 10.8e9)):
        tracker.process(obs(f"b{i}", frequency=rf, start=i * .001))
    assert any(len(h.tracks) == 1 for h in tracker.plausible_hypotheses())
    assert all(len(t.rf_modes) <= 2 for h in tracker.hypotheses for t in h.tracks)


def test_state_is_bounded_as_observations_arrive():
    tracker = BlobEmitterTracker(beam_width=4, max_tracks=3, max_modes=2,
        max_match_branches=2, history_limit=6, track_history_limit=3)
    for i in range(100):
        tracker.process(obs(f"W{i}:B0", start=i * .2))
        assert len(tracker.hypotheses) <= 4
        assert len(tracker._recent_ids) <= 6
        for h in tracker.hypotheses:
            assert len(h.tracks) <= 3
            assert len(h.recent_assignments) <= 6
            assert all(len(t.rf_modes) <= 2 and len(t.recent_blob_ids) <= 3 for t in h.tracks)


def test_adapter_retains_membership_and_exports_only_summary_features():
    x = np.array([[10., 9400., .3, 179.9], [12., 9402., .4, -179.9],
                  [14., 9500., .5, 10.], [16., 9501., .6, 11.]])
    names = ["ToA", "Frequency", "PulseWidth", "AoA"]
    labels = np.array([9, 9, 11, -1])
    original = labels.copy()
    observations = summarize_blobs(x, names, labels, "W0")
    assert np.array_equal(labels, original)
    assert [o.pulse_count for o in observations] == [2, 1]
    assert observations[0].frequency_median_hz == 9401e6
    assert observations[0].end_s == pytest.approx(12e-6)
    assert abs(abs(observations[0].aoa_deg) - 180) < .01
    assert not any(k in asdict(observations[0]) for k in
                   ("pdws", "pdw_ids", "toa_sequence", "truth", "emitter_id"))
    tracker = BlobEmitterTracker()
    before = [asdict(o) for o in observations]
    for o in observations:
        tracker.process(o)
    assert [asdict(o) for o in observations] == before


def test_duplicate_or_out_of_order_observations_fail():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a", start=.2))
    with pytest.raises(ValueError, match="Duplicate"):
        tracker.process(obs("a", start=.2))
    with pytest.raises(ValueError, match="start-time"):
        tracker.process(obs("b", start=.1))
    with pytest.raises(TypeError, match="BlobObservation"):
        tracker.process([1, 2, 3])


def test_expired_identity_is_not_reused():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a"))
    assert tracker.process(obs("b", start=2.))["candidate_track_ids"] == ["E2"]


def test_incorrect_or_nonfinite_features_are_rejected():
    with pytest.raises(ValueError):
        replace(obs("a"), aoa_deg=float("nan"))
    with pytest.raises(ValueError):
        replace(obs("a"), frequency_p05_hz=11e9)


def test_blob_updates_contract_covariance_and_export_dynamic_gates():
    tracker = BlobEmitterTracker()
    tracker.process(obs("W0:B1", start=0.))
    first = tracker.hypotheses[0].tracks[0]
    first_variance = first.aoa_variance_deg2
    tracker.process(obs("W1:B1", start=.2, aoa=30.05))
    continued = [h for h in tracker.hypotheses
                 if dict(h.recent_assignments).get("W1:B1") == "E1"]
    assert continued
    updated = continued[0].tracks[0]
    assert updated.aoa_variance_deg2 < first_variance + tracker.aoa_q * .1
    monitor = tracker.track_monitor(updated, updated.last_end_s + .1)
    assert monitor["covariance"]["aoa_variance_deg2"] > updated.aoa_variance_deg2
    assert monitor["gates"]["aoa_half_width_deg"] > 0
    assert 0 < monitor["existence_confidence"] < 1


def test_joint_nis_uses_chi_square_gate_and_records_likelihood():
    tracker = BlobEmitterTracker(gate_probability=.9973)
    tracker.process(obs("W0:B1", start=0.))
    tracker.process(obs("W1:B1", start=.2, aoa=30.1, pw=.305e-6))
    track = next(t for t in tracker.hypotheses[0].tracks if t.track_id == "E1")
    monitor = tracker.track_monitor(track)
    innovation = monitor["last_innovation"]
    assert innovation["degrees_of_freedom"] == 2
    assert innovation["nis"] < innovation["chi_square_gate"]
    assert innovation["log_likelihood_ratio"] > 0
    assert monitor["gates"]["probability"] == pytest.approx(.9973)


def test_missed_detection_multiplies_existence_odds_by_one_minus_pd():
    tracker = BlobEmitterTracker(detection_probability=.7,
        max_gap_s=10., archive_after_s=10.)
    tracker.process(obs("W0:B1", start=0.))
    track = tracker.hypotheses[0].tracks[0]
    probability_before = tracker._logistic(track.existence_log_odds)
    tracker.complete_window("W0", .1)
    tracker.complete_window("W1", .2)
    track = tracker.hypotheses[0].tracks[0]
    predicted = tracker.survival_probability * probability_before
    predicted_odds = predicted / (1. - predicted)
    assert np.exp(track.existence_log_odds) == pytest.approx(
        predicted_odds * (1. - tracker.detection_probability))


def test_survival_prediction_prevents_saturated_track_from_coasting_forever():
    tracker = BlobEmitterTracker(detection_probability=.65,
        survival_probability=.999, max_gap_s=10., archive_after_s=10.)
    tracker.process(obs("W0:B1", start=0.))
    saturated = replace(tracker.hypotheses[0].tracks[0], existence_log_odds=20.)
    tracker.hypotheses[0] = replace(tracker.hypotheses[0], tracks=(saturated,))
    tracker.complete_window("W0", .1)
    for i in range(1, 9):
        tracker.complete_window(f"W{i}", .1 * (i + 1))
    probability = tracker.track_monitor(tracker.hypotheses[0].tracks[0])[
        "track_confidence"]
    assert probability < .3


def test_blob_association_reports_relative_confidence_not_probability():
    tracker = BlobEmitterTracker()
    tracker.process(obs("W0:B1"))
    tracker.process(obs("W0:B2", frequency=10.2e9, start=.001))
    metrics = tracker.association_metrics("W0:B2")
    assert metrics["confidence_is_calibrated_probability"] is False
    assert sum(metrics["relative_association_confidence"].values()) == pytest.approx(1.)
    assert 0 < metrics["best_confidence"] <= 1


def test_tracker_state_contains_only_blob_summaries_not_pulse_lists():
    tracker = BlobEmitterTracker()
    tracker.process(obs("W0:B1"))
    snapshot = tracker.snapshot()
    text = repr(snapshot).lower()
    assert "pdw_ids" not in text
    assert "toa_sequence" not in text
    assert snapshot["association_metrics"]["W0:B1"][
        "confidence_is_calibrated_probability"] is False


def test_blob_train_adapter_exports_timing_without_member_references():
    x = np.array([[0., 9400., .30, 30.], [85., 9401., .31, 30.1],
                  [170., 9399., .29, 29.9]])
    labels = np.zeros(3, dtype=int)
    pulse_train = PulseTrainObservation(schema_version=1, train_id="W0:B0:T1",
        start_toa_s=0., end_toa_s=170e-6, pri_s=85e-6, pri_sigma_s=None,
        frequency_hz=9.4e9, frequency_sigma_hz=None, aoa_deg=30.,
        aoa_sigma_deg=None, pulse_width_s=.3e-6, pulse_width_sigma_s=None,
        pulse_count=3, pdw_ids=(10, 11, 12))
    train = AssociatedTrain(pulse_train, 0, 0., .1e-6, 3, 3)
    result = summarize_blob_trains(x, ["ToA", "Frequency", "PulseWidth", "AoA"],
        labels, np.array([10, 11, 12]), "W0", [train])
    assert len(result) == 1
    measurement = result[0]
    assert measurement.pri_s == pytest.approx(85e-6)
    assert measurement.phase_epoch_s == 0.
    assert measurement.timing_pulse_count == 3
    assert "pdw_ids" not in asdict(measurement)
    assert "toa_sequence" not in asdict(measurement)


def test_pri_and_phase_evidence_gate_blob_track_association():
    tracker = BlobEmitterTracker(timing_gate_s=5e-6)
    first = replace(obs("W0:B1", start=0.), train_id="W0:B1:T1",
        pri_s=85e-6, pri_sigma_s=.1e-6, phase_epoch_s=0.,
        timing_rms_residual_s=.2e-6, timing_pulse_count=100)
    compatible = replace(obs("W1:B1", start=.2), train_id="W1:B1:T1",
        pri_s=85.2e-6, pri_sigma_s=.1e-6, phase_epoch_s=2350 * 85.1e-6,
        timing_rms_residual_s=.2e-6, timing_pulse_count=100)
    incompatible = replace(obs("W2:B1", start=.4), train_id="W2:B1:T1",
        pri_s=100e-6, pri_sigma_s=.1e-6, phase_epoch_s=.4,
        timing_rms_residual_s=.2e-6, timing_pulse_count=100)
    tracker.process(first)
    assert "E1" in tracker.process(compatible)["candidate_track_ids"]
    assert "E1" not in tracker.process(incompatible)["candidate_track_ids"]


def test_correlated_blobs_keep_only_best_existence_evidence_per_window():
    tracker = BlobEmitterTracker()
    tracker.process(obs("W0:B1", start=0.))
    tracker.process(obs("W0:B2", frequency=10.2e9, start=.001))
    same_window = next(t for t in tracker.hypotheses[0].tracks if t.track_id == "E1")
    tracker.process(obs("W0:B3", frequency=10.2e9, start=.002, aoa=30.2))
    third = next(t for t in tracker.hypotheses[0].tracks if t.track_id == "E1")
    # Correlated same-window blobs retain only the strongest likelihood ratio,
    # rather than multiplying all of them as independent detections.
    assert third.existence_log_odds == pytest.approx(
        tracker._logit(tracker.birth_existence_confidence)
        + third.window_best_log_likelihood_ratio)
    tracker.process(obs("W1:B1", frequency=9.4e9, start=.2, aoa=30.01))
    next_window = next(t for t in tracker.hypotheses[0].tracks if t.track_id == "E1")
    assert next_window.existence_log_odds > third.existence_log_odds


def test_track_confidence_decays_on_probabilistic_missed_detection():
    tracker = BlobEmitterTracker(max_gap_s=10., archive_after_s=10.)
    tracker.process(obs("W0:B1", start=0.))
    track = tracker.hypotheses[0].tracks[0]
    fresh = tracker.track_monitor(track, track.last_end_s)
    tracker.complete_window("W0", .1)
    tracker.complete_window("W1", .2)
    track = tracker.hypotheses[0].tracks[0]
    coasted = tracker.track_monitor(track, .2)
    assert coasted["track_confidence"] < fresh["track_confidence"]
    assert coasted["existence_confidence"] == coasted["track_confidence"]


def test_new_measurement_restores_operational_track_confidence():
    tracker = BlobEmitterTracker(max_gap_s=10., archive_after_s=10.)
    tracker.process(obs("W0:B1", start=0.))
    tracker.complete_window("W0", .1)
    tracker.complete_window("W1", .2)
    track = tracker.hypotheses[0].tracks[0]
    coasted = tracker.track_monitor(track, .2)
    tracker.process(obs("W1:B1", start=.6))
    updated = next(t for t in tracker.hypotheses[0].tracks if t.track_id == "E1")
    restored = tracker.track_monitor(updated, updated.last_end_s)
    assert restored["track_confidence"] > coasted["track_confidence"]
    assert restored["last_innovation"]["nis"] is not None


def test_phase_mismatch_is_soft_but_pri_mismatch_remains_gated():
    tracker = BlobEmitterTracker(timing_gate_s=5e-6)
    first = replace(obs("W0:B1", start=0.), train_id="W0:B1:T1",
        pri_s=85e-6, pri_sigma_s=.1e-6, phase_epoch_s=0.,
        timing_rms_residual_s=.2e-6, timing_pulse_count=100)
    shifted = replace(obs("W1:B1", start=.2), train_id="W1:B1:T1",
        pri_s=85.1e-6, pri_sigma_s=.1e-6, phase_epoch_s=.2,
        timing_rms_residual_s=.2e-6, timing_pulse_count=100)
    tracker.process(first)
    assert "E1" in tracker.process(shifted)["candidate_track_ids"]


def test_lost_track_is_archived_and_removed_from_active_association():
    tracker = BlobEmitterTracker(max_gap_s=10., detection_probability=.9,
        archive_after_s=.5, archive_confidence=.3)
    tracker.process(obs("W0:B1", start=0.))
    tracker.complete_window("W0", .1)
    tracker.complete_window("W1", 1.)
    snapshot = tracker.snapshot()
    assert not snapshot["hypotheses"][0]["active_tracks"]
    assert snapshot["archived_track_count"] == 1
    assert snapshot["archived_tracks"][0]["track_id"] == "E1"
    assert snapshot["archived_tracks"][0]["reason"] == "LOST_CONFIDENCE"


def test_expired_track_archives_even_if_confidence_is_high():
    tracker = BlobEmitterTracker(max_gap_s=.3, archive_after_s=10.)
    tracker.process(obs("W0:B1", start=0.))
    tracker.coast_to(.5)
    assert tracker.snapshot()["archived_tracks"][0]["reason"] == "MAX_GAP"


def test_established_track_wins_over_compatible_young_fragment():
    tracker = BlobEmitterTracker()
    for i in range(8):
        tracker.process(obs(f"W{i}:B1", start=i * .02))
    established = tracker.hypotheses[0].tracks[0]
    young = tracker._new_track(obs("seed", start=.15), "YOUNG")
    measurement = obs("candidate", start=.18, aoa=30.05)
    assert tracker._compatibility(measurement, established)[0] < (
        tracker._compatibility(measurement, young)[0])


def test_tentative_compatible_fragment_consolidates_into_mature_parent():
    tracker = BlobEmitterTracker(consolidation_min_updates=3,
        consolidation_parent_updates=10)
    parent = replace(tracker._new_track(obs("parent"), "PARENT"), updates=12)
    child = replace(tracker._new_track(obs("child", aoa=30.1, pw=.305e-6), "CHILD"),
                    updates=3)
    hypothesis, aliases = tracker._consolidate_hypothesis(
        type(tracker.hypotheses[0])((parent, child),
            (("parent", "PARENT"), ("child", "CHILD")), 0.))
    assert len(hypothesis.tracks) == 1
    assert hypothesis.tracks[0].track_id == "PARENT"
    assert dict(hypothesis.recent_assignments)["child"] == "PARENT"
    assert aliases == [("CHILD", "PARENT", 3)]


def test_distant_fragment_does_not_consolidate():
    tracker = BlobEmitterTracker()
    parent = replace(tracker._new_track(obs("parent", aoa=10.), "PARENT"), updates=12)
    child = replace(tracker._new_track(obs("child", aoa=50.), "CHILD"), updates=3)
    hypothesis, aliases = tracker._consolidate_hypothesis(
        type(tracker.hypotheses[0])((parent, child), (), 0.))
    assert len(hypothesis.tracks) == 2
    assert not aliases


@pytest.mark.parametrize("first_pri, second_pri", [(85e-6, 170.4e-6), (170e-6, 85.2e-6)])
def test_harmonic_nis_uses_selected_variance_and_update_uses_same_projection(first_pri, second_pri):
    tracker = BlobEmitterTracker()
    first = replace(obs("W0:B1"), train_id="a", pri_s=first_pri,
        pri_sigma_s=.1e-6, phase_epoch_s=0., timing_rms_residual_s=.1e-6,
        timing_pulse_count=100)
    second = replace(first, observation_id="W1:B1", window_id="W1",
                     start_s=.2, end_s=.3, pri_s=second_pri)
    track = tracker._new_track(first, "E1")
    cost, mode, stats = tracker._compatibility(second, track)
    timing = stats["timing"]
    assert stats["nis"] == pytest.approx(timing["pri_z"] ** 2)
    axis = stats["axes"]["pri_base_s"]
    assert axis["S"] == pytest.approx(timing["innovation_variance_s2"], rel=1e-12, abs=0)
    updated = tracker._updated_track(track, second, mode, stats)
    assert updated.pri_s == pytest.approx(track.pri_s + axis["K"] *
        timing["innovation_s"] * timing["track_harmonic"], rel=1e-12, abs=0)


def test_monitor_records_rejections_misses_and_does_not_change_inference():
    events = []
    monitored = BlobEmitterTracker(consistency_sink=events.append, max_gap_s=10.)
    plain = BlobEmitterTracker(max_gap_s=10.)
    for tracker in (monitored, plain):
        tracker.process(obs("W0:B1"))
        tracker.process(obs("W1:B1", start=.2, aoa=30.1))
        tracker.process(obs("W1:B2", start=.21, aoa=90.))
        tracker.complete_window("W1", .31)
        tracker.complete_window("W2", .4)
    assert monitored.snapshot() == plain.snapshot()
    assert {e["outcome"] for e in events} >= {"accepted", "rejected", "missed"}
    for event in events:
        assert 0 < event["existence_before"] < 1
        assert 0 < event["existence_after"] < 1
        if event["outcome"] == "missed":
            assert event["nis"] is None and event["axes"] == {}
            assert event["existence_after"] < event["existence_before"]
        else:
            expected = 0.
            for component in event["axes"].values():
                assert component["S"] == component["P_predicted"] + component["R"]
                assert component["K"] == component["P_predicted"] / component["S"]
                expected += component["innovation"] ** 2 / component["S"]
            assert event["nis"] == pytest.approx(expected)
