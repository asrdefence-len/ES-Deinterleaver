from collections import Counter
import json

from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.model2_birth import (
    Model2AutomaticBirthTracker,
    Model2BirthConfig,
)


def pdw(index: int, emitter: str) -> PDW:
    emitter_index = index // 2
    if emitter == "left":
        frequency_mhz = (1000.0, 1200.0, 1400.0)[emitter_index % 3]
        pulse_width_us = (0.10, 0.27)[emitter_index % 2]
        aoa_deg = 20.0
    else:
        frequency_mhz = (900.0, 1100.0)[emitter_index % 2]
        pulse_width_us = 0.40
        aoa_deg = 35.0
    return PDW(
        pdw_id=index,
        toa_s=index * 10e-6,
        frequency_hz=frequency_mhz * 1e6,
        pulse_width_s=pulse_width_us * 1e-6,
        aoa_deg=aoa_deg,
        amplitude_db=-70.0,
    )


def test_truth_blind_birth_creates_and_confirms_two_pure_hmm_tracks():
    tracker = Model2AutomaticBirthTracker(Model2BirthConfig(
        confirmation_pulses=12,
        threshold_calibration_window=6,
        enable_timing_after_confirmation=False,
    ))
    assignments = []
    for index in range(80):
        emitter = "left" if index % 2 == 0 else "right"
        decision = tracker.process(pdw(index, emitter))
        assignments.append((decision.track_id, emitter))

    snapshots = tracker.snapshots()
    assert len(snapshots) == 2
    assert all(row["status"] == "confirmed" for row in snapshots)
    composition = {
        row["track_id"]: Counter(
            emitter for track_id, emitter in assignments
            if track_id == row["track_id"]
        )
        for row in snapshots
    }
    assert {counts.most_common(1)[0][0] for counts in composition.values()} == {
        "left", "right",
    }
    assert all(len(counts) == 1 for counts in composition.values())
    assert sorted(len(row["model"]["rf_states"]) for row in snapshots) == [2, 3]


def test_rejected_near_aoa_pulse_does_not_create_duplicate_track():
    tracker = Model2AutomaticBirthTracker(Model2BirthConfig(
        confirmation_pulses=8,
        threshold_calibration_window=4,
        enable_timing_after_confirmation=False,
    ))
    for index in range(12):
        tracker.process(pdw(index, "left"))
    before = tracker.track_count
    incompatible = PDW(
        pdw_id=100,
        toa_s=1e-3,
        frequency_hz=5e9,
        pulse_width_s=2e-6,
        aoa_deg=20.0,
        amplitude_db=-70.0,
    )
    decision = tracker.process(incompatible)
    assert decision.outcome == "rejected"
    assert decision.rejection_reason == "confirmed_aoa_birth_suppression"
    assert tracker.track_count == before


def test_external_truth_cannot_change_birth_or_assignment_decisions():
    config = Model2BirthConfig(
        confirmation_pulses=10,
        threshold_calibration_window=5,
        enable_timing_after_confirmation=False,
    )
    left = Model2AutomaticBirthTracker(config)
    right = Model2AutomaticBirthTracker(config)
    sequence = [pdw(index, "left" if index % 2 == 0 else "right")
                for index in range(50)]
    left_decisions = [left.process(item) for item in sequence]
    relabelled_truth = ["unrelated" for _item in sequence]
    right_decisions = [right.process(item) for item, _truth
                       in zip(sequence, relabelled_truth)]
    assert left_decisions == right_decisions
    assert left.snapshots() == right.snapshots()


def test_safe_confirmed_pruning_preserves_every_assignment_decision():
    common = dict(
        confirmation_pulses=10,
        threshold_calibration_window=5,
        enable_timing_after_confirmation=False,
    )
    exact = Model2AutomaticBirthTracker(Model2BirthConfig(
        **common, enable_safe_confirmed_pruning=False,
    ))
    pruned = Model2AutomaticBirthTracker(Model2BirthConfig(
        **common, enable_safe_confirmed_pruning=True,
    ))
    sequence = [pdw(index, "left" if index % 2 == 0 else "right")
                for index in range(100)]
    exact_decisions = [exact.process(item) for item in sequence]
    pruned_decisions = [pruned.process(item) for item in sequence]
    assert [item.track_id for item in exact_decisions] == [
        item.track_id for item in pruned_decisions
    ]
    assert [item.outcome for item in exact_decisions] == [
        item.outcome for item in pruned_decisions
    ]
    assert exact.snapshots() == pruned.snapshots()


def staged_pdw(
    index: int,
    *,
    frequency_mhz: float,
    pulse_width_us: float,
) -> PDW:
    return PDW(
        pdw_id=index,
        toa_s=index * 10e-6,
        frequency_hz=frequency_mhz * 1e6,
        pulse_width_s=pulse_width_us * 1e-6,
        aoa_deg=20.0,
        amplitude_db=-70.0,
    )


def staged_config(**overrides) -> Model2BirthConfig:
    values = dict(
        confirmation_pulses=12,
        threshold_calibration_window=4,
        enable_timing_after_confirmation=False,
        enable_staged_birth_validation=True,
        tentative_seed_pulses=4,
        tentative_freeze_pulses=8,
        validation_reference_window=3,
        validation_min_pass_fraction=0.75,
    )
    values.update(overrides)
    return Model2BirthConfig(**values)


def test_staged_tentative_features_split_same_aoa_emitters_and_confirm_both():
    tracker = Model2AutomaticBirthTracker(staged_config())
    labelled = []
    for index in range(4):
        labelled.append((
            staged_pdw(index, frequency_mhz=1000, pulse_width_us=0.10),
            "left",
        ))
    for index in range(4, 28):
        right = index % 2 == 0
        labelled.append((
            staged_pdw(
                index,
                frequency_mhz=1400 if right else 1000,
                pulse_width_us=0.40 if right else 0.10,
            ),
            "right" if right else "left",
        ))
    assignments = [(tracker.process(item).track_id, label)
                   for item, label in labelled]
    snapshots = tracker.snapshots()
    assert len(snapshots) == 2
    assert all(row["status"] == "confirmed" for row in snapshots)
    for row in snapshots:
        labels = {
            label for track_id, label in assignments
            if track_id == row["track_id"]
        }
        assert len(labels) == 1


def test_staged_validation_keeps_inconclusive_candidate_for_another_block():
    tracker = Model2AutomaticBirthTracker(staged_config())
    outcomes = []
    for index in range(8):
        outcomes.append(tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=0.10,
        )).outcome)
    for index in range(8, 12):
        outcomes.append(tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=0.50,
        )).outcome)
    snapshot = tracker.snapshots()[0]
    assert outcomes[-1] == "tentative_update"
    assert snapshot["status"] == "tentative"
    assert snapshot["birth_validation"]["pass_fraction"] == 0.0
    assert tracker.track_count == 1

    retry_outcomes = []
    for index in range(12, 16):
        retry_outcomes.append(tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=0.10,
        )).outcome)
    assert "confirmed" in retry_outcomes
    assert tracker.snapshots()[0]["status"] == "confirmed"


def test_staged_validation_does_not_promote_on_pathological_threshold():
    tracker = Model2AutomaticBirthTracker(staged_config(
        confirmation_pulses=20,
        threshold_calibration_window=8,
        tentative_freeze_pulses=10,
        validation_reference_window=4,
        validation_min_pass_fraction=0.90,
    ))
    for index in range(10):
        tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=0.10,
        ))
    for index in range(10, 20):
        width = 0.50 if index == 10 else 0.10
        tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=width,
        ))
    validation = tracker.snapshots()[0]["birth_validation"]
    assert validation["pass_fraction"] == 0.9
    assert validation["pathological_threshold"]
    assert validation["robust_threshold"] > validation["raw_threshold"]
    assert tracker.snapshots()[0]["threshold"] is None
    assert tracker.snapshots()[0]["status"] == "tentative"

    decision = tracker.process(staged_pdw(
        20, frequency_mhz=1000, pulse_width_us=0.10,
    ))
    assert decision.outcome == "confirmed"
    assert tracker.snapshots()[0]["status"] == "confirmed"


def test_staged_validation_rejects_pathological_training_reference():
    tracker = Model2AutomaticBirthTracker(staged_config(
        confirmation_pulses=20,
        threshold_calibration_window=8,
        tentative_freeze_pulses=10,
        validation_reference_window=4,
        validation_min_pass_fraction=0.90,
    ))
    for index in range(10):
        width = 0.50 if index == 9 else 0.10
        tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=width,
        ))
    for index in range(10, 20):
        tracker.process(staged_pdw(
            index, frequency_mhz=1000, pulse_width_us=0.10,
        ))
    validation = tracker.snapshots()[0]["birth_validation"]
    assert any(validation["reference_pathological_components"])
    assert tracker.snapshots()[0]["status"] == "rejected_birth"
    json.dumps(tracker.snapshots(), allow_nan=False)
