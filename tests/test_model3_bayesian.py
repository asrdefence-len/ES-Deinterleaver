import math

import pytest

from sequential_pdw_tracker.hmm_emitter import HMMModel1Config
from sequential_pdw_tracker.hmm_emitter_model2 import HMMModel2Config
from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.model3_bayesian import (
    BayesianModel3Config,
    BayesianModel3Tracker,
)


def pdw(index: int, emitter: str = "left") -> PDW:
    if emitter == "left":
        frequency_mhz = (1000.0, 1200.0)[index % 2]
        pulse_width_us = (0.10, 0.27)[index % 2]
        aoa_deg = 20.0
    else:
        frequency_mhz = (1400.0, 1600.0)[index % 2]
        pulse_width_us = 0.43
        aoa_deg = 60.0
    return PDW(
        pdw_id=index,
        toa_s=index * 100e-6,
        frequency_hz=frequency_mhz * 1e6,
        pulse_width_s=pulse_width_us * 1e-6,
        aoa_deg=aoa_deg,
        amplitude_db=-70.0,
    )


def config(**overrides) -> BayesianModel3Config:
    values = dict(
        clutter_rate_hz=20.0,
        birth_event_rate_hz=5.0,
        track_rate_prior_shape=2.0,
        track_rate_prior_exposure_s=0.002,
        max_global_hypotheses=128,
    )
    values.update(overrides)
    return BayesianModel3Config(**values)


def test_every_event_returns_a_normalized_posterior_not_a_forced_assignment():
    tracker = BayesianModel3Tracker(config())
    update = tracker.process(pdw(0))
    assert sum(update.association_probabilities.values()) == pytest.approx(1.0)
    assert set(update.association_probabilities) == {"clutter", "birth:B0"}
    assert 0.0 < update.association_probabilities["birth:B0"] < 1.0
    assert update.track_existence_probabilities["B0"] == pytest.approx(
        update.association_probabilities["birth:B0"]
    )
    assert tracker.summary()["hard_association_feedback"] is False
    assert tracker.summary()["confirmation_threshold"] is None
    assert tracker.summary()["emitter_model_config"]["model1"][
        "max_rf_states"
    ] == 64


def test_repeated_coherent_pulses_raise_birth_track_existence():
    tracker = BayesianModel3Tracker(config())
    first = tracker.process(pdw(0))
    initial = first.track_existence_probabilities["B0"]
    latest = first
    for index in range(1, 12):
        latest = tracker.process(pdw(index))
    assert latest.track_existence_probabilities["B0"] > initial
    assert latest.track_existence_probabilities["B0"] > 0.95
    assert latest.map_action_for_reporting == "track:B0"
    assert all(0.0 <= value <= 1.0 for value in latest.track_existence_probabilities.values())


def test_second_incompatible_emitter_retains_a_separate_birth_hypothesis():
    tracker = BayesianModel3Tracker(config(max_global_hypotheses=256))
    for index in range(8):
        tracker.process(pdw(index, "left"))
    update = None
    for index in range(8, 20):
        update = tracker.process(pdw(index, "right"))
    assert update is not None
    right_births = [
        track_id for track_id, probability in update.track_existence_probabilities.items()
        if track_id != "B0" and probability > 0.5
    ]
    assert right_births
    assert update.track_existence_probabilities["B0"] > 0.9


def test_hypothesis_pruning_is_reported_and_posterior_is_renormalized():
    tracker = BayesianModel3Tracker(config(max_global_hypotheses=4))
    update = None
    for index in range(6):
        update = tracker.process(pdw(index, "left" if index % 2 == 0 else "right"))
    assert update is not None
    assert update.retained_global_hypotheses == 4
    assert 0.0 < update.retained_posterior_mass <= 1.0
    assert sum(update.association_probabilities.values()) == pytest.approx(1.0)
    assert math.isfinite(sum(update.track_existence_probabilities.values()))


def test_shared_state_dag_reuses_likelihoods_and_track_updates():
    tracker = BayesianModel3Tracker(config())
    for index in range(12):
        update = tracker.process(pdw(index))
    metrics = tracker.summary()["sharing_metrics"]
    assert metrics["track_mark_evaluations"] < metrics["track_mark_requests"]
    assert metrics["track_state_updates"] < metrics["track_update_requests"]
    assert metrics["birth_state_constructions"] <= 12
    assert (
        metrics["retained_child_materializations"]
        < metrics["global_child_proposals"]
    )
    assert all(
        0.0 <= probability <= 1.0
        for probability in update.track_existence_probabilities.values()
    )


def test_pruning_precedes_conditional_hmm_materialization():
    tracker = BayesianModel3Tracker(config(max_global_hypotheses=4))
    for index in range(12):
        tracker.process(pdw(index, "left" if index < 6 else "right"))
    metrics = tracker.summary()["sharing_metrics"]
    assert metrics["retained_child_materializations"] <= 12 * 4
    assert metrics["track_state_updates"] < metrics["track_mark_evaluations"]
    assert metrics["birth_state_constructions"] < 12


def test_local_waveform_posterior_has_an_independent_bounded_budget():
    tracker = BayesianModel3Tracker(config(
        max_global_hypotheses=32,
        max_local_waveform_hypotheses=2,
    ))
    measured_local_mass = []
    for index in range(16):
        update = tracker.process(pdw(index))
        if update.minimum_local_waveform_retained_mass is not None:
            measured_local_mass.append(
                update.minimum_local_waveform_retained_mass
            )

    assert measured_local_mass
    assert all(0.0 < value <= 1.0 for value in measured_local_mass)
    assert all(
        row["local_waveform_hypotheses"] <= 2
        for row in tracker.snapshots()
    )
    metrics = tracker.summary()["sharing_metrics"]
    assert metrics["local_waveform_updates"] > 0
    assert metrics["local_waveform_materializations"] <= (
        2 * metrics["local_waveform_updates"]
    )


def test_current_event_association_marginalizes_local_state_birth_before_pruning():
    narrow = BayesianModel3Tracker(config(
        max_local_waveform_hypotheses=1,
    ))
    wide = BayesianModel3Tracker(config(
        max_local_waveform_hypotheses=16,
    ))

    first_narrow = narrow.process(pdw(0))
    first_wide = wide.process(pdw(0))
    assert first_narrow.association_probabilities == pytest.approx(
        first_wide.association_probabilities
    )

    second_narrow = narrow.process(pdw(1))
    second_wide = wide.process(pdw(1))
    assert second_narrow.association_probabilities == pytest.approx(
        second_wide.association_probabilities
    )
    assert second_narrow.minimum_local_waveform_retained_mass is not None
    assert second_wide.minimum_local_waveform_retained_mass is not None
    assert second_wide.minimum_local_waveform_retained_mass >= (
        second_narrow.minimum_local_waveform_retained_mass
    )


def test_model3_waveform_state_posterior_does_not_use_model2_distance_gates():
    narrow_gates = HMMModel2Config(
        model1=HMMModel1Config(rf_new_state_gate_hz=1.0),
        pulse_width_new_state_gate_log=1e-6,
    )
    wide_gates = HMMModel2Config(
        model1=HMMModel1Config(rf_new_state_gate_hz=10e9),
        pulse_width_new_state_gate_log=10.0,
    )
    narrow = BayesianModel3Tracker(config(), narrow_gates)
    wide = BayesianModel3Tracker(config(), wide_gates)

    for index in range(20):
        item = pdw(index, "left" if index < 10 else "right")
        narrow_update = narrow.process(item)
        wide_update = wide.process(item)
        assert narrow_update == wide_update

    narrow_snapshots = narrow.snapshots()
    wide_snapshots = wide.snapshots()
    for rows in (narrow_snapshots, wide_snapshots):
        for row in rows:
            row["representative_model"].pop("config")
            row["representative_model"].pop("model2_config")
    assert narrow_snapshots == wide_snapshots
    assert "distance gates are inactive" in narrow.summary()[
        "waveform_state_birth"
    ]
