import math

import pytest

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
