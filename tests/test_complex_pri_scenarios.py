"""Regression checks for complex-PRI simulation and observation integrity."""
import pytest
from testbed.complex_pri_scenarios import KINDS, generate_observation, transmitted_pris


@pytest.mark.parametrize("kind", KINDS)
def test_all_complex_pri_patterns_generate_valid_observations(kind):
    obs, truth = generate_observation(kind, seed=42, detection_probability=0.70)
    assert len(obs.pri_intervals_s) == obs.pulse_count - 1
    assert all(v > 0 for v in obs.pri_intervals_s)
    assert obs.start_toa_s <= obs.end_toa_s
    assert len(truth["transmitted_intervals_us"]) == 180
    assert truth["detected_count"] == obs.pulse_count


def test_stagger5_repeats_exactly_without_noise():
    assert transmitted_pris("stagger5", 10) == [
        780, 950, 1130, 870, 1270] * 2


def test_deterministic_reproduction():
    a, truth_a = generate_observation("sinusoidal", seed=17)
    b, truth_b = generate_observation("sinusoidal", seed=17)
    assert a == b
    assert truth_a == truth_b


def test_missed_pulses_produce_longer_observed_intervals():
    obs, _ = generate_observation(
        "constant", seed=5, detection_probability=0.70, toa_sigma_s=0)
    assert any(v > 0.0015 for v in obs.pri_intervals_s)


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        generate_observation("constant", detection_probability=0)
