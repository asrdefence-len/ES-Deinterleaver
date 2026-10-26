"""Analytical and synthetic checks for the fixed-parameter TOA model."""
import math
import random
import pytest
from testbed.bayesian_timing_baseline import (
    toa_sequence_log_likelihood as ll, _normal_logpdf)


def test_one_detection_is_normal_density():
    assert ll([0.0], 0.0, .001, 1e-6) == pytest.approx(
        _normal_logpdf(0.0, 0.0, 1e-6))


def test_two_detections_with_no_misses_matches_exact_density():
    sigma = 1e-6
    times = [0.0, .001]
    expected = (_normal_logpdf(0, 0, sigma) +
                _normal_logpdf(.001, .001, sigma))
    assert ll(times, 0, .001, sigma, 1.0) == pytest.approx(expected)


def test_missing_pulse_has_geometric_prior():
    sigma = 1e-6
    times = [0.0, .002]
    expected = (_normal_logpdf(0, 0, sigma) +
                math.log(.5*.5) +
                _normal_logpdf(.002, .002, sigma))
    assert ll(times, 0, .001, sigma, .5) == pytest.approx(expected)


def test_correct_pri_outscores_incorrect_pri():
    rng = random.Random(23)
    t = 0.0
    times = []
    for _ in range(45):
        if rng.random() < .8:
            times.append(t+rng.gauss(0, 50e-9))
        t += .001
    times.sort()
    epoch = 0.0 if times[0] < .0005 else .001
    # Choose a sequence with its first detection at slot zero by
    # anchoring epoch at the first observation.
    epoch = times[0]
    assert ll(times, epoch, .001, 50e-9, .8) > ll(
        times, epoch, .0009, 50e-9, .8)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ll([0.0], 0.0, -1)
    with pytest.raises(ValueError):
        ll([.002, .001], 0.0, .001)
    with pytest.raises(ValueError):
        ll([], 0.0, .001)
