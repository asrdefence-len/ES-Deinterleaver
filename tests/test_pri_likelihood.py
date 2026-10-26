"""Focused checks of missing-pulse likelihood and contamination gate."""
import math
import pytest
from testbed.pri_likelihood import interval_log_likelihood, fit_constant_jitter
from testbed.complex_pri_scenarios import generate_observation


def test_missing_pulse_interval_likely_under_constant_model():
    a = interval_log_likelihood(.002, .001, detection_probability=.7)
    b = interval_log_likelihood(.002, .001, detection_probability=1)
    assert a > b


def test_outlier_does_not_produce_negative_infinity():
    score = interval_log_likelihood(.017, .001, detection_probability=.95)
    assert math.isfinite(score)


def test_invalid_detection_probability():
    with pytest.raises(ValueError):
        interval_log_likelihood(.001, .001, detection_probability=0)


@pytest.mark.parametrize("pd", [1.0, .95, .7])
def test_constant_with_missing_pulses(pd):
    obs, _ = generate_observation("constant", seed=3,
                                  detection_probability=pd, count=120)
    result = fit_constant_jitter(obs, detection_probability=pd)
    assert result["winner"] == "constant"
    assert abs(result["fits"]["constant"]["pri_s"]-.001) < .00005


@pytest.mark.parametrize("pd", [1.0, .95, .7])
def test_jitter_with_missing_pulses(pd):
    obs, _ = generate_observation("jitter", seed=3,
                                  detection_probability=pd, count=120)
    result = fit_constant_jitter(obs, detection_probability=pd)
    assert result["winner"] == "jitter"
