"""Analytical tests of Stage A detection likelihood and its identifiability."""
import math
import pytest
from joint_detection_diagnostic import detection_log_likelihood, score


def test_equal_pd_symmetric_hypotheses_cancel():
    a = score((1, 0), (0, 1), (.7, .7))
    b = score((0, 1), (1, 0), (.7, .7))
    assert a == pytest.approx(b)


def test_emitter_specific_pd_produces_analytic_odds():
    a = score((1, 0), (0, 1), (.99, .60))
    b = score((0, 1), (1, 0), (.99, .60))
    assert math.exp(a-b) == pytest.approx((.99*.40)/(.60*.01))


def test_different_missed_slot_counts_change_odds():
    a = score((1, 0), (1, 0), (.7, .7))
    b = score((0, 1), (0, 2), (.7, .7))
    assert math.exp(a-b) == pytest.approx(1/.3)


def test_perfect_detection_excludes_misses():
    assert detection_log_likelihood(1, 1, 1.0) == -math.inf


def test_invalid_pd_rejected():
    with pytest.raises(ValueError):
        detection_log_likelihood(1, 0, 0)
