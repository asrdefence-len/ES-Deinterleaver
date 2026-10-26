"""Tests for the experimental two-emitter joint timing filter."""
import math
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker


def test_association_probabilities_normalize():
    t=JointTimingTracker([(0.,.001,1e-6,1e-7),
                          (.0002,.00075,1e-6,1e-7)])
    for toa,snr in [(0.,30),(.0002,0),(.001,30),(.00095,10)]:
        # Input must be sorted, so use a separate sorted loop below.
        pass
    for toa,snr in sorted([(0.,30),(.0002,0),(.001,30),(.00095,10)]):
        s=t.process(toa,snr)
        assert sum(s["association_probabilities"]) == pytest.approx(1.)
        assert all(0<=p<=1 for p in s["association_probabilities"])


def test_state_changes_with_measurement_quality():
    initial=[(0.,.001,50e-9,1e-6),(.0003,.0008,50e-9,1e-6)]
    hi=JointTimingTracker(initial)
    lo=JointTimingTracker(initial)
    hi.process(0.,20)
    lo.process(0.,20)
    a=hi.process(.001+300e-9,30)
    b=lo.process(.001+300e-9,0)
    assert a["best_tracks"][0].pri != b["best_tracks"][0].pri


def test_near_coincident_measurement_is_ambiguous():
    t=JointTimingTracker([(0.,.001,1e-6,1e-9),
                          (0.,.001,1e-6,1e-9)])
    s=t.process(0.,0)
    assert s["association_probabilities"][0] == pytest.approx(.5)
    assert s["association_probabilities"][1] == pytest.approx(.5)


def test_reject_out_of_order():
    t=JointTimingTracker([(0.,.001,1e-6,1e-7),
                          (0.,.00075,1e-6,1e-7)])
    t.process(0.,10)
    with pytest.raises(ValueError):
        t.process(0.,10)
