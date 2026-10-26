"""Conditional geometric gap-prior checks (not full multi-emitter Pd model)."""
import math
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker

INITIAL=[(0.,.001,1e-6,.2e-6),(.0003,.001,1e-6,.2e-6)]


def test_default_matches_explicit_none():
    a=JointTimingTracker(INITIAL,beam_width=32)
    b=JointTimingTracker(INITIAL,beam_width=32,detection_probability=None)
    for toa in (0.,.0003,.001,.0013):
        x=a.process(toa,20.)
        y=b.process(toa,20.)
        assert x["association_probabilities"]==pytest.approx(y["association_probabilities"])


def test_pd_one_forbids_skipped_pulses():
    t=JointTimingTracker(INITIAL,beam_width=32,detection_probability=1.)
    t.process(0.,20.)
    assert all(h.tracks[h.last_emitter].index == 0 for h in t.hypotheses)


def test_invalid_detection_probability():
    for value in (0.,-0.1,1.1,float("nan")):
        with pytest.raises(ValueError):
            JointTimingTracker(INITIAL,detection_probability=value)
