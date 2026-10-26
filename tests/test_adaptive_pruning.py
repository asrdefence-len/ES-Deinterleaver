"""Opt-in mass pruning regression tests."""
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker


INITIAL=[(0.,.001,1e-6,.2e-6),(.0003,.001,1e-6,.2e-6)]
OBS=(0.,.0003,.001,.0013,.002,.0023)


def test_disabled_and_full_mass_match():
    a=JointTimingTracker(INITIAL,beam_width=16)
    b=JointTimingTracker(INITIAL,beam_width=16,retained_mass=1.)
    for toa in OBS:
        x=a.process(toa,20.)
        y=b.process(toa,20.)
        assert x["association_probabilities"]==pytest.approx(y["association_probabilities"])
        assert x["best_assignments"]==y["best_assignments"]
        assert x["best_pulse_indices"]==y["best_pulse_indices"]


def test_mass_pruning_reduces_hypotheses():
    t=JointTimingTracker(INITIAL,beam_width=32,retained_mass=.9,min_beam=2)
    for toa in OBS:
        s=t.process(toa,20.)
        assert 2<=s["retained_hypotheses"]<=32
        d=s["pruning_diagnostics"]
        assert 0<=d["discarded_mass_before_renormalization"]<=1
        assert d["candidate_count"]>=s["retained_hypotheses"]


def test_invalid_mass_parameters():
    with pytest.raises(ValueError):
        JointTimingTracker(INITIAL,retained_mass=1.01)
    with pytest.raises(ValueError):
        JointTimingTracker(INITIAL,beam_width=4,min_beam=5)
