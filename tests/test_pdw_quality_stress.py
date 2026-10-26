"""Receiver-independent PDW measurement uncertainty regression tests."""
import math
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker
from pdw_quality_stress import simulate, trial, experiment


def test_direct_sigma_without_snr():
    t=JointTimingTracker([(100e-6,.001,1e-6,.2e-6),
                          (122e-6,.000999,1e-6,.2e-6)],beam_width=8)
    snap=t.process(100e-6,toa_sigma_s=100e-9)
    assert abs(sum(snap["association_probabilities"])-1)<1e-10


def test_direct_sigma_matches_legacy_when_equal():
    from two_emitter_snr_baseline import toa_sigma
    initial=[(100e-6,.001,1e-6,.2e-6),(122e-6,.000999,1e-6,.2e-6)]
    a=JointTimingTracker(initial,beam_width=16)
    b=JointTimingTracker(initial,beam_width=16)
    x=a.process(100e-6,20.)
    y=b.process(100e-6,toa_sigma_s=toa_sigma(20.))
    assert x["association_probabilities"] == pytest.approx(y["association_probabilities"])


@pytest.mark.parametrize("bad",[0.,-1.,float("nan"),float("inf")])
def test_invalid_toa_sigma(bad):
    t=JointTimingTracker([(100e-6,.001,1e-6,.2e-6)]*2)
    with pytest.raises(ValueError,match="toa_sigma_s"):
        t.process(100e-6,toa_sigma_s=bad)


def test_simulated_pdws_reproducible_and_ordered():
    a=simulate(12,count=10,variable_sigma_ns=(50.,250.))
    b=simulate(12,count=10,variable_sigma_ns=(50.,250.))
    assert a==b
    assert [x[0] for x in a[1]]==sorted(x[0] for x in a[1])
    assert set(round(x[1]*1e9) for x in a[1]) <= {50,250}


def test_abstention_is_output_only():
    a=trial(19017,count=10,abstain_threshold=.5)
    b=trial(19017,count=10,abstain_threshold=.9)
    assert a["accuracy"]==b["accuracy"]
    assert a["brier"]==b["brier"]
    assert a["assignment_coverage"]==1.
    assert b["assignment_coverage"]<=a["assignment_coverage"]


def test_experiment_smoke():
    x=experiment(trials=2,count=8,toa_sigma_ns=100.)
    assert x["detected_pdws"]>0
    assert 0<=x["assignment_coverage"]<=1
