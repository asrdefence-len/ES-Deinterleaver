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

def test_experiment_forwards_detection_probability(monkeypatch):
    import crossing_pulse_stress as cps
    seen = []
    original = cps.trial

    def recording_trial(*args, **kwargs):
        seen.append(kwargs.get("detection_probability"))
        return original(*args, **kwargs)

    monkeypatch.setattr(cps, "trial", recording_trial)
    cps.experiment("very_close", trials=1, count=6, beam=8,
                   detection_probability=.7)
    assert seen == [.7]

def test_joint_pd_penalizes_other_emitter_nondetection():
    from testbed.joint_bayesian_timing import JointHypothesis
    t=JointTimingTracker(INITIAL, beam_width=64, max_skip=2,
                         joint_detection_probability=.7, joint_guard_s=0.)
    t.process(0.,30.)
    # The competing emitter has a predicted slot at 0.0003 s.
    t.process(.0006,30.)
    assert all(len(h.accounted_indices)==2 for h in t.hypotheses)
    assert any(h.accounted_indices[1]>=0 for h in t.hypotheses)


def test_joint_model_validation():
    with pytest.raises(ValueError):
        JointTimingTracker(INITIAL, joint_detection_probability=.7,
                           detection_probability=.7)
    with pytest.raises(ValueError):
        JointTimingTracker(INITIAL, joint_detection_probability=0.)


def test_joint_model_cli_experiment_smoke():
    from crossing_pulse_stress import experiment
    r=experiment("very_close",trials=1,count=6,beam=16,
                 joint_detection_probability=.7)
    assert 0 <= r["joint_accuracy"] <= 1
