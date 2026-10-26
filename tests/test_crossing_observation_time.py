"""Paired crossing model integration and receiver silence regression."""
import pytest
from crossing_pulse_stress import trial, experiment


def test_observation_time_crossing_trial_runs():
    row=trial(19017,scenario="very_close",count=12,pd=.7,beam=16,
              observation_detection_probability=.7)
    assert 0 <= row["accuracy"] <= 1
    assert 0 <= row["brier"] <= 1


def test_observation_time_silent_receiver_window():
    row=trial(19017,scenario="very_close",count=12,pd=.7,beam=16,
              observation_detection_probability=.7,
              silent_window=(3e-3,6e-3))
    assert row["n"] > 0
    assert 0 <= row["accuracy"] <= 1


def test_paired_trial_observation_count_matches():
    opts=dict(scenario="very_close",count=12,pd=.7,beam=16)
    a=trial(19017,**opts)
    b=trial(19017,**opts,observation_detection_probability=.7)
    c=trial(19017,**opts,joint_detection_probability=.7)
    assert a["n"]==b["n"]==c["n"]


def test_silent_window_rejects_invalid_bounds():
    with pytest.raises(ValueError,match="silent window"):
        trial(19017,count=12,silent_window=(5e-3,3e-3))
