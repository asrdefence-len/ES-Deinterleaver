"""Tests for paired close-PRI crossing scenarios."""
import pytest
from crossing_pulse_stress import SCENARIOS, trial, experiment


def test_crossing_pri_differences():
    assert SCENARIOS["baseline"] == pytest.approx(10e-6)
    assert SCENARIOS["close"] == pytest.approx(1e-6)
    assert SCENARIOS["very_close"] == pytest.approx(.1e-6)


def test_repeatable_trace_and_calibration():
    a=trial(9,"very_close",count=8,beam=16,record_trace=True)
    b=trial(9,"very_close",count=8,beam=16,record_trace=True)
    assert a==b
    assert len(a["trace"])==a["n"]
    assert all(0<=row[5]<=1 for row in a["trace"])


def test_experiment_counts_and_metrics():
    x=experiment("close",trials=2,count=8,pd=.7,beam=16)
    assert sum(b["n"] for b in x["calibration"])==x["detected_pdws"]
    assert 0<=x["truth_history_survival_fraction"]<=1
    assert 0<=x["joint_brier"]<=1


def test_invalid_scenario():
    with pytest.raises(ValueError):
        trial(1,"unknown")
