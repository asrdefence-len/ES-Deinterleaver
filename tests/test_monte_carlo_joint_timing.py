"""Reproducibility and honest diagnostics for joint Monte Carlo."""
import pytest
from monte_carlo_joint_timing import simulate, one_trial, experiment


def test_snr_schedule_and_truth_indices():
    rows,truth=simulate(7,count=12,pd=1.)
    assert len(rows)==24
    assert {x[1] for x in rows if x[2]==0}=={0.,30.}
    assert all(0<=x[3]<12 for x in rows)


def test_trial_is_reproducible():
    a=one_trial(7,count=8,pd=.95,beam=16)
    assert a==one_trial(7,count=8,pd=.95,beam=16)
    assert 0<=a["accuracy"]<=1
    assert a["first_truth_pruned"] is None or a["first_truth_pruned"]>=0


def test_aggregate_counts_and_calibration():
    r=experiment(trials=2,count=8,pd=.7,beam=16)
    assert sum(x["n"] for x in r["calibration"])==r["detected_pdws"]
    assert 0<=r["joint_brier"]<=1


def test_invalid_pd():
    with pytest.raises(ValueError):
        simulate(0,count=10,pd=0)
