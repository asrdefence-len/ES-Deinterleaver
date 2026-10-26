"""Paired PDW quality experiment checks."""
import pytest
from paired_pdw_quality import compare_trial, experiment


def test_identical_quality_has_identical_arms():
    r=compare_trial(19017,count=10,sigmas_ns=(100.,))
    assert r["aware"]==r["blind"]


def test_reproducible_paired_experiment():
    a=experiment(trials=2,count=10)
    b=experiment(trials=2,count=10)
    assert a==b
    assert a["quality_aware"]["detected_pdws"]==a["quality_blind"]["detected_pdws"]


def test_coverage_and_false_confident_accounting():
    r=experiment(trials=2,count=10,threshold=.8)
    for arm in ("quality_aware","quality_blind"):
        x=r[arm]
        assert 0<=x["assignment_coverage"]<=1
        assert 0<=x["false_confident_rate_all_pdws"]<=x["assignment_coverage"]
        assert x["false_confident_rate_all_pdws"]==pytest.approx(
            x["assignment_coverage"]*x["false_confident_rate_among_assigned"])
        assert sum(b["n"] for b in x["calibration"])==x["detected_pdws"]
        assert sum(b["n"] for b in x["confidence_calibration"])==x["detected_pdws"]


@pytest.mark.parametrize("bad",[0.,-10.,float("nan")])
def test_bad_fixed_sigma(bad):
    with pytest.raises(ValueError):
        compare_trial(19017,count=10,fixed_sigma_ns=bad)
