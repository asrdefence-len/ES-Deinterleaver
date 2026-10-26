from testbed.bayesian_timing_state import BayesianTimingEstimator
from diagnose_timing_ambiguity import run


def test_mixture_moments_single_hypothesis():
    e = BayesianTimingEstimator(.001, .00001)
    e.process(0)
    s = e.snapshot()
    assert s["mixture_pri_s"] == s["pri_s"]
    assert abs(s["mixture_pri_sigma_s"]-s["pri_sigma_s"]) < 1e-12


def test_diagnostic_is_reproducible():
    assert run(.5, 0, .002, 32) == run(.5, 0, .002, 32)
