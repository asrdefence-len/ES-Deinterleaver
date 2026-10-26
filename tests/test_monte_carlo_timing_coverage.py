"""Smoke tests for timing uncertainty Monte Carlo report."""
from monte_carlo_timing_coverage import one_trial, summarise


def test_trial_reproducible():
    a = one_trial(.7, 1, 0)
    assert a == one_trial(.7, 1, 0)
    assert a["detections"] >= 1
    assert a["reported_sigma_ns"] >= 0


def test_summary_rates_bounded():
    rows = [one_trial(.95, i, 0) for i in range(3)]
    s = summarise(rows)[0]
    assert s["trials"] == 3
    for key in ("coverage_1sigma", "coverage_2sigma", "coverage_3sigma",
                "index_sequence_accuracy", "missed_count_accuracy"):
        assert 0 <= s[key] <= 1
    assert s["coverage_1sigma"] <= s["coverage_2sigma"] <= s["coverage_3sigma"]
