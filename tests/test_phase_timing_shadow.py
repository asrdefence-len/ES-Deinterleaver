import numpy as np
import pytest

from sequential_pdw_tracker.phase_timing_shadow import (
    PhaseTimingShadowConfig,
    analyze_phase_timing_shadow,
    fit_phase_timing_shadow,
)


def synthetic_train(period_us: float = 10.0, pulses: int = 1200):
    rng = np.random.default_rng(17)
    intervals = rng.geometric(0.45, size=pulses - 1)
    jitter_us = rng.normal(0.0, 0.25, size=pulses - 1)
    gaps_us = intervals * period_us + jitter_us
    toa_us = np.r_[0.0, np.cumsum(gaps_us)]
    active = np.full(pulses, 0.98)
    return toa_us * 1e-6, active


def test_fit_prefers_fundamental_over_subharmonic_with_skipped_intervals():
    toa_s, _active = synthetic_train()
    gaps_s = np.diff(toa_s)
    config = PhaseTimingShadowConfig(
        min_period_s=4e-6,
        max_period_s=12e-6,
        period_step_s=5e-9,
        min_training_gaps=50,
    )
    fit, _periods, _scores = fit_phase_timing_shadow(gaps_s, config)
    assert fit.period_s * 1e6 == pytest.approx(10.0, abs=0.02)
    assert fit.detection_probability == pytest.approx(0.45, abs=0.05)


def test_held_out_phase_evidence_beats_half_cycle_control_without_feedback():
    toa_s, active = synthetic_train()
    config = PhaseTimingShadowConfig(
        min_period_s=7e-6,
        max_period_s=12e-6,
        period_step_s=5e-9,
        min_training_gaps=50,
    )
    result, diagnostics = analyze_phase_timing_shadow(toa_s, active, config)
    phase = result["held_out"]["phase_log_likelihood_ratio"]
    control = result["held_out"]["half_cycle_negative_control_log_likelihood_ratio"]
    assert result["feedback_weight"] == 0.0
    assert result["transmitter_pri_asserted"] is False
    assert phase["mean"] > 1.0
    assert phase["positive_fraction"] > 0.9
    assert control["mean"] < 0.0
    assert len(diagnostics["held_out_residual_s"]) == phase["count"]
    assert result["fit"]["competing_local_maxima"]
    assert result["held_out"]["scoreable_segments"]


def test_visibility_threshold_excludes_coast_gaps_from_fit_and_evaluation():
    toa_s, active = synthetic_train(pulses=600)
    coast = np.arange(100, 600, 100)
    active[coast] = 0.01
    result, _diagnostics = analyze_phase_timing_shadow(
        toa_s,
        active,
        PhaseTimingShadowConfig(
            min_period_s=8e-6,
            max_period_s=12e-6,
            period_step_s=5e-9,
            min_training_gaps=40,
        ),
    )
    assert result["fit"]["dominant_interval_s"] * 1e6 == pytest.approx(10.0, abs=0.02)
    assert result["held_out"]["active_gaps"] < int(600 * 0.4)
