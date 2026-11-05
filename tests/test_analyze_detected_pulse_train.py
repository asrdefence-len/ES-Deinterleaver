import numpy as np

from analyze_detected_pulse_train import analyze_toas, plot_analysis
from analyze_phase_timing_shadow import plot_phase_timing_shadow
from sequential_pdw_tracker.phase_timing_shadow import (
    PhaseTimingShadowConfig,
    analyze_phase_timing_shadow,
)


def test_detected_train_analysis_describes_lags_without_claiming_pri(tmp_path):
    toa_us = np.cumsum([0, *([10, 20, 10, 30] * 30)])
    analysis = analyze_toas(toa_us * 1e-6, blocks=6)
    assert analysis["transmitter_pri_estimate"] is None
    assert analysis["detected_pulses"] == len(toa_us)
    assert analysis["lag_difference_peaks"]
    output = tmp_path / "train.png"
    plot_analysis(toa_us * 1e-6, analysis, output)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_phase_timing_shadow_plot_writes_held_out_diagnostic(tmp_path):
    rng = np.random.default_rng(3)
    intervals = rng.geometric(0.5, size=499)
    gaps_us = intervals * 9.0 + rng.normal(0.0, 0.2, size=len(intervals))
    toa_s = np.r_[0.0, np.cumsum(gaps_us)] * 1e-6
    result, diagnostics = analyze_phase_timing_shadow(
        toa_s,
        np.full(len(toa_s), 0.99),
        PhaseTimingShadowConfig(
            min_period_s=7e-6,
            max_period_s=11e-6,
            period_step_s=10e-9,
            min_training_gaps=30,
        ),
    )
    output = tmp_path / "phase-shadow.png"
    plot_phase_timing_shadow(result, diagnostics, output)
    assert output.is_file()
    assert output.stat().st_size > 0
