import numpy as np

from analyze_detected_pulse_train import analyze_toas, plot_analysis


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
