import json

from plot_model2_timing_association import plot_comparison


def test_plot_comparison_writes_image(tmp_path):
    result = {
        "input_boundary": {"held_out_start_toa_s": 1.0},
        "calibration": {"acceptance_threshold": -20.0},
        "timing_model": {"maximum_boost": 0.75},
        "baseline": {
            "true_positive": 1,
            "false_negative": 1,
            "false_positive": 0,
        },
        "timing_on": {
            "true_positive": 2,
            "false_negative": 0,
            "false_positive": 0,
        },
    }
    records = [
        {
            "toa_s": 1.1,
            "evaluation_is_target": True,
            "baseline": {"accepted": True, "baseline_log_evidence": -19.0},
            "timing_on": {
                "accepted": True,
                "combined_log_evidence": -18.8,
                "timing_applied_boost": 0.2,
            },
        },
        {
            "toa_s": 1.2,
            "evaluation_is_target": True,
            "baseline": {"accepted": False, "baseline_log_evidence": -20.2},
            "timing_on": {
                "accepted": True,
                "combined_log_evidence": -19.7,
                "timing_applied_boost": 0.5,
            },
        },
        {
            "toa_s": 1.3,
            "evaluation_is_target": False,
            "baseline": {"accepted": False, "baseline_log_evidence": -80.0},
            "timing_on": {
                "accepted": False,
                "combined_log_evidence": -80.0,
                "timing_applied_boost": 0.0,
            },
        },
    ]
    output = tmp_path / "comparison.png"
    plot_comparison(result, records, output)
    assert output.exists()
    assert output.stat().st_size > 1000
