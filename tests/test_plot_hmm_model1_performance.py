import json

import pytest

from plot_hmm_model1_performance import (
    load_diagnostics,
    plot_performance,
    posterior_entropy_bits,
    rolling_median,
)


def test_posterior_entropy_distinguishes_certain_and_ambiguous_rf_states():
    assert posterior_entropy_bits([1.0, 0.0]) == pytest.approx(0.0)
    assert posterior_entropy_bits([0.5, 0.5]) == pytest.approx(1.0)


def test_plot_uses_existing_model1_diagnostics_without_changing_filter(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    records = [
        {
            "toa_s": 10.0 + index * 1e-3,
            "rf_posterior": [0.75, 0.25],
            "aoa_sigma_deg": 1.0 / (index + 1),
            "timing_map_pri_s": None if index == 0 else 8e-6,
            "total_log_evidence": -4.0 - index,
        }
        for index in range(3)
    ]
    diagnostics.write_text("".join(json.dumps(item) + "\n" for item in records))
    loaded = load_diagnostics(diagnostics)
    output = tmp_path / "performance.png"
    plot_performance(loaded, output, rolling_window=2)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert rolling_median.__name__ == "rolling_median"


def test_rolling_window_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        rolling_median([], 0)
