from dataclasses import asdict
import json

from plot_hmm_model2_performance import plot_performance
from sequential_pdw_tracker.hmm_emitter_model2 import HMMEmitterModel2
from sequential_pdw_tracker.model import PDW


def test_model2_plot_uses_existing_diagnostics(tmp_path):
    tracker = HMMEmitterModel2()
    records = []
    for index in range(50):
        update = tracker.process(PDW(
            pdw_id=index,
            toa_s=index * 10e-6,
            frequency_hz=(1.0 + 0.2 * (index % 3)) * 1e9,
            pulse_width_s=(0.10, 0.27, 0.43)[index % 3] * 1e-6,
            aoa_deg=32.0,
            amplitude_db=-70.0,
        ))
        records.append(asdict(update))
    diagnostics = tmp_path / "model2.jsonl"
    diagnostics.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    output = tmp_path / "model2.png"
    plot_performance(records, output, rolling_window=10)
    assert diagnostics.is_file()
    assert output.is_file()
    assert output.stat().st_size > 0
