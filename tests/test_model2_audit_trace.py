import numpy as np

from sequential_pdw_tracker.audit_trace import Model2AuditTraceRecorder
from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.model2_birth import Model2BirthDecision


def _pdw(index: int) -> PDW:
    return PDW(
        pdw_id=index,
        toa_s=index * 1e-3,
        frequency_hz=(10_000 + index) * 1e6,
        pulse_width_s=0.1e-6,
        aoa_deg=20.0,
        amplitude_db=-70.0,
    )


def _decision(index: int, track_id: str | None) -> Model2BirthDecision:
    return Model2BirthDecision(
        pdw_id=index,
        toa_s=index * 1e-3,
        outcome="confirmed_update" if track_id else "rejected",
        track_id=track_id,
        track_confirmed_after_update=track_id is not None,
        confirmed_margins={} if track_id is None else {track_id: 2.5},
        timing_boosts={} if track_id is None else {track_id: 0.25},
        tentative_aoa_errors_deg={},
        confirmed_score_upper_bound_pruned=(),
        rejection_reason=None if track_id else "test",
    )


def test_audit_trace_records_post_decision_measurements_and_codes(tmp_path):
    recorder = Model2AuditTraceRecorder(2)
    recorder.record(_pdw(0), 46, _decision(0, "T1"))
    recorder.record(_pdw(1), 38, _decision(1, None))
    output = tmp_path / "audit.npz"
    recorder.save(output)

    with np.load(output, allow_pickle=False) as trace:
        assert trace["pdw_id"].tolist() == [0, 1]
        assert trace["track_ids"].tolist() == ["T1"]
        assert trace["track_code"].tolist() == [0, -1]
        assert trace["truth_labels"].tolist() == ["46", "38"]
        assert trace["truth_code"].tolist() == [0, 1]
        assert trace["assignment_margin"][0] == 2.5
        assert np.isnan(trace["assignment_margin"][1])
        assert trace["timing_boost"][0] == 0.25


def test_audit_trace_requires_npz_suffix(tmp_path):
    recorder = Model2AuditTraceRecorder(1)
    recorder.record(_pdw(0), 46, _decision(0, "T1"))
    try:
        recorder.save(tmp_path / "audit.bin")
    except ValueError as error:
        assert "must end in .npz" in str(error)
    else:
        raise AssertionError("non-NPZ trace path was accepted")
