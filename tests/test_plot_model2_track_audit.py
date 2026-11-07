from sequential_pdw_tracker.audit_trace import Model2AuditTraceRecorder
from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.model2_birth import Model2BirthDecision
from plot_model2_track_audit import plot_track_audit_pages


def test_track_audit_writes_five_tracks_per_page(tmp_path):
    recorder = Model2AuditTraceRecorder(28)
    tracks = []
    events = []
    for track_index in range(7):
        track_id = f"T{track_index + 1}"
        truth = str(track_index + 10)
        tracks.append({
            "track_id": track_id,
            "status": "confirmed" if track_index < 5 else "tentative",
            "assigned_count_evaluation": 4,
            "dominant_truth_emitter_evaluation_only": truth,
            "evaluation_purity": 1.0,
            "truth_counts_evaluation_only": {truth: 4},
        })
        events.append({
            "track_id": track_id,
            "event": "birth",
            "toa_s": track_index * 4e-3,
        })
        for offset in range(4):
            index = track_index * 4 + offset
            pdw = PDW(
                pdw_id=index,
                toa_s=index * 1e-3,
                frequency_hz=(10_000 + track_index * 50) * 1e6,
                pulse_width_s=(0.1 + track_index * 0.01) * 1e-6,
                aoa_deg=-100.0 + track_index * 25.0,
                amplitude_db=-70.0,
            )
            decision = Model2BirthDecision(
                pdw_id=index,
                toa_s=pdw.toa_s,
                outcome="confirmed_update",
                track_id=track_id,
                track_confirmed_after_update=track_index < 5,
                confirmed_margins={track_id: 1.0},
                timing_boosts={track_id: 0.0},
                tentative_aoa_errors_deg={},
                confirmed_score_upper_bound_pruned=(),
                rejection_reason=None,
            )
            recorder.record(pdw, truth, decision)

    trace = tmp_path / "audit.npz"
    recorder.save(trace)
    result = {
        "tracker": {"tracks": tracks, "lifecycle_events": events},
        "target_evaluation": {
            "46": {"primary_track_id": "T1"},
            "38": {"primary_track_id": "T2"},
        },
    }
    outputs = plot_track_audit_pages(
        result,
        trace,
        tmp_path / "pages",
        tracks_per_page=5,
        background_points=20,
    )
    assert [path.name for path in outputs] == [
        "track_audit_page_01.png", "track_audit_page_02.png",
    ]
    assert all(path.exists() and path.stat().st_size > 1000 for path in outputs)
