from plot_model2_full_scene_birth import plot_full_scene


def test_plot_full_scene_writes_image(tmp_path):
    tracks = []
    events = []
    for index in range(2):
        track_id = f"T{index + 1}"
        tracks.append({
            "track_id": track_id,
            "status": "confirmed",
            "assigned_count_evaluation": 100 - index * 10,
            "dominant_truth_emitter_evaluation_only": str(index),
            "dominant_truth_count_evaluation_only": 100 - index * 10,
            "evaluation_purity": 1.0,
            "truth_counts_evaluation_only": {str(index): 100 - index * 10},
        })
        events.extend((
            {"pdw_id": index, "event": "birth", "track_id": track_id},
            {"pdw_id": 100 + index, "event": "confirmed", "track_id": track_id},
        ))
    result = {
        "input_boundary": {"source_pdws": 1000},
        "config": {"birth": {"max_tracks": 2}},
        "tracker": {
            "tracks": tracks,
            "lifecycle_events": events,
            "capacity_first_reached": {"processed_pdws": 2},
            "assigned_pdws": 190,
            "assignment_coverage": 0.19,
            "rejection_reasons": {
                "track_capacity": 700,
                "confirmed_aoa_birth_suppression": 110,
            },
        },
        "controlled_reference": {"E46_recall": 0.99, "E38_recall": 0.97},
        "target_evaluation": {
            "46": {"assigned_coverage": 0.0},
            "38": {"assigned_coverage": 0.0},
        },
    }
    output = tmp_path / "full_scene.png"
    plot_full_scene(result, output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_plot_full_scene_scales_to_many_tracks_without_reaching_capacity(
    tmp_path,
):
    tracks = []
    events = []
    for index in range(36):
        track_id = f"T{index + 1}"
        count = 1000 - index * 20
        tracks.append({
            "track_id": track_id,
            "status": "confirmed" if index < 24 else "tentative",
            "assigned_count_evaluation": count,
            "dominant_truth_emitter_evaluation_only": str(index),
            "dominant_truth_count_evaluation_only": count,
            "evaluation_purity": 1.0,
            "truth_counts_evaluation_only": {str(index): count},
        })
        events.append({
            "pdw_id": index,
            "event": "birth",
            "track_id": track_id,
        })
        if index < 24:
            events.append({
                "pdw_id": 500 + index,
                "event": "confirmed",
                "track_id": track_id,
            })
    result = {
        "input_boundary": {"source_pdws": 50_000},
        "config": {"birth": {"max_tracks": 256}},
        "tracker": {
            "tracks": tracks,
            "lifecycle_events": events,
            "capacity_first_reached": None,
            "assigned_pdws": sum(
                row["assigned_count_evaluation"] for row in tracks
            ),
            "assignment_coverage": 0.6,
            "rejection_reasons": {},
        },
        "controlled_reference": {"E46_recall": 0.99, "E38_recall": 0.97},
        "target_evaluation": {
            "46": {"assigned_coverage": 0.98},
            "38": {"assigned_coverage": 0.96},
        },
    }
    output = tmp_path / "many_tracks.png"
    plot_full_scene(result, output)
    assert output.exists()
    assert output.stat().st_size > 1000
