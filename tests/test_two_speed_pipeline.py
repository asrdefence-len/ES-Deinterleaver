"""Smoke and accounting checks for the two-speed experiment."""
from compare_two_speed_pipeline import run


def test_pipeline_counts_and_preserves_causality():
    result = run()
    assert result["pdw_count"] > 3000
    assert result["observation_count"] > 0
    assert result["upstream_tracks"] >= 1
    assert result["downstream_tracks"] >= 1
    assert len(result["windows"]) == 9
    for row in result["windows"]:
        assert 0 <= row["correct_identity"] <= row["downstream_assigned"]
        assert row["downstream_assigned"] <= row["upstream_assigned"]
        assert row["upstream_assigned"] <= row["observed"]
    assert len(result["downstream_assignments"]) <= result["pdw_count"]
