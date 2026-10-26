"""Tests for identity-continuity diagnostic (no tracker modification)."""
import pytest
from identity_continuity_stress import streaks,evaluate,experiment


def test_streaks():
    assert streaks([False,True,True,True,False,True],3)==[(1,3)]
    assert streaks([True,True],3)==[]
    with pytest.raises(ValueError):
        streaks([True],0)


def test_identity_regions_partition_and_map():
    r=evaluate(19017,count=12)
    assert sum(v["n"] for v in r["regions"].values())==len(r["records"])
    assert all(rec["final_map_emitter"] in (0,1) for rec in r["records"])
    assert sum(v["n"] for v in r["per_emitter"].values())==len(r["records"])


def test_repeatable_experiment():
    a=experiment(trials=2,count=12)
    assert a==experiment(trials=2,count=12)
    assert sum(v["n"] for v in a["regions"].values())>0


def test_threshold_does_not_change_online_labels():
    a=evaluate(19017,count=12,threshold=.5)
    b=evaluate(19017,count=12,threshold=.9)
    assert [r["predicted_emitter"] for r in a["records"]]==[
        r["predicted_emitter"] for r in b["records"]]
    assert [r["final_map_emitter"] for r in a["records"]]==[
        r["final_map_emitter"] for r in b["records"]]
