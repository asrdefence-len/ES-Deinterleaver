"""Incremental interface and track consistency checks."""
import json
from pathlib import Path
import pytest
from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver


def sample():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    return generate(scenario)[0]


def test_bootstrap_and_incremental_chunks():
    pdws = sample()
    tracker = StreamingDeinterleaver(bootstrap_pdws=600)
    assert tracker.process_chunk(pdws[:100]) == {}
    assert not tracker.bootstrapped
    first = tracker.process_chunk(pdws[100:600])
    assert tracker.bootstrapped
    assert first
    later = tracker.process_chunk(pdws[600:700])
    assert set(later).isdisjoint(first)
    assert len(tracker.tracks) >= 1
    assert all(len(t.pdw_ids) == len(set(t.pdw_ids)) for t in tracker.tracks.values())
    assert set(tracker.assignments).isdisjoint(tracker.ambiguous_pdw_ids)
    assert set(tracker.assignments).isdisjoint(tracker.unexplained_pdw_ids)


def test_rejects_out_of_order():
    pdws = sample()
    tracker = StreamingDeinterleaver()
    tracker.process_chunk(pdws[:2])
    with pytest.raises(ValueError):
        tracker.process_chunk([pdws[0]])


def test_empty_and_invalid_configuration():
    tracker = StreamingDeinterleaver()
    assert tracker.process_chunk([]) == {}
    assert not tracker.bootstrapped
    with pytest.raises(ValueError):
        StreamingDeinterleaver(bootstrap_pdws=10)
