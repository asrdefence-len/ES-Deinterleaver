"""Tests for the streaming-to-tracker interface and downstream identity manager."""
import json
from pathlib import Path
import pytest

from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver
from testbed.pulse_train_observation import (
    PulseTrainObservation, observations_from_streaming,
)
from testbed.emitter_track_manager import EmitterTrackManager


def observation(train_id, start, end, pri, aoa=45.0):
    return PulseTrainObservation(
        schema_version=1, train_id=train_id, start_toa_s=start,
        end_toa_s=end, pri_s=pri, pri_sigma_s=None,
        frequency_hz=9.4e9, frequency_sigma_hz=None,
        aoa_deg=aoa, aoa_sigma_deg=None,
        pulse_width_s=10e-6, pulse_width_sigma_s=None,
        pulse_count=1, pdw_ids=(len(train_id),),
    )


def test_pri_change_preserves_emitter_id_when_other_features_stable():
    manager = EmitterTrackManager()
    assert manager.process(observation("PT1", 0, 0.19, 0.001)) == "E1"
    assert manager.process(observation("PT2", 0.2, 0.39, 0.0005)) == "E1"
    assert manager.process(observation("PT3", 0.4, 0.59, 0.001)) == "E1"
    assert manager.snapshot()["tracks"][0]["train_ids"] == ["PT1", "PT2", "PT3"]


def test_distinct_bearing_and_ambiguous_association():
    manager = EmitterTrackManager(aoa_gate_deg=2)
    assert manager.process(observation("PT1", 0, 0.1, 0.001, 40)) == "E1"
    assert manager.process(observation("PT2", 0, 0.1, 0.001, 50)) == "E2"
    assert manager.process(observation("PT3", 0.11, 0.2, 0.001, 40)) == "E1"
    with pytest.raises(ValueError):
        manager.process(observation("PT3", 0.21, 0.3, 0.001, 40))


def test_ambiguous_match_is_withheld():
    manager = EmitterTrackManager(aoa_gate_deg=5)
    manager.process(observation("PT1", 0, 0.1, 0.001, 44))
    manager.process(observation("PT2", 0, 0.1, 0.001, 48))
    assert manager.process(observation("PT3", 0.11, 0.2, 0.001, 46)) is None
    assert manager.snapshot()["pending_train_ids"] == ["PT3"]


def test_adapter_produces_no_truth_or_persistent_emitter_id():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, truth = generate(scenario)
    processor = StreamingDeinterleaver(bootstrap_pdws=600)
    for start in range(0, len(pdws), 100):
        processor.process_chunk(pdws[start:start + 100])
    observations = observations_from_streaming(processor, pdws)
    assert observations
    for item in observations:
        assert item.pulse_count == len(item.pdw_ids)
        assert set(item.pdw_ids).issubset(truth)
        assert not hasattr(item, "emitter_id")
        assert item.association_confidence is None
