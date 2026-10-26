"""First-stage tests of downstream emitter tracking with controlled observations."""
from testbed.imm_emitter_tracker import IMMEmitterTracker
from testbed.pulse_train_observation import PulseTrainObservation


def observation(name, start, pri_us=1000, rf_ghz=9.4, aoa=45,
                intervals=None, frequencies=None):
    intervals = intervals or [pri_us] * 12
    times = [start]
    for interval in intervals:
        times.append(times[-1] + interval * 1e-6)
    freqs = frequencies or [rf_ghz * 1e9] * len(times)
    return PulseTrainObservation(
        schema_version=1, train_id=name, start_toa_s=times[0],
        end_toa_s=times[-1], pri_s=sum(intervals) / len(intervals) * 1e-6,
        pri_sigma_s=2e-6, frequency_hz=sum(freqs) / len(freqs),
        frequency_sigma_hz=100_000, aoa_deg=aoa, aoa_sigma_deg=0.3,
        pulse_width_s=10e-6, pulse_width_sigma_s=0.1e-6,
        pulse_count=len(times), pdw_ids=tuple(range(len(times))),
        toa_sequence_s=tuple(times), frequency_sequence_hz=tuple(freqs),
        pri_intervals_s=tuple(x * 1e-6 for x in intervals),
    )


def test_pri_change_keeps_identity():
    t = IMMEmitterTracker()
    assert t.process(observation("a", 0)) == "E1"
    assert t.process(observation("b", 0.1, pri_us=750)) == "E1"
    assert t.process(observation("c", 0.2, pri_us=1000)) == "E1"
    assert len(t.tracks) == 1


def test_frequency_switch_keeps_identity():
    t = IMMEmitterTracker()
    assert t.process(observation("a", 0)) == "E1"
    assert t.process(observation("b", 0.1, rf_ghz=9.45)) == "E1"


def test_joint_change_keeps_identity():
    t = IMMEmitterTracker()
    assert t.process(observation("a", 0)) == "E1"
    assert t.process(observation("b", 0.1, pri_us=750, rf_ghz=9.45)) == "E1"


def test_stagger_model_prefers_stagger_over_jitter():
    t = IMMEmitterTracker()
    t.process(observation("a", 0))
    stagger = [900, 1000, 1100] * 8
    t.process(observation("b", 0.1, intervals=stagger))
    probabilities = t.tracks["E1"].snapshot()["pri_models"]
    assert probabilities["stagger"] > probabilities["jitter"]


def test_new_spatially_separated_emitter_gets_new_id():
    t = IMMEmitterTracker()
    assert t.process(observation("a", 0, aoa=45)) == "E1"
    assert t.process(observation("b", 0.1, aoa=65)) == "E2"


def test_same_bearing_simultaneous_trains_not_forced_together():
    t = IMMEmitterTracker()
    assert t.process(observation("a", 0)) == "E1"
    # Overlapping windows are not forcibly merged by this baseline.
    assert t.process(observation("b", 0.005, pri_us=750)) == "E2"


def test_ambiguous_identity_withheld():
    t = IMMEmitterTracker()
    t.process(observation("a", 0, aoa=44))
    t.process(observation("b", 0.005, aoa=46))
    assert t.process(observation("c", 0.1, aoa=45)) is None
    assert t.snapshot()["pending_train_ids"] == ["c"]


def test_duplicate_observation_rejected():
    import pytest
    t = IMMEmitterTracker()
    x = observation("a", 0)
    t.process(x)
    with pytest.raises(ValueError):
        t.process(x)
