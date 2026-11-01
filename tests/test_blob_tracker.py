from dataclasses import asdict, replace

import numpy as np
import pytest

from testbed.blob_observation import BlobObservation, summarize_blobs
from testbed.blob_tracker import BlobEmitterTracker


def obs(name, frequency=9.4e9, aoa=30., start=0., pw=.3e-6):
    return BlobObservation(name, name.split(":")[0], 0, start, start + .1,
        100, aoa, .5, frequency - 1e6, frequency, frequency + 1e6,
        pw - .05e-6, pw, pw + .05e-6)


def test_overlapping_rf_modes_retain_same_and_different_emitter_hypotheses():
    tracker = BlobEmitterTracker()
    observations = [obs("W0:B9", 10.2e9, start=0.),
                    obs("W0:B12", 10.8e9, start=.001),
                    obs("W0:B11", 10.6e9, start=.002)]
    for o in observations:
        tracker.process(o)
    counts = {len({dict(h.recent_assignments)[o.observation_id] for o in observations})
              for h in tracker.plausible_hypotheses()}
    assert counts == {1, 2, 3}
    assert tracker.association_options("W0:B11")["status"] == "AMBIGUOUS"
    assert any(len(t.rf_modes) == 3 for h in tracker.hypotheses for t in h.tracks)


def test_known_mode_continuity_keeps_identity_across_windows():
    tracker = BlobEmitterTracker()
    tracker.process(obs("W0:B1"))
    options = tracker.process(obs("W1:B9", start=.2))
    assert options == {"status": "PROVISIONAL", "candidate_track_ids": ["E1"]}


def test_different_bearings_do_not_share_track():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a", aoa=10))
    options = tracker.process(obs("b", aoa=50, start=.01))
    assert options["candidate_track_ids"] == ["E2"]


def test_circular_bearing_continuity():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a", aoa=179.8))
    assert tracker.process(obs("b", aoa=-179.8, start=.2))["candidate_track_ids"] == ["E1"]


def test_capacity_does_not_force_incompatible_association():
    tracker = BlobEmitterTracker(max_tracks=1)
    tracker.process(obs("a", aoa=10))
    assert tracker.process(obs("b", aoa=50, start=.01))["status"] == "UNRESOLVED_CAPACITY"


def test_rf_mode_memory_limit_does_not_force_a_new_emitter():
    tracker = BlobEmitterTracker(max_modes=2)
    for i, rf in enumerate((9.4e9, 10.2e9, 10.8e9)):
        tracker.process(obs(f"b{i}", frequency=rf, start=i * .001))
    assert any(len(h.tracks) == 1 for h in tracker.plausible_hypotheses())
    assert all(len(t.rf_modes) <= 2 for h in tracker.hypotheses for t in h.tracks)


def test_state_is_bounded_as_observations_arrive():
    tracker = BlobEmitterTracker(beam_width=4, max_tracks=3, max_modes=2,
        max_match_branches=2, history_limit=6, track_history_limit=3)
    for i in range(100):
        tracker.process(obs(f"W{i}:B0", start=i * .2))
        assert len(tracker.hypotheses) <= 4
        assert len(tracker._recent_ids) <= 6
        for h in tracker.hypotheses:
            assert len(h.tracks) <= 3
            assert len(h.recent_assignments) <= 6
            assert all(len(t.rf_modes) <= 2 and len(t.recent_blob_ids) <= 3 for t in h.tracks)


def test_adapter_retains_membership_and_exports_only_summary_features():
    x = np.array([[10., 9400., .3, 179.9], [12., 9402., .4, -179.9],
                  [14., 9500., .5, 10.], [16., 9501., .6, 11.]])
    names = ["ToA", "Frequency", "PulseWidth", "AoA"]
    labels = np.array([9, 9, 11, -1])
    original = labels.copy()
    observations = summarize_blobs(x, names, labels, "W0")
    assert np.array_equal(labels, original)
    assert [o.pulse_count for o in observations] == [2, 1]
    assert observations[0].frequency_median_hz == 9401e6
    assert observations[0].end_s == pytest.approx(12e-6)
    assert abs(abs(observations[0].aoa_deg) - 180) < .01
    assert not any(k in asdict(observations[0]) for k in
                   ("pdws", "pdw_ids", "toa_sequence", "truth", "emitter_id"))
    tracker = BlobEmitterTracker()
    before = [asdict(o) for o in observations]
    for o in observations:
        tracker.process(o)
    assert [asdict(o) for o in observations] == before


def test_duplicate_or_out_of_order_observations_fail():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a", start=.2))
    with pytest.raises(ValueError, match="Duplicate"):
        tracker.process(obs("a", start=.2))
    with pytest.raises(ValueError, match="start-time"):
        tracker.process(obs("b", start=.1))
    with pytest.raises(TypeError, match="BlobObservation"):
        tracker.process([1, 2, 3])


def test_expired_identity_is_not_reused():
    tracker = BlobEmitterTracker()
    tracker.process(obs("a"))
    assert tracker.process(obs("b", start=2.))["candidate_track_ids"] == ["E2"]


def test_incorrect_or_nonfinite_features_are_rejected():
    with pytest.raises(ValueError):
        replace(obs("a"), aoa_deg=float("nan"))
    with pytest.raises(ValueError):
        replace(obs("a"), frequency_p05_hz=11e9)
