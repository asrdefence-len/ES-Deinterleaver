"""Stage A observation-time semantics; deterministic and receiver-idealised."""
import math
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker


def tracker(pd=.7):
    return JointTimingTracker(
        [(100e-6, .001, .01e-6, .001e-6),
         (102e-6, .001, .01e-6, .001e-6)],
        beam_width=8, max_skip=2,
        observation_detection_probability=pd, deadline_sigmas=3.)


def test_idle_watermark_counts_each_missing_slot_once():
    t = tracker()
    t.advance_time(500e-6)
    assert all(h.accounted_indices == (0, 0) for h in t.hypotheses)
    weights = [h.log_weight for h in t.hypotheses]
    t.advance_time(500e-6)
    assert [h.log_weight for h in t.hypotheses] == pytest.approx(weights)
    t.advance_time(900e-6)
    assert all(h.accounted_indices == (0, 0) for h in t.hypotheses)


def test_close_pulses_not_prematurely_declared_missing():
    t = tracker()
    t.process(100e-6, 30.)
    t.advance_time(101e-6)
    assert t.hypotheses[0].accounted_indices == (0, -1)
    t.advance_time(103e-6)
    assert t.hypotheses[0].accounted_indices == (0, 0)


def test_future_pdw_and_late_pdw_rejection():
    t = tracker()
    t.advance_time(99e-6)
    with pytest.raises(ValueError, match="watermark"):
        t.process(98e-6, 20.)
    t.process(100e-6, 30.)
    with pytest.raises(ValueError, match="latest PDW"):
        t.advance_time(99.5e-6)


def test_advance_without_any_pdws():
    t = tracker()
    t.advance_time(2.5e-3)
    assert t.hypotheses[0].accounted_indices == (2, 2)
    with pytest.raises(ValueError, match="watermark"):
        t.process(2.1e-3, 20.)


def test_models_mutually_exclusive():
    initial=[(100e-6,.001,.01e-6,.001e-6)]*2
    with pytest.raises(ValueError,match="mutually exclusive"):
        JointTimingTracker(initial,observation_detection_probability=.7,
                           joint_detection_probability=.7)
