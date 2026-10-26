"""Regression checks for branch-level Stage A diagnostics."""
import math
import pytest
from inspect_joint_detection import inspect_branches
from testbed.joint_bayesian_timing import JointTimingTracker


def make_tracker(guard):
    return JointTimingTracker(
        [(100e-6,.001,.1e-6,.001e-6),
         (102e-6,.001,.1e-6,.001e-6)],
        max_skip=2,beam_width=16,
        joint_detection_probability=.7,joint_guard_s=guard)


def test_guard_defers_other_emitter_slot():
    guarded=inspect_branches(make_tracker(5e-6),100e-6,30.,top=4)
    unguarded=inspect_branches(make_tracker(0.),103e-6,30.,top=4)
    a_guard=[r for r in guarded if r["emitter"]==0 and r["index"]==0][0]
    a_open=[r for r in unguarded if r["emitter"]==0 and r["index"]==0][0]
    assert a_guard["missed_other"] == 0
    assert a_open["missed_other"] == 1
    assert a_open["detection_ll"] == pytest.approx(math.log(.7)+math.log(.3))


def test_branch_log_weight_decomposition():
    rows=inspect_branches(make_tracker(0.),103e-6,30.)
    for r in rows:
        assert r["total"] == pytest.approx(r["timing_ll"]+r["detection_ll"])


def test_instrumentation_does_not_advance_tracker():
    tracker=make_tracker(0.)
    before=tracker.hypotheses
    inspect_branches(tracker,103e-6,30.)
    assert tracker.hypotheses is before
