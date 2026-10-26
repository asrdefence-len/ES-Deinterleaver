"""Regression checks for shared association-history ancestry."""
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker, history_tuples


def test_shared_history_and_snapshot_are_consistent():
    t=JointTimingTracker([(0.,.001,1e-6,.2e-6),
                          (.0003,.001,1e-6,.2e-6)],beam_width=8)
    for toa in (0.,.0003,.001,.0013,.002,.0023):
        s=t.process(toa,20.)
        assert len(s["best_assignments"])==len(s["best_pulse_indices"])
        assert s["best_assignments"]==t.hypotheses[0].assignments
        assert s["best_pulse_indices"]==t.hypotheses[0].indices
        assert sum(s["association_probabilities"])==pytest.approx(1.)
        for h in t.hypotheses:
            assert history_tuples(h.history)==(h.assignments,h.indices)


def test_ancestry_is_shared_between_sibling_branches():
    t=JointTimingTracker([(0.,.001,1e-6,.2e-6),
                          (.0003,.001,1e-6,.2e-6)],beam_width=128)
    t.process(0.,20.)
    parents={id(h.history) for h in t.hypotheses}
    t.process(.0003,20.)
    # Different candidate branches can refer to the same preceding history
    ancestry=[h.history.parent for h in t.hypotheses]
    assert len({id(p) for p in ancestry}) < len(ancestry)
    assert all(id(p) in parents for p in ancestry)
