"""Truth-history instrumentation regression."""
from crossing_pulse_stress import trial
from testbed.joint_bayesian_timing import JointTimingTracker


def test_truth_rank_one_for_single_unambiguous_pdw():
    t=JointTimingTracker([(100e-6,.001,.1e-6,.01e-6),
                          (200e-6,.001,.1e-6,.01e-6)],
                         beam_width=8,max_skip=2,
                         observation_detection_probability=.7)
    t.process(100e-6,30.,diagnostic_truth=((0,),(0,)))
    d=t.truth_diagnostic
    assert d["rank_before_pruning"] == 1
    assert d["posterior_mass_before_pruning"] > .99
    assert len(d["branch_terms"]) == 1
    assert d["branch_terms"][0]["missed_selected"] == 0


def test_diagnostic_does_not_change_trial_metrics():
    opts=dict(scenario="very_close",count=12,pd=.7,beam=16,
              observation_detection_probability=.7)
    a=trial(19017,**opts)
    b=trial(19017,**opts,diagnose_truth=True)
    for key in ("accuracy","brier","truth_history_retained","first_truth_pruned"):
        assert a[key] == b[key]


def test_truth_absence_due_to_accounted_pulse():
    t=JointTimingTracker([(100e-6,.001,.1e-6,.01e-6),
                          (200e-6,.001,.1e-6,.01e-6)],
                         beam_width=8,max_skip=2,
                         observation_detection_probability=.7)
    # Before any detections, a watermark irrevocably accounts for index zero.
    t.advance_time(500e-6)
    t.process(1.1e-3,30.,diagnostic_truth=((0,),(0,)))
    d=t.truth_diagnostic
    assert d["rank_before_pruning"] is None
    assert d["absence_reason"] == "already_accounted"
    assert d["matching_parent_count"] == 1
    assert d["matching_parents"][0]["accounted_index"] >= 0


def test_truth_absence_due_to_skip_window():
    t=JointTimingTracker([(100e-6,.001,.1e-6,.01e-6),
                          (200e-6,.001,.1e-6,.01e-6)],
                         beam_width=8,max_skip=2)
    t.process(5.1e-3,30.,diagnostic_truth=((0,),(5,)))
    assert t.truth_diagnostic["absence_reason"] == "beyond_skip_window"
