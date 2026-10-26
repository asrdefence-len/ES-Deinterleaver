"""Correction A: receiver TOA uncertainty in nondetection deadlines."""
import pytest
from testbed.joint_bayesian_timing import JointTimingTracker
from crossing_pulse_stress import trial


def make_tracker(receiver_sigma):
    return JointTimingTracker(
        [(100e-6, .001, .01e-6, .001e-6),
         (100.5e-6, .001, .01e-6, .001e-6)],
        beam_width=8, max_skip=2,
        observation_detection_probability=.7, deadline_sigmas=3.,
        expected_toa_sigma_s=receiver_sigma)


def test_880ns_late_pulse_is_not_prematurely_excluded():
    old=make_tracker(0.)
    corrected=make_tracker(400e-9)
    # A PDW from the other emitter moves observation time beyond the
    # state-only deadline for emitter A's first pulse.
    for t in (old,corrected):
        t.process(100.5e-6,30.)
        t.advance_time(100.5e-6)
    old_b = [h for h in old.hypotheses if h.assignments == (1,) and h.indices == (0,)]
    new_b = [h for h in corrected.hypotheses if h.assignments == (1,) and h.indices == (0,)]
    assert old_b and new_b
    assert all(h.accounted_indices[0] >= 0 for h in old_b)
    assert all(h.accounted_indices[0] == -1 for h in new_b)
    # The A pulse is late by 880 ns but within 3*400 ns.
    corrected.process(100e-6+880e-9,30.,
                      diagnostic_truth=((1,0),(0,0)))
    assert corrected.truth_diagnostic["rank_before_pruning"] is not None


def test_zero_noise_preserves_default_behavior():
    a=make_tracker(0.)
    b=JointTimingTracker(
        [(100e-6,.001,.01e-6,.001e-6),(100.5e-6,.001,.01e-6,.001e-6)],
        beam_width=8,max_skip=2,observation_detection_probability=.7,
        deadline_sigmas=3.)
    a.advance_time(500e-6)
    b.advance_time(500e-6)
    assert a.hypotheses == b.hypotheses


@pytest.mark.parametrize("sigma",[-1.,float("nan"),float("inf")])
def test_invalid_receiver_uncertainty_rejected(sigma):
    with pytest.raises(ValueError,match="expected_toa_sigma_s"):
        make_tracker(sigma)


def test_crossing_trial_accepts_receiver_uncertainty():
    row=trial(19017,count=12,pd=.7,beam=16,
              observation_detection_probability=.7,
              expected_toa_sigma_s=400e-9)
    assert 0 <= row["accuracy"] <= 1
