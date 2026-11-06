import pytest

from sequential_pdw_tracker.hmm_emitter_model2 import HMMEmitterModel2
from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.model2_association import (
    Model2AssociationConfig,
    Model2SingleTrackAssociationArm,
    positive_phase_timing_support,
)
from sequential_pdw_tracker.phase_timing_shadow import PhaseTimingFit


def pdw(index: int, toa_us: float) -> PDW:
    return PDW(
        pdw_id=index,
        toa_s=toa_us * 1e-6,
        frequency_hz=1000e6,
        pulse_width_s=0.10e-6,
        aoa_deg=32.0,
        amplitude_db=-70.0,
    )


def timing_fit() -> PhaseTimingFit:
    return PhaseTimingFit(
        period_s=10e-6,
        sigma_s=1e-6,
        detection_probability=0.5,
        training_score=0.0,
        training_active_gaps=100,
        training_scored_gaps=100,
    )


def test_positive_phase_support_is_neutral_when_bad_and_capped_when_good():
    config = Model2AssociationConfig(
        timing_weight=2.0,
        timing_max_boost=0.4,
        timing_max_intervals=4,
    )
    aligned = positive_phase_timing_support(20e-6, 0.0, timing_fit(), config)
    half_cycle = positive_phase_timing_support(25e-6, 0.0, timing_fit(), config)
    too_far = positive_phase_timing_support(100e-6, 0.0, timing_fit(), config)

    assert aligned.scoreable is True
    assert aligned.raw_log_likelihood_ratio > 0.0
    assert aligned.applied_boost == pytest.approx(0.4)
    assert half_cycle.scoreable is True
    assert half_cycle.raw_log_likelihood_ratio < 0.0
    assert half_cycle.applied_boost == 0.0
    assert too_far.scoreable is False
    assert too_far.applied_boost == 0.0


def test_timing_arm_can_only_rescue_an_aligned_near_threshold_candidate():
    model = HMMEmitterModel2()
    for index in range(80):
        model.process(pdw(index, index * 10))
    candidate = pdw(100, 800)
    baseline_score = model.predictive_evidence(candidate).total_log_evidence
    threshold = baseline_score + 0.2
    config = Model2AssociationConfig(timing_max_boost=0.4)

    baseline = Model2SingleTrackAssociationArm(
        model,
        threshold=threshold,
        last_accepted_toa_s=790e-6,
        config=config,
    ).process(candidate)

    timed_model = HMMEmitterModel2()
    for index in range(80):
        timed_model.process(pdw(index, index * 10))
    timed = Model2SingleTrackAssociationArm(
        timed_model,
        threshold=threshold,
        last_accepted_toa_s=790e-6,
        timing_fit=timing_fit(),
        config=config,
    ).process(candidate)

    assert baseline.accepted is False
    assert baseline.timing_applied_boost == 0.0
    assert timed.accepted is True
    assert 0.0 < timed.timing_applied_boost <= config.timing_max_boost
