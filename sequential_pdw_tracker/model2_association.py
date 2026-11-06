"""Controlled one-track association comparison for HMM Model 2.

Two arms receive the same candidate PDWs.  Both use the same trained Model 2
predictive evidence and threshold; the experimental arm adds only non-negative,
bounded phase-timing support.  Neither arm sees evaluation truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .hmm_emitter_model2 import HMMEmitterModel2, Model2PredictiveEvidence
from .model import PDW
from .phase_timing_shadow import PhaseTimingFit


@dataclass(frozen=True)
class Model2AssociationConfig:
    state_training_fraction: float = 0.40
    association_start_fraction: float = 0.60
    calibration_target_recall: float = 0.99
    timing_weight: float = 1.0
    timing_max_boost: float = 0.75
    timing_max_intervals: int = 16

    def __post_init__(self) -> None:
        if not (
            0.0 < self.state_training_fraction
            < self.association_start_fraction
            < 1.0
        ):
            raise ValueError("training fractions must increase within (0, 1)")
        if not 0.0 < self.calibration_target_recall < 1.0:
            raise ValueError("calibration_target_recall must be in (0, 1)")
        if self.timing_weight < 0.0 or self.timing_max_boost <= 0.0:
            raise ValueError("timing weight and cap must be non-negative")
        if self.timing_max_intervals < 1:
            raise ValueError("timing_max_intervals must be positive")


@dataclass(frozen=True)
class TimingSupport:
    scoreable: bool
    intervals: int | None
    residual_s: float | None
    raw_log_likelihood_ratio: float | None
    applied_boost: float


@dataclass(frozen=True)
class Model2AssociationDecision:
    pdw_id: int
    toa_s: float
    accepted: bool
    threshold: float
    baseline_log_evidence: float
    combined_log_evidence: float
    rf_log_evidence: float
    aoa_log_evidence: float
    pulse_width_log_evidence: float
    timing_scoreable: bool
    timing_intervals: int | None
    timing_residual_s: float | None
    timing_raw_log_likelihood_ratio: float | None
    timing_applied_boost: float


def positive_phase_timing_support(
    toa_s: float,
    last_accepted_toa_s: float,
    fit: PhaseTimingFit,
    config: Model2AssociationConfig,
) -> TimingSupport:
    gap_s = toa_s - last_accepted_toa_s
    if gap_s <= 0.0:
        return TimingSupport(False, None, None, None, 0.0)
    intervals = int(round(gap_s / fit.period_s))
    if intervals < 1 or intervals > config.timing_max_intervals:
        return TimingSupport(False, None, None, None, 0.0)
    residual_s = gap_s - intervals * fit.period_s
    raw = (
        math.log(fit.period_s / (math.sqrt(2.0 * math.pi) * fit.sigma_s))
        - 0.5 * (residual_s / fit.sigma_s) ** 2
    )
    boost = min(
        config.timing_max_boost,
        config.timing_weight * max(raw, 0.0),
    )
    return TimingSupport(True, intervals, residual_s, raw, boost)


class Model2SingleTrackAssociationArm:
    """Sequential accept/reject arm around one already trained E46 model."""

    def __init__(
        self,
        model: HMMEmitterModel2,
        *,
        threshold: float,
        last_accepted_toa_s: float,
        timing_fit: PhaseTimingFit | None = None,
        config: Model2AssociationConfig | None = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.last_accepted_toa_s = float(last_accepted_toa_s)
        self.timing_fit = timing_fit
        self.config = config or Model2AssociationConfig()
        self.accepted_count = 0
        self.rejected_count = 0

    def process(self, pdw: PDW) -> Model2AssociationDecision:
        evidence: Model2PredictiveEvidence = self.model.predictive_evidence(pdw)
        timing = (
            TimingSupport(False, None, None, None, 0.0)
            if self.timing_fit is None
            else positive_phase_timing_support(
                pdw.toa_s,
                self.last_accepted_toa_s,
                self.timing_fit,
                self.config,
            )
        )
        combined = evidence.total_log_evidence + timing.applied_boost
        accepted = combined >= self.threshold
        if accepted:
            self.model.process(pdw)
            self.last_accepted_toa_s = pdw.toa_s
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        return Model2AssociationDecision(
            pdw_id=pdw.pdw_id,
            toa_s=pdw.toa_s,
            accepted=accepted,
            threshold=self.threshold,
            baseline_log_evidence=evidence.total_log_evidence,
            combined_log_evidence=combined,
            rf_log_evidence=evidence.rf_log_evidence,
            aoa_log_evidence=evidence.aoa_log_evidence,
            pulse_width_log_evidence=evidence.pulse_width_log_evidence,
            timing_scoreable=timing.scoreable,
            timing_intervals=timing.intervals,
            timing_residual_s=timing.residual_s,
            timing_raw_log_likelihood_ratio=timing.raw_log_likelihood_ratio,
            timing_applied_boost=timing.applied_boost,
        )
