"""Mutually exclusive competition between trained Model 2 emitter filters."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .hmm_emitter_model2 import HMMEmitterModel2
from .model import PDW
from .model2_association import (
    Model2AssociationConfig,
    TimingSupport,
    positive_phase_timing_support,
)
from .phase_timing_shadow import PhaseTimingFit


@dataclass(frozen=True)
class CompetitiveTrackScore:
    rf_log_evidence: float
    aoa_log_evidence: float
    pulse_width_log_evidence: float
    model2_log_evidence: float
    acceptance_threshold: float
    model2_margin: float
    timing_scoreable: bool
    timing_raw_log_likelihood_ratio: float | None
    timing_applied_boost: float
    combined_margin: float
    normalized_hypothesis_weight: float


@dataclass(frozen=True)
class Model2CompetitionDecision:
    pdw_id: int
    toa_s: float
    selected_track: str | None
    neither_weight: float
    track_scores: Mapping[str, CompetitiveTrackScore]


class Model2CompetitiveAssociation:
    """Score one pulse against every track, then update at most one filter.

    Each track's Model 2 log evidence is centred by its independently frozen
    target-recall threshold. The resulting margins and a zero-margin ``neither``
    hypothesis are softmax-normalized for inspection. These are relative
    hypothesis weights, not calibrated posterior probabilities.
    """

    def __init__(
        self,
        models: Mapping[str, HMMEmitterModel2],
        *,
        thresholds: Mapping[str, float],
        last_accepted_toa_s: Mapping[str, float],
        timing_fits: Mapping[str, PhaseTimingFit] | None = None,
        config: Model2AssociationConfig | None = None,
    ):
        if len(models) < 2:
            raise ValueError("competitive association requires at least two tracks")
        keys = set(models)
        if set(thresholds) != keys or set(last_accepted_toa_s) != keys:
            raise ValueError("models, thresholds and last-accepted times must match")
        if timing_fits is not None and set(timing_fits) != keys:
            raise ValueError("timing fits must be supplied for every track")
        self.models = dict(models)
        self.thresholds = {key: float(thresholds[key]) for key in models}
        self.last_accepted_toa_s = {
            key: float(last_accepted_toa_s[key]) for key in models
        }
        self.timing_fits = None if timing_fits is None else dict(timing_fits)
        self.config = config or Model2AssociationConfig()
        self.selected_counts = {key: 0 for key in models}
        self.neither_count = 0

    def process(self, pdw: PDW) -> Model2CompetitionDecision:
        provisional = {}
        margins = []
        for track_id, model in self.models.items():
            evidence = model.predictive_evidence(pdw)
            timing = (
                TimingSupport(False, None, None, None, 0.0)
                if self.timing_fits is None
                else positive_phase_timing_support(
                    pdw.toa_s,
                    self.last_accepted_toa_s[track_id],
                    self.timing_fits[track_id],
                    self.config,
                )
            )
            margin = evidence.total_log_evidence - self.thresholds[track_id]
            combined = margin + timing.applied_boost
            provisional[track_id] = (evidence, timing, margin, combined)
            margins.append(combined)

        log_weights = [0.0, *margins]
        maximum = max(log_weights)
        weights = [math.exp(value - maximum) for value in log_weights]
        normalizer = sum(weights)
        weights = [value / normalizer for value in weights]
        neither_weight = weights[0]

        best_track = max(
            self.models,
            key=lambda track_id: provisional[track_id][3],
        )
        selected_track = (
            best_track if provisional[best_track][3] >= 0.0 else None
        )

        track_scores = {}
        for index, track_id in enumerate(self.models, start=1):
            evidence, timing, margin, combined = provisional[track_id]
            track_scores[track_id] = CompetitiveTrackScore(
                rf_log_evidence=evidence.rf_log_evidence,
                aoa_log_evidence=evidence.aoa_log_evidence,
                pulse_width_log_evidence=evidence.pulse_width_log_evidence,
                model2_log_evidence=evidence.total_log_evidence,
                acceptance_threshold=self.thresholds[track_id],
                model2_margin=margin,
                timing_scoreable=timing.scoreable,
                timing_raw_log_likelihood_ratio=timing.raw_log_likelihood_ratio,
                timing_applied_boost=timing.applied_boost,
                combined_margin=combined,
                normalized_hypothesis_weight=weights[index],
            )

        if selected_track is None:
            self.neither_count += 1
        else:
            self.models[selected_track].process(pdw)
            self.last_accepted_toa_s[selected_track] = pdw.toa_s
            self.selected_counts[selected_track] += 1

        return Model2CompetitionDecision(
            pdw_id=pdw.pdw_id,
            toa_s=pdw.toa_s,
            selected_track=selected_track,
            neither_weight=neither_weight,
            track_scores=track_scores,
        )
