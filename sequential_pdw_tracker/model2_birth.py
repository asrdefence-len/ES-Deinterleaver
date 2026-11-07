"""Truth-blind staged initiation for Model 2 emitter HMM filters."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .hmm_emitter import _angular_error_deg
from .hmm_emitter_model2 import HMMEmitterModel2, HMMModel2Config
from .model import PDW
from .model2_association import (
    Model2AssociationConfig,
    positive_phase_timing_support,
)
from .phase_timing_shadow import (
    PhaseTimingFit,
    PhaseTimingShadowConfig,
    fit_phase_timing_shadow,
)


@dataclass(frozen=True)
class Model2BirthConfig:
    tentative_aoa_gate_deg: float = 3.0
    confirmed_birth_suppression_gate_deg: float = 3.0
    confirmation_pulses: int = 256
    threshold_calibration_window: int = 128
    calibration_target_recall: float = 0.99
    max_tracks: int = 8
    enable_timing_after_confirmation: bool = True
    enable_safe_confirmed_pruning: bool = False
    enable_staged_birth_validation: bool = False
    tentative_seed_pulses: int = 8
    tentative_freeze_pulses: int = 128
    validation_reference_window: int = 64
    validation_reference_quantile: float = 0.05
    validation_min_pass_fraction: float = 0.90
    validation_robust_sigma: float = 6.0
    validation_pathology_sigma: float = 12.0

    def __post_init__(self) -> None:
        if self.tentative_aoa_gate_deg <= 0.0:
            raise ValueError("tentative AOA gate must be positive")
        if self.confirmed_birth_suppression_gate_deg <= 0.0:
            raise ValueError("birth-suppression AOA gate must be positive")
        if self.confirmation_pulses < 3:
            raise ValueError("confirmation requires at least three pulses")
        if not 2 <= self.threshold_calibration_window < self.confirmation_pulses:
            raise ValueError("calibration window must be smaller than confirmation")
        if not 0.0 < self.calibration_target_recall < 1.0:
            raise ValueError("target recall must be in (0, 1)")
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        if self.enable_staged_birth_validation:
            if not 1 <= self.tentative_seed_pulses < self.tentative_freeze_pulses:
                raise ValueError("tentative seed must precede model freeze")
            if not self.tentative_freeze_pulses < self.confirmation_pulses:
                raise ValueError("tentative freeze must precede confirmation")
            if not 2 <= self.validation_reference_window < self.tentative_freeze_pulses:
                raise ValueError("validation reference window is invalid")
            if not 0.0 < self.validation_reference_quantile < 0.5:
                raise ValueError(
                    "validation reference quantile must be in (0, 0.5)"
                )
            if not 0.0 < self.validation_min_pass_fraction <= 1.0:
                raise ValueError("validation pass fraction must be in (0, 1]")
            if self.validation_robust_sigma <= 0.0:
                raise ValueError("validation robust sigma must be positive")
            if self.validation_pathology_sigma <= self.validation_robust_sigma:
                raise ValueError("pathology sigma must exceed robust sigma")


@dataclass
class _InitiatedTrack:
    track_id: str
    model: HMMEmitterModel2
    first_toa_s: float
    last_toa_s: float
    pulse_count: int
    confirmed: bool = False
    threshold: float | None = None
    timing_fit: PhaseTimingFit | None = None
    predictive_scores: list[float] = field(default_factory=list)
    toa_history_s: list[float] = field(default_factory=list)
    active_history: list[float] = field(default_factory=list)
    aoa_mean_deg: float = 0.0
    predictive_component_scores: list[tuple[float, float, float]] = field(
        default_factory=list,
    )
    validation_component_scores: list[tuple[float, float, float]] = field(
        default_factory=list,
    )
    validation_reference_floors: tuple[float, float, float] | None = None
    validation_reference_raw_floors: tuple[float, float, float] | None = None
    validation_reference_pathological: tuple[bool, bool, bool] = (
        False, False, False,
    )
    validation_status: str = "not_started"
    validation_pass_fraction: float | None = None
    validation_component_pass_fractions: tuple[float, float, float] | None = None
    validation_raw_threshold: float | None = None
    validation_robust_threshold: float | None = None
    validation_pathological_threshold: bool = False


@dataclass(frozen=True)
class Model2BirthDecision:
    pdw_id: int
    toa_s: float
    outcome: str
    track_id: str | None
    track_confirmed_after_update: bool
    confirmed_margins: dict[str, float]
    timing_boosts: dict[str, float]
    tentative_aoa_errors_deg: dict[str, float]
    confirmed_score_upper_bound_pruned: tuple[str, ...]
    rejection_reason: str | None


class Model2AutomaticBirthTracker:
    """Initiate HMMs without truth, then use frozen Model 2 competition.

    Tentative tracks deliberately use only circular AOA continuity while their
    RF and PW state banks are open for discovery. After enough pulses, a track
    freezes a threshold from its own recent predictive evidence and joins the
    validated Model 2 competition. A rejected pulse close in AOA to a confirmed
    track is not allowed to create a duplicate identity.
    """

    def __init__(
        self,
        birth_config: Model2BirthConfig | None = None,
        model_config: HMMModel2Config | None = None,
        association_config: Model2AssociationConfig | None = None,
        timing_config: PhaseTimingShadowConfig | None = None,
    ):
        self.birth_config = birth_config or Model2BirthConfig()
        self.model_config = model_config or HMMModel2Config()
        self.association_config = association_config or Model2AssociationConfig()
        self.timing_config = timing_config or PhaseTimingShadowConfig()
        self._tracks: list[_InitiatedTrack] = []
        self._next_track_number = 1

    @property
    def track_count(self) -> int:
        return sum(track.validation_status != "failed" for track in self._tracks)

    @property
    def historical_track_count(self) -> int:
        return len(self._tracks)

    def _aoa_error(self, track: _InitiatedTrack, pdw: PDW) -> float:
        return abs(_angular_error_deg(pdw.aoa_deg, track.aoa_mean_deg))

    def _fit_timing(self, track: _InitiatedTrack) -> PhaseTimingFit | None:
        toa_s = np.asarray(track.toa_history_s, dtype=float)
        active = np.asarray(track.active_history, dtype=float)
        gaps_s = np.diff(toa_s)
        mask = (
            (gaps_s > 0.0)
            & (active[1:] >= self.timing_config.active_posterior_threshold)
        )
        if int(mask.sum()) < self.timing_config.min_training_gaps:
            return None
        try:
            fit, _periods, _scores = fit_phase_timing_shadow(
                gaps_s[mask], self.timing_config,
            )
        except ValueError:
            return None
        return fit

    def _create_track(self, pdw: PDW) -> _InitiatedTrack:
        model = HMMEmitterModel2(self.model_config)
        update = model.process(pdw)
        track = _InitiatedTrack(
            track_id=f"T{self._next_track_number}",
            model=model,
            first_toa_s=pdw.toa_s,
            last_toa_s=pdw.toa_s,
            pulse_count=1,
            toa_history_s=[pdw.toa_s],
            active_history=[update.shadow_active_posterior],
            aoa_mean_deg=update.aoa_mean_deg,
        )
        self._next_track_number += 1
        self._tracks.append(track)
        return track

    def _update_track(self, track: _InitiatedTrack, pdw: PDW) -> bool:
        evidence = track.model.predictive_evidence(pdw)
        predictive = evidence.total_log_evidence
        update = track.model.process(pdw)
        track.predictive_scores.append(predictive)
        track.predictive_component_scores.append((
            evidence.rf_log_evidence,
            evidence.aoa_log_evidence,
            evidence.pulse_width_log_evidence,
        ))
        track.toa_history_s.append(pdw.toa_s)
        track.active_history.append(update.shadow_active_posterior)
        track.aoa_mean_deg = update.aoa_mean_deg
        track.last_toa_s = pdw.toa_s
        track.pulse_count += 1
        just_confirmed = False
        if (self.birth_config.enable_staged_birth_validation
                and track.pulse_count == self.birth_config.tentative_freeze_pulses):
            reference = np.asarray(
                track.predictive_component_scores[
                    -self.birth_config.validation_reference_window:
                ],
                dtype=float,
            )
            raw_floors = np.quantile(
                reference,
                self.birth_config.validation_reference_quantile,
                axis=0,
            )
            robust_floors = []
            pathological = []
            for component_index in range(reference.shape[1]):
                robust, median, scale = self._robust_lower_threshold(
                    reference[:, component_index],
                    self.birth_config.validation_robust_sigma,
                )
                robust_floors.append(robust)
                pathological.append(bool(
                    raw_floors[component_index]
                    < median
                    - self.birth_config.validation_pathology_sigma * scale
                ))
            track.validation_reference_raw_floors = tuple(
                float(value) for value in raw_floors
            )
            track.validation_reference_floors = tuple(
                float(value) for value in np.maximum(
                    raw_floors,
                    np.asarray(robust_floors),
                )
            )
            track.validation_reference_pathological = tuple(pathological)
            track.validation_status = "collecting"
        if (not self.birth_config.enable_staged_birth_validation
                and not track.confirmed
                and track.pulse_count >= self.birth_config.confirmation_pulses):
            calibration = np.asarray(
                track.predictive_scores[
                    -self.birth_config.threshold_calibration_window:
                ],
                dtype=float,
            )
            track.threshold = float(np.quantile(
                calibration,
                1.0 - self.birth_config.calibration_target_recall,
            ))
            track.confirmed = True
            if self.birth_config.enable_timing_after_confirmation:
                track.timing_fit = self._fit_timing(track)
            just_confirmed = True
        return just_confirmed

    @staticmethod
    def _robust_lower_threshold(
        values: np.ndarray,
        sigma_multiplier: float,
    ) -> tuple[float, float, float]:
        median = float(np.median(values))
        mad_sigma = 1.4826 * float(np.median(np.abs(values - median)))
        scale = max(mad_sigma, 1.0)
        return median - sigma_multiplier * scale, median, scale

    def _update_validation_track(
        self,
        track: _InitiatedTrack,
        pdw: PDW,
    ) -> tuple[bool, bool]:
        """Score held-out data without adapting the frozen candidate model."""
        if track.validation_reference_floors is None:
            raise RuntimeError("validation candidate has no training reference")
        evidence = track.model.predictive_evidence(pdw)
        components = (
            evidence.rf_log_evidence,
            evidence.aoa_log_evidence,
            evidence.pulse_width_log_evidence,
        )
        track.validation_component_scores.append(components)
        validation_window = (
            self.birth_config.confirmation_pulses
            - self.birth_config.tentative_freeze_pulses
        )
        if len(track.validation_component_scores) > validation_window:
            track.validation_component_scores.pop(0)
        track.last_toa_s = pdw.toa_s
        track.pulse_count += 1
        if len(track.validation_component_scores) < validation_window:
            return False, False

        validation = np.asarray(track.validation_component_scores, dtype=float)
        floors = np.asarray(track.validation_reference_floors, dtype=float)
        component_pass_fractions_array = np.mean(
            validation >= floors,
            axis=0,
        )
        component_pass_fractions = tuple(
            float(value) for value in component_pass_fractions_array
        )
        pass_fraction = float(np.mean(np.all(validation >= floors, axis=1)))
        totals = validation.sum(axis=1)
        raw_threshold = float(np.quantile(
            totals,
            1.0 - self.birth_config.calibration_target_recall,
        ))
        robust_threshold, median, scale = self._robust_lower_threshold(
            totals,
            self.birth_config.validation_robust_sigma,
        )
        pathological = (
            raw_threshold
            < median - self.birth_config.validation_pathology_sigma * scale
        )
        track.validation_pass_fraction = pass_fraction
        track.validation_component_pass_fractions = component_pass_fractions
        track.validation_raw_threshold = raw_threshold
        track.validation_robust_threshold = robust_threshold
        track.validation_pathological_threshold = pathological
        identity_component_indices = (0, 1, 2)
        reference_pathological = any(
            track.validation_reference_pathological
        )
        passed = (
            min(
                component_pass_fractions[index]
                for index in identity_component_indices
            )
            >= self.birth_config.validation_min_pass_fraction
            and not reference_pathological
        )
        if reference_pathological:
            track.validation_status = "failed"
            return False, True
        if pathological or not passed:
            return False, False

        track.threshold = max(raw_threshold, robust_threshold)
        track.confirmed = True
        track.validation_status = "passed"
        if self.birth_config.enable_timing_after_confirmation:
            track.timing_fit = self._fit_timing(track)
        return True, False

    def _tentative_match_rank(
        self,
        track: _InitiatedTrack,
        pdw: PDW,
    ) -> tuple[float, float] | None:
        if track.confirmed or track.validation_status == "failed":
            return None
        aoa_error = self._aoa_error(track, pdw)
        if aoa_error > self.birth_config.tentative_aoa_gate_deg:
            return None
        aoa_normalized = aoa_error / self.birth_config.tentative_aoa_gate_deg
        if (not self.birth_config.enable_staged_birth_validation
                or track.pulse_count < self.birth_config.tentative_seed_pulses):
            return 0.0, aoa_normalized
        novelty = track.model.feature_novelty(pdw)
        known_count = int(novelty.rf_known) + int(novelty.pulse_width_known)
        if known_count == 0:
            return None
        distance = (
            min(novelty.rf_normalized_distance, 1.0)
            + min(novelty.pulse_width_normalized_distance, 1.0)
            + aoa_normalized
        )
        return -float(known_count), distance

    def process(self, pdw: PDW) -> Model2BirthDecision:
        confirmed_margins = {}
        timing_boosts = {}
        upper_bound_pruned = []
        for track in self._tracks:
            if not track.confirmed:
                continue
            if track.threshold is None:
                raise RuntimeError("confirmed track is missing its threshold")
            boost = 0.0
            if (self.birth_config.enable_timing_after_confirmation
                    and track.timing_fit is not None):
                boost = positive_phase_timing_support(
                    pdw.toa_s,
                    track.last_toa_s,
                    track.timing_fit,
                    self.association_config,
                ).applied_boost
            timing_boosts[track.track_id] = boost
            if self.birth_config.enable_safe_confirmed_pruning:
                upper_margin = (
                    track.model.predictive_aoa_log_evidence(pdw)
                    + track.model.predictive_rf_pw_log_evidence_upper_bound()
                    - track.threshold
                    + boost
                )
                if upper_margin < 0.0:
                    confirmed_margins[track.track_id] = upper_margin
                    upper_bound_pruned.append(track.track_id)
                    continue
            evidence = track.model.predictive_evidence(pdw).total_log_evidence
            confirmed_margins[track.track_id] = evidence - track.threshold + boost

        if confirmed_margins:
            winner_id = max(confirmed_margins, key=confirmed_margins.get)
            if confirmed_margins[winner_id] >= 0.0:
                track = next(item for item in self._tracks
                             if item.track_id == winner_id)
                self._update_track(track, pdw)
                return Model2BirthDecision(
                    pdw.pdw_id, pdw.toa_s, "confirmed_update", track.track_id,
                    True, confirmed_margins, timing_boosts, {},
                    tuple(upper_bound_pruned), None,
                )

        tentative_errors = {
            track.track_id: self._aoa_error(track, pdw)
            for track in self._tracks
            if not track.confirmed and track.validation_status != "failed"
        }
        eligible = {}
        for track in self._tracks:
            rank = self._tentative_match_rank(track, pdw)
            if rank is not None:
                eligible[track.track_id] = rank
        if eligible:
            winner_id = min(eligible, key=lambda track_id: eligible[track_id])
            track = next(item for item in self._tracks
                         if item.track_id == winner_id)
            if (self.birth_config.enable_staged_birth_validation
                    and track.validation_status == "collecting"):
                just_confirmed, validation_failed = (
                    self._update_validation_track(track, pdw)
                )
            else:
                just_confirmed = self._update_track(track, pdw)
                validation_failed = False
            return Model2BirthDecision(
                pdw.pdw_id,
                pdw.toa_s,
                (
                    "confirmed" if just_confirmed
                    else "validation_rejected" if validation_failed
                    else "tentative_update"
                ),
                track.track_id,
                track.confirmed,
                confirmed_margins,
                timing_boosts,
                tentative_errors,
                tuple(upper_bound_pruned),
                None,
            )

        near_confirmed = any(
            self._aoa_error(track, pdw)
            <= self.birth_config.confirmed_birth_suppression_gate_deg
            for track in self._tracks if track.confirmed
        )
        if near_confirmed:
            return Model2BirthDecision(
                pdw.pdw_id, pdw.toa_s, "rejected", None, False,
                confirmed_margins, timing_boosts, tentative_errors,
                tuple(upper_bound_pruned),
                "confirmed_aoa_birth_suppression",
            )
        if self.track_count >= self.birth_config.max_tracks:
            return Model2BirthDecision(
                pdw.pdw_id, pdw.toa_s, "rejected", None, False,
                confirmed_margins, timing_boosts, tentative_errors,
                tuple(upper_bound_pruned),
                "track_capacity",
            )

        track = self._create_track(pdw)
        return Model2BirthDecision(
            pdw.pdw_id, pdw.toa_s, "birth", track.track_id, False,
            confirmed_margins, timing_boosts, tentative_errors,
            tuple(upper_bound_pruned), None,
        )

    def snapshots(self) -> list[dict]:
        rows = []
        for track in self._tracks:
            rows.append({
                "track_id": track.track_id,
                "status": (
                    "confirmed" if track.confirmed
                    else "rejected_birth" if track.validation_status == "failed"
                    else "tentative"
                ),
                "first_toa_s": track.first_toa_s,
                "last_toa_s": track.last_toa_s,
                "pulse_count": track.pulse_count,
                "threshold": track.threshold,
                "birth_validation": {
                    "enabled": self.birth_config.enable_staged_birth_validation,
                    "status": track.validation_status,
                    "training_updates": min(
                        track.pulse_count,
                        self.birth_config.tentative_freeze_pulses,
                    ),
                    "held_out_scores": len(
                        track.validation_component_scores
                    ),
                    "reference_component_floors": (
                        track.validation_reference_floors
                    ),
                    "reference_raw_component_floors": (
                        track.validation_reference_raw_floors
                    ),
                    "reference_pathological_components": (
                        track.validation_reference_pathological
                    ),
                    "pass_fraction": track.validation_pass_fraction,
                    "component_pass_fractions": (
                        track.validation_component_pass_fractions
                    ),
                    "required_pass_fraction": (
                        self.birth_config.validation_min_pass_fraction
                    ),
                    "confirmation_components": (
                        "rf_frozen_evidence",
                        "aoa_gate_and_frozen_evidence",
                        "pulse_width_frozen_evidence",
                    ),
                    "raw_threshold": track.validation_raw_threshold,
                    "robust_threshold": track.validation_robust_threshold,
                    "pathological_threshold": (
                        track.validation_pathological_threshold
                    ),
                },
                "timing_fit": (
                    None if track.timing_fit is None else {
                        "period_s": track.timing_fit.period_s,
                        "sigma_s": track.timing_fit.sigma_s,
                        "detection_probability": (
                            track.timing_fit.detection_probability
                        ),
                        "training_score": track.timing_fit.training_score,
                        "training_active_gaps": (
                            track.timing_fit.training_active_gaps
                        ),
                        "training_scored_gaps": (
                            track.timing_fit.training_scored_gaps
                        ),
                    }
                ),
                "model": track.model.snapshot(),
            })
        return rows
