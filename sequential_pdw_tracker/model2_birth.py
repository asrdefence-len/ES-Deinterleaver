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
        return len(self._tracks)

    def _aoa_error(self, track: _InitiatedTrack, pdw: PDW) -> float:
        mean = track.model.snapshot()["aoa"]["mean_deg"]
        return abs(_angular_error_deg(pdw.aoa_deg, float(mean)))

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
        )
        self._next_track_number += 1
        self._tracks.append(track)
        return track

    def _update_track(self, track: _InitiatedTrack, pdw: PDW) -> bool:
        predictive = track.model.predictive_evidence(pdw).total_log_evidence
        update = track.model.process(pdw)
        track.predictive_scores.append(predictive)
        track.toa_history_s.append(pdw.toa_s)
        track.active_history.append(update.shadow_active_posterior)
        track.last_toa_s = pdw.toa_s
        track.pulse_count += 1
        just_confirmed = False
        if (not track.confirmed
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

    def process(self, pdw: PDW) -> Model2BirthDecision:
        confirmed_margins = {}
        timing_boosts = {}
        for track in self._tracks:
            if not track.confirmed:
                continue
            if track.threshold is None:
                raise RuntimeError("confirmed track is missing its threshold")
            evidence = track.model.predictive_evidence(pdw).total_log_evidence
            boost = 0.0
            if (self.birth_config.enable_timing_after_confirmation
                    and track.timing_fit is not None):
                boost = positive_phase_timing_support(
                    pdw.toa_s,
                    track.last_toa_s,
                    track.timing_fit,
                    self.association_config,
                ).applied_boost
            confirmed_margins[track.track_id] = evidence - track.threshold + boost
            timing_boosts[track.track_id] = boost

        if confirmed_margins:
            winner_id = max(confirmed_margins, key=confirmed_margins.get)
            if confirmed_margins[winner_id] >= 0.0:
                track = next(item for item in self._tracks
                             if item.track_id == winner_id)
                self._update_track(track, pdw)
                return Model2BirthDecision(
                    pdw.pdw_id, pdw.toa_s, "confirmed_update", track.track_id,
                    True, confirmed_margins, timing_boosts, {}, None,
                )

        tentative_errors = {
            track.track_id: self._aoa_error(track, pdw)
            for track in self._tracks if not track.confirmed
        }
        eligible = {
            track_id: error for track_id, error in tentative_errors.items()
            if error <= self.birth_config.tentative_aoa_gate_deg
        }
        if eligible:
            winner_id = min(eligible, key=eligible.get)
            track = next(item for item in self._tracks
                         if item.track_id == winner_id)
            just_confirmed = self._update_track(track, pdw)
            return Model2BirthDecision(
                pdw.pdw_id,
                pdw.toa_s,
                "confirmed" if just_confirmed else "tentative_update",
                track.track_id,
                track.confirmed,
                confirmed_margins,
                timing_boosts,
                tentative_errors,
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
                "confirmed_aoa_birth_suppression",
            )
        if len(self._tracks) >= self.birth_config.max_tracks:
            return Model2BirthDecision(
                pdw.pdw_id, pdw.toa_s, "rejected", None, False,
                confirmed_margins, timing_boosts, tentative_errors,
                "track_capacity",
            )

        track = self._create_track(pdw)
        return Model2BirthDecision(
            pdw.pdw_id, pdw.toa_s, "birth", track.track_id, False,
            confirmed_margins, timing_boosts, tentative_errors, None,
        )

    def snapshots(self) -> list[dict]:
        rows = []
        for track in self._tracks:
            rows.append({
                "track_id": track.track_id,
                "status": "confirmed" if track.confirmed else "tentative",
                "first_toa_s": track.first_toa_s,
                "last_toa_s": track.last_toa_s,
                "pulse_count": track.pulse_count,
                "threshold": track.threshold,
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
