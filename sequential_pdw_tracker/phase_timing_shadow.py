"""Held-out phase-coherence evaluation for a detected pulse train.

The fitted model is deliberately diagnostic.  It estimates one dominant
interval lattice from an early training portion, then evaluates later active
gaps without refitting.  It does not assert a complete transmitter PRI model
and it never contributes to association.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class PhaseTimingShadowConfig:
    train_fraction: float = 0.60
    active_posterior_threshold: float = 0.80
    min_period_s: float = 5e-6
    max_period_s: float = 15e-6
    period_step_s: float = 2e-9
    max_intervals: int = 16
    sigma_floor_s: float = 0.15e-6
    min_training_gaps: int = 100
    burst_separation_s: float = 50e-3
    min_reported_segment_gaps: int = 20
    candidate_report_separation_s: float = 0.25e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.active_posterior_threshold < 1.0:
            raise ValueError("active_posterior_threshold must be in (0, 1)")
        if not 0.0 < self.min_period_s < self.max_period_s:
            raise ValueError("invalid phase-period search range")
        if self.period_step_s <= 0.0 or self.max_intervals < 1:
            raise ValueError("invalid phase-period search resolution")
        if (self.sigma_floor_s <= 0.0 or self.min_training_gaps < 4
                or self.burst_separation_s <= 0.0
                or self.min_reported_segment_gaps < 1
                or self.candidate_report_separation_s <= 0.0):
            raise ValueError("invalid phase-noise configuration")


@dataclass(frozen=True)
class PhaseTimingFit:
    period_s: float
    sigma_s: float
    detection_probability: float
    training_score: float
    training_active_gaps: int
    training_scored_gaps: int


def _wrap_phase(residual_s: np.ndarray, period_s: float) -> np.ndarray:
    return (residual_s + 0.5 * period_s) % period_s - 0.5 * period_s


def _fit_candidate(
    gaps_s: np.ndarray,
    period_s: float,
    config: PhaseTimingShadowConfig,
) -> tuple[float, float, float, int]:
    raw_intervals = np.rint(gaps_s / period_s)
    valid = (raw_intervals >= 1.0) & (raw_intervals <= config.max_intervals)
    valid_count = int(valid.sum())
    if valid_count < config.min_training_gaps:
        return -math.inf, math.nan, math.nan, valid_count

    intervals = raw_intervals[valid]
    residual = gaps_s[valid] - intervals * period_s
    sigma_s = max(
        config.sigma_floor_s,
        1.4826 * float(np.median(np.abs(residual))),
    )
    detection_probability = float(np.clip(1.0 / np.mean(intervals), 0.01, 0.99))
    log_likelihood = (
        -0.5 * (LOG_2PI + 2.0 * math.log(sigma_s))
        - 0.5 * (residual / sigma_s) ** 2
        + math.log(detection_probability)
        + (intervals - 1.0) * math.log1p(-detection_probability)
    )
    coverage = valid_count / len(gaps_s)
    score = float(np.mean(log_likelihood) + math.log(max(coverage, 1e-12)))
    return score, sigma_s, detection_probability, valid_count


def fit_phase_timing_shadow(
    gaps_s: np.ndarray,
    config: PhaseTimingShadowConfig | None = None,
) -> tuple[PhaseTimingFit, np.ndarray, np.ndarray]:
    """Fit one dominant interval lattice to already selected active gaps."""
    config = config or PhaseTimingShadowConfig()
    gaps_s = np.asarray(gaps_s, dtype=float)
    if len(gaps_s) < config.min_training_gaps or np.any(gaps_s <= 0.0):
        raise ValueError("insufficient positive training gaps")

    periods = np.arange(
        config.min_period_s,
        config.max_period_s + 0.5 * config.period_step_s,
        config.period_step_s,
    )
    scores = np.empty(len(periods), dtype=float)
    candidate_details: list[tuple[float, float, int]] = []
    for index, period_s in enumerate(periods):
        score, sigma_s, detection_probability, valid_count = _fit_candidate(
            gaps_s, float(period_s), config,
        )
        scores[index] = score
        candidate_details.append((sigma_s, detection_probability, valid_count))

    if not np.isfinite(scores).any():
        raise ValueError("no phase-period candidate had sufficient support")
    best_index = int(np.nanargmax(scores))
    sigma_s, detection_probability, valid_count = candidate_details[best_index]
    fit = PhaseTimingFit(
        period_s=float(periods[best_index]),
        sigma_s=float(sigma_s),
        detection_probability=float(detection_probability),
        training_score=float(scores[best_index]),
        training_active_gaps=len(gaps_s),
        training_scored_gaps=valid_count,
    )
    return fit, periods, scores


def _summary(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _held_out_segments(
    toa_s: np.ndarray,
    residual_s: np.ndarray,
    log_ratio: np.ndarray,
    separation_s: float,
    minimum_gaps: int,
) -> list[dict]:
    if not len(toa_s):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(toa_s) > separation_s) + 1]
    ends = np.r_[starts[1:], len(toa_s)]
    rows = []
    for start, end in zip(starts, ends):
        if end - start < minimum_gaps:
            continue
        burst_residual = residual_s[start:end]
        burst_ratio = log_ratio[start:end]
        rows.append({
            "start_toa_s": float(toa_s[start]),
            "end_toa_s": float(toa_s[end - 1]),
            "scored_gaps": int(end - start),
            "rms_phase_residual_s": float(np.sqrt(np.mean(burst_residual**2))),
            "mean_phase_log_likelihood_ratio": float(np.mean(burst_ratio)),
            "positive_phase_evidence_fraction": float(np.mean(burst_ratio > 0.0)),
        })
    return rows


def analyze_phase_timing_shadow(
    toa_s: np.ndarray,
    active_posterior: np.ndarray,
    config: PhaseTimingShadowConfig | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Train chronologically, then evaluate phase prediction on held-out gaps."""
    config = config or PhaseTimingShadowConfig()
    toa_s = np.asarray(toa_s, dtype=float)
    active_posterior = np.asarray(active_posterior, dtype=float)
    if len(toa_s) < 3 or len(active_posterior) != len(toa_s):
        raise ValueError("TOA and visibility arrays must have equal length >= 3")
    if np.any(np.diff(toa_s) < 0.0):
        raise ValueError("TOAs must be non-decreasing")
    if np.any((active_posterior < 0.0) | (active_posterior > 1.0)):
        raise ValueError("active posterior must be in [0, 1]")

    gaps_s = np.diff(toa_s)
    gap_active = active_posterior[1:]
    split_index = int(math.floor(len(gaps_s) * config.train_fraction))
    indices = np.arange(len(gaps_s))
    positive = gaps_s > 0.0
    active = gap_active >= config.active_posterior_threshold
    training_mask = positive & active & (indices < split_index)
    held_out_active_mask = positive & active & (indices >= split_index)

    fit, candidate_periods, candidate_scores = fit_phase_timing_shadow(
        gaps_s[training_mask], config,
    )
    raw_intervals = np.rint(gaps_s / fit.period_s)
    scoreable = (
        held_out_active_mask
        & (raw_intervals >= 1.0)
        & (raw_intervals <= config.max_intervals)
    )
    intervals = raw_intervals[scoreable].astype(int)
    residual_s = gaps_s[scoreable] - intervals * fit.period_s

    # Conditional phase likelihood ratio: a fitted Gaussian phase residual
    # versus an unrelated arrival uniformly distributed over one phase cycle.
    phase_log_likelihood_ratio = (
        math.log(fit.period_s / (math.sqrt(2.0 * math.pi) * fit.sigma_s))
        - 0.5 * (residual_s / fit.sigma_s) ** 2
    )
    negative_control_residual_s = _wrap_phase(
        residual_s + 0.5 * fit.period_s,
        fit.period_s,
    )
    negative_control_log_likelihood_ratio = (
        math.log(fit.period_s / (math.sqrt(2.0 * math.pi) * fit.sigma_s))
        - 0.5 * (negative_control_residual_s / fit.sigma_s) ** 2
    )

    local_maximum = np.zeros(len(candidate_scores), dtype=bool)
    if len(candidate_scores) >= 3:
        local_maximum[1:-1] = (
            (candidate_scores[1:-1] > candidate_scores[:-2])
            & (candidate_scores[1:-1] >= candidate_scores[2:])
        )
    ordered_candidates = np.flatnonzero(
        local_maximum & np.isfinite(candidate_scores)
    )
    ordered_candidates = ordered_candidates[
        np.argsort(candidate_scores[ordered_candidates])[::-1]
    ]
    candidate_indices = []
    for index in ordered_candidates:
        if all(
            abs(candidate_periods[index] - candidate_periods[selected])
            >= config.candidate_report_separation_s
            for selected in candidate_indices
        ):
            candidate_indices.append(int(index))
        if len(candidate_indices) == 8:
            break
    competing_candidates = [
        {
            "interval_s": float(candidate_periods[index]),
            "objective_relative_to_best": float(
                candidate_scores[index] - fit.training_score
            ),
        }
        for index in candidate_indices
    ]
    held_out_toa_s = toa_s[1:][scoreable]
    held_out_segments = _held_out_segments(
        held_out_toa_s,
        residual_s,
        phase_log_likelihood_ratio,
        config.burst_separation_s,
        config.min_reported_segment_gaps,
    )

    held_out_active_count = int(held_out_active_mask.sum())
    result = {
        "status": (
            "held-out phase-coherence shadow; diagnostic only and excluded "
            "from active association"
        ),
        "feedback_weight": 0.0,
        "transmitter_pri_asserted": False,
        "interpretation": (
            "one dominant detected-train interval lattice; not a complete "
            "waveform timing topology"
        ),
        "split": {
            "training_fraction": config.train_fraction,
            "training_end_gap_index": split_index - 1,
            "held_out_start_gap_index": split_index,
        },
        "fit": {
            "dominant_interval_s": fit.period_s,
            "phase_sigma_s": fit.sigma_s,
            "implied_detection_probability": fit.detection_probability,
            "training_objective": fit.training_score,
            "training_active_gaps": fit.training_active_gaps,
            "training_scored_gaps": fit.training_scored_gaps,
            "competing_local_maxima": competing_candidates,
        },
        "held_out": {
            "active_gaps": held_out_active_count,
            "scored_gaps": int(scoreable.sum()),
            "scoreable_fraction": (
                float(scoreable.sum() / held_out_active_count)
                if held_out_active_count else None
            ),
            "median_absolute_phase_residual_s": (
                float(np.median(np.abs(residual_s))) if len(residual_s) else None
            ),
            "rms_phase_residual_s": (
                float(np.sqrt(np.mean(residual_s**2))) if len(residual_s) else None
            ),
            "within_one_sigma_fraction": (
                float(np.mean(np.abs(residual_s) <= fit.sigma_s))
                if len(residual_s) else None
            ),
            "within_two_sigma_fraction": (
                float(np.mean(np.abs(residual_s) <= 2.0 * fit.sigma_s))
                if len(residual_s) else None
            ),
            "phase_log_likelihood_ratio": {
                **_summary(phase_log_likelihood_ratio),
                "positive_fraction": (
                    float(np.mean(phase_log_likelihood_ratio > 0.0))
                    if len(phase_log_likelihood_ratio) else None
                ),
                "definition": (
                    "fitted Gaussian phase residual versus uniform phase; "
                    "conditioned on an active, scoreable gap"
                ),
            },
            "half_cycle_negative_control_log_likelihood_ratio": {
                **_summary(negative_control_log_likelihood_ratio),
                "positive_fraction": (
                    float(np.mean(negative_control_log_likelihood_ratio > 0.0))
                    if len(negative_control_log_likelihood_ratio) else None
                ),
            },
            "scoreable_segments": held_out_segments,
            "scoreable_segment_definition": (
                "groups separated by at least burst_separation_s; fragments "
                "smaller than min_reported_segment_gaps are omitted"
            ),
        },
        "config": asdict(config),
    }
    diagnostics = {
        "candidate_periods_s": candidate_periods,
        "candidate_scores": candidate_scores,
        "held_out_toa_s": held_out_toa_s,
        "held_out_intervals": intervals,
        "held_out_residual_s": residual_s,
        "held_out_phase_log_likelihood_ratio": phase_log_likelihood_ratio,
        "negative_control_log_likelihood_ratio": negative_control_log_likelihood_ratio,
    }
    return result, diagnostics
