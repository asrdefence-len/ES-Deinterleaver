"""Public data types for the sequential tracker.

All quantities use SI units except angle in degrees and amplitude in dB.
Ground-truth labels deliberately do not appear in :class:`PDW`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PDW:
    pdw_id: int
    toa_s: float
    frequency_hz: float
    pulse_width_s: float
    aoa_deg: float
    amplitude_db: float


@dataclass(frozen=True)
class AssociationDecision:
    pdw_id: int
    toa_s: float
    outcome: str
    track_id: str | None
    posterior: Mapping[str, float]
    component_likelihoods: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: str
    pulse_count: int
    first_toa_s: float
    last_toa_s: float
    aoa_deg: float
    aoa_sigma_deg: float
    pulse_width_s: float
    pulse_width_log_sigma: float
    pri_s: float | None
    pri_sigma_s: float | None
    pri_support: int
    rf_modes_hz: tuple[float, ...]
    last_rf_mode: int
