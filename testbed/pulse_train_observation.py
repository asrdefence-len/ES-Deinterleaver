"""Versioned boundary between high-rate pulse processing and downstream tracking.

A PulseTrainObservation describes a *temporary pulse-train hypothesis*, not a
persistent physical emitter. No simulator truth or emitter identity is present.
Times and units are explicit. Uncertainty is optional when not estimated.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PulseTrainObservation:
    schema_version: int
    train_id: str
    start_toa_s: float
    end_toa_s: float
    pri_s: float
    pri_sigma_s: Optional[float]
    frequency_hz: float
    frequency_sigma_hz: Optional[float]
    aoa_deg: float
    aoa_sigma_deg: Optional[float]
    pulse_width_s: float
    pulse_width_sigma_s: Optional[float]
    pulse_count: int
    pdw_ids: Tuple[int, ...]
    association_confidence: Optional[float] = None
    ambiguous_pdw_ids: Tuple[int, ...] = ()
    missed_pulse_count: Optional[int] = None

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported observation schema version")
        if not self.train_id or self.start_toa_s > self.end_toa_s:
            raise ValueError("Invalid pulse-train identity or time interval")
        if self.pri_s <= 0 or self.pulse_width_s <= 0:
            raise ValueError("PRI and pulse width must be positive")
        if self.pulse_count < 1 or self.pulse_count != len(self.pdw_ids):
            raise ValueError("pulse_count must equal the number of PDW references")
        if len(set(self.pdw_ids)) != len(self.pdw_ids):
            raise ValueError("Duplicate PDW references")
        for value in (self.pri_sigma_s, self.frequency_sigma_hz,
                      self.aoa_sigma_deg, self.pulse_width_sigma_s):
            if value is not None and value < 0:
                raise ValueError("Uncertainty must be nonnegative")
        if self.association_confidence is not None and not (0 <= self.association_confidence <= 1):
            raise ValueError("Confidence must be between zero and one")
        if self.missed_pulse_count is not None and self.missed_pulse_count < 0:
            raise ValueError("Missed-pulse count must be nonnegative")


def observations_from_streaming(processor, pdws):
    """Export snapshots of existing streaming pulse trains without reassigning IDs.

    This is an adapter for the *fixed-PRI baseline*, not a new MHT or an
    emitter tracker. Confidence and measurement uncertainty are left unknown
    rather than invented. Only PDWs supplied to this call are available.
    """
    import statistics

    by_id = {p.pdw_id: p for p in pdws}
    output = []
    for train in processor.tracks.values():
        members = [by_id[i] for i in train.pdw_ids if i in by_id]
        if not members:
            continue
        members.sort(key=lambda p: p.toa_s)
        output.append(PulseTrainObservation(
            schema_version=1,
            train_id=train.track_id,
            start_toa_s=members[0].toa_s,
            end_toa_s=members[-1].toa_s,
            pri_s=train.pri_s,
            pri_sigma_s=None,
            frequency_hz=statistics.median(p.frequency_hz for p in members),
            frequency_sigma_hz=None,
            aoa_deg=statistics.median(p.aoa_deg for p in members),
            aoa_sigma_deg=None,
            pulse_width_s=statistics.median(p.pulse_width_s for p in members),
            pulse_width_sigma_s=None,
            pulse_count=len(members),
            pdw_ids=tuple(p.pdw_id for p in members),
            association_confidence=None,
        ))
    return output
