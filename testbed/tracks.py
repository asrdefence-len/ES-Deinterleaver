"""Track state and PRI models; no access to simulator truth."""
from dataclasses import dataclass, field
from enum import Enum
from statistics import median


class PRIType(str, Enum):
    UNKNOWN = "UNKNOWN"
    FIXED = "FIXED"
    JITTERED = "JITTERED"
    STAGGERED = "STAGGERED"
    SWITCHED = "SWITCHED"


@dataclass
class PRIModel:
    kind: PRIType = PRIType.UNKNOWN
    mean_s: float | None = None
    spread_s: float | None = None
    pattern_s: tuple[float, ...] = ()
    confidence: float = 0.0

    def update_from_intervals(self, intervals_s: list[float], relative_fixed_tolerance=0.02):
        """Conservative preliminary estimate, for already associated consecutive pulses.

        Missing pulses and mixed emitters invalidate a naive adjacent-TOA estimate.
        Do not call this on an interleaved observation stream.
        """
        positive = [v for v in intervals_s if v > 0]
        if len(positive) < 3:
            self.kind = PRIType.UNKNOWN
            self.confidence = 0.0
            return
        centre = median(positive)
        deviations = [abs(v - centre) for v in positive]
        spread = median(deviations)
        self.mean_s = centre
        self.spread_s = spread
        if spread <= centre * relative_fixed_tolerance:
            self.kind = PRIType.FIXED
            self.confidence = min(0.8, len(positive) / 20)
        else:
            self.kind = PRIType.UNKNOWN
            self.confidence = 0.0


@dataclass
class FrequencyModel:
    mean_hz: float | None = None
    spread_hz: float | None = None


@dataclass
class EmitterTrack:
    track_id: str
    associated_pdw_ids: list[int] = field(default_factory=list)
    last_toa_s: float | None = None
    pri_model: PRIModel = field(default_factory=PRIModel)
    frequency_model: FrequencyModel = field(default_factory=FrequencyModel)
    bearing_deg: float | None = None
    association_confidence: float = 0.0

    def add_pdw(self, pdw):
        if self.last_toa_s is not None and pdw.toa_s < self.last_toa_s:
            raise ValueError("PDWs must be associated in TOA order")
        if pdw.pdw_id in self.associated_pdw_ids:
            raise ValueError("PDW already associated")
        self.associated_pdw_ids.append(pdw.pdw_id)
        self.last_toa_s = pdw.toa_s
        n = len(self.associated_pdw_ids)
        old = self.frequency_model.mean_hz
        self.frequency_model.mean_hz = pdw.frequency_hz if old is None else old + (pdw.frequency_hz - old) / n
        old_bearing = self.bearing_deg
        self.bearing_deg = pdw.aoa_deg if old_bearing is None else old_bearing + (pdw.aoa_deg - old_bearing) / n
