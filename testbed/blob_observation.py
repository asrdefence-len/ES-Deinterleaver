"""Fixed-membership blob summaries emitted by the grouping stage.

Only bounded summary features cross the tracker interface. No pulse list,
TOA sequence, PRI discovery, truth label, or pulse reassignment is present.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BlobObservation:
    observation_id: str
    window_id: str
    blob_id: int
    start_s: float
    end_s: float
    pulse_count: int
    aoa_deg: float
    aoa_spread_deg: float
    frequency_p05_hz: float
    frequency_median_hz: float
    frequency_p95_hz: float
    pw_p05_s: float
    pw_median_s: float
    pw_p95_s: float

    def __post_init__(self):
        numbers = (self.start_s, self.end_s, self.aoa_deg, self.aoa_spread_deg,
            self.frequency_p05_hz, self.frequency_median_hz, self.frequency_p95_hz,
            self.pw_p05_s, self.pw_median_s, self.pw_p95_s)
        if (not self.observation_id or not self.window_id or self.pulse_count < 1
                or not all(math.isfinite(v) for v in numbers)
                or self.start_s > self.end_s or self.aoa_spread_deg < 0
                or not 0 < self.frequency_p05_hz <= self.frequency_median_hz <= self.frequency_p95_hz
                or not 0 < self.pw_p05_s <= self.pw_median_s <= self.pw_p95_s):
            raise ValueError("Invalid blob observation")


def summarize_blobs(data, feature_names, labels, window_id, *,
                    toa_scale=1e-6, frequency_scale=1e6, pw_scale=1e-6):
    """Grouping-stage adapter: summarize existing memberships once.

    Does not change labels or extract pulse trains. Unassigned label -1 is
    not promoted to a blob. Unit scales convert raw measurements to SI.
    """
    import numpy as np

    x, lab = np.asarray(data, float), np.asarray(labels)
    if (x.ndim != 2 or lab.shape != (len(x),)
            or not np.issubdtype(lab.dtype, np.integer)
            or not all(np.isfinite(v) and v > 0 for v in (toa_scale, frequency_scale, pw_scale))):
        raise ValueError("Invalid grouping-stage input or unit scales")
    names = list(feature_names)
    columns = [names.index(n) for n in ("ToA", "Frequency", "PulseWidth", "AoA")]
    output = []
    for blob in np.unique(lab[lab >= 0]):
        z = x[lab == blob][:, columns]
        if not np.all(np.isfinite(z)) or np.any(z[:, 1:3] <= 0):
            raise ValueError("Require finite measurements and positive RF/PW")
        angles = np.deg2rad(z[:, 3])
        mean_angle = float(np.rad2deg(np.angle(np.mean(np.exp(1j * angles)))))
        deviations = (z[:, 3] - mean_angle + 180) % 360 - 180
        spread = float(np.sqrt(np.mean(deviations**2)))
        rf = np.quantile(z[:, 1] * frequency_scale, [.05, .5, .95])
        pw = np.quantile(z[:, 2] * pw_scale, [.05, .5, .95])
        output.append(BlobObservation(
            f"{window_id}:B{int(blob)}", str(window_id), int(blob),
            float(z[:, 0].min() * toa_scale), float(z[:, 0].max() * toa_scale),
            len(z), mean_angle, spread, *map(float, rf), *map(float, pw)))
    return tuple(sorted(output, key=lambda o: (o.start_s, o.observation_id)))
