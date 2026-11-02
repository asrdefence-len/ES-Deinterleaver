"""Bounded blob or blob-train summaries emitted by the grouping stage.

Only bounded summary features cross the tracker interface. No pulse list,
TOA sequence, truth label, or pulse reassignment is present. Optional timing
fields are produced upstream by pulse-train analysis inside a frozen blob.
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
    train_id: str | None = None
    pri_s: float | None = None
    pri_sigma_s: float | None = None
    phase_epoch_s: float | None = None
    timing_rms_residual_s: float | None = None
    timing_pulse_count: int | None = None

    def __post_init__(self):
        numbers = (self.start_s, self.end_s, self.aoa_deg, self.aoa_spread_deg,
            self.frequency_p05_hz, self.frequency_median_hz, self.frequency_p95_hz,
            self.pw_p05_s, self.pw_median_s, self.pw_p95_s)
        timing = (self.pri_s, self.pri_sigma_s, self.phase_epoch_s,
                  self.timing_rms_residual_s, self.timing_pulse_count)
        if (not self.observation_id or not self.window_id or self.pulse_count < 1
                or not all(math.isfinite(v) for v in numbers)
                or self.start_s > self.end_s or self.aoa_spread_deg < 0
                or not 0 < self.frequency_p05_hz <= self.frequency_median_hz <= self.frequency_p95_hz
                or not 0 < self.pw_p05_s <= self.pw_median_s <= self.pw_p95_s):
            raise ValueError("Invalid blob observation")
        if any(value is not None for value in timing):
            if (self.train_id is None or any(value is None for value in timing)
                    or not all(math.isfinite(value) for value in timing[:-1])
                    or self.pri_s <= 0 or self.pri_sigma_s <= 0
                    or self.timing_rms_residual_s < 0
                    or not isinstance(self.timing_pulse_count, int)
                    or not 1 <= self.timing_pulse_count <= self.pulse_count):
                raise ValueError("Invalid blob timing observation")


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


def summarize_blob_trains(data, feature_names, labels, pdw_ids, window_id,
                          associated_trains, *, toa_scale=1e-6,
                          frequency_scale=1e6, pw_scale=1e-6,
                          pri_sigma_floor_s=.05e-6):
    """Export one bounded measurement per discovered train, plus fallbacks.

    Pulse membership is used only inside this grouping-stage adapter. The
    returned observations contain no PDW references or TOA sequences. Blobs
    with no discovered train retain their whole-blob summary.
    """
    import numpy as np

    x, lab, ids = np.asarray(data, float), np.asarray(labels), np.asarray(pdw_ids)
    if x.ndim != 2 or lab.shape != (len(x),) or ids.shape != (len(x),):
        raise ValueError("Invalid blob-train adapter input")
    if len(set(map(int, ids))) != len(ids):
        raise ValueError("Require unique PDW IDs")
    by_id = {int(pdw_id): index for index, pdw_id in enumerate(ids)}
    by_blob = {}
    for train in associated_trains:
        by_blob.setdefault(int(train.blob_id), []).append(train)
    output = []
    for blob in np.unique(lab[lab >= 0]):
        trains = by_blob.get(int(blob), [])
        if not trains:
            output.extend(summarize_blobs(x[lab == blob], feature_names,
                np.full(np.count_nonzero(lab == blob), int(blob), dtype=int),
                window_id, toa_scale=toa_scale,
                frequency_scale=frequency_scale, pw_scale=pw_scale))
            continue
        for train in trains:
            refs = train.observation.pdw_ids
            if any(int(ref) not in by_id for ref in refs):
                raise ValueError("Pulse-train member is outside the blob window")
            indices = np.asarray([by_id[int(ref)] for ref in refs], int)
            if np.any(lab[indices] != blob):
                raise ValueError("Pulse train crosses frozen blob membership")
            base = summarize_blobs(x[indices], feature_names,
                np.full(len(indices), int(blob), dtype=int), window_id,
                toa_scale=toa_scale, frequency_scale=frequency_scale,
                pw_scale=pw_scale)[0]
            span = max(1, train.span_slots)
            regression_sigma = (train.rms_residual_s * math.sqrt(12.) /
                                (math.sqrt(len(indices)) * span))
            pri_sigma = max(pri_sigma_floor_s, regression_sigma)
            output.append(BlobObservation(
                observation_id=train.observation.train_id,
                window_id=base.window_id, blob_id=base.blob_id,
                start_s=base.start_s, end_s=base.end_s,
                pulse_count=base.pulse_count, aoa_deg=base.aoa_deg,
                aoa_spread_deg=base.aoa_spread_deg,
                frequency_p05_hz=base.frequency_p05_hz,
                frequency_median_hz=base.frequency_median_hz,
                frequency_p95_hz=base.frequency_p95_hz,
                pw_p05_s=base.pw_p05_s, pw_median_s=base.pw_median_s,
                pw_p95_s=base.pw_p95_s,
                train_id=train.observation.train_id,
                pri_s=train.observation.pri_s, pri_sigma_s=pri_sigma,
                phase_epoch_s=train.phase_epoch_s,
                timing_rms_residual_s=train.rms_residual_s,
                timing_pulse_count=len(indices)))
    return tuple(sorted(output, key=lambda o: (o.start_s, o.observation_id)))
