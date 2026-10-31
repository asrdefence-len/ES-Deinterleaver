"""Experimental blob association from temporal activity; no truth inputs.

Inspired by Mottier et al., TAES 2024, DOI 10.1109/TAES.2024.3367287.
This is NOT a reproduction of HACOT/IHACOT: fixed common histogram bins,
an explicit distance threshold and complete-link merging replace their
adaptive binning, hierarchical aggregation and statistical pruning.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class BlobPair:
    left: int
    right: int
    left_count: int
    right_count: int
    wasserstein_fraction: float
    aoa_distance_deg: float
    eligible: bool
    reason: str


def _circular_summary(degrees):
    z = np.mean(np.exp(1j * np.deg2rad(degrees)))
    return float(np.rad2deg(np.angle(z)) % 360), float(abs(z))


def score_blob_pairs(toa, labels, amplitude=None, *, amplitude_scale="uniform",
                     aoa_deg=None, min_pulses=100, bins=128,
                     min_aoa_resultant=0.5):
    """W1 between level-weighted TOA histograms on ONE common time axis.

    Output units are fractions of the entire observation-window duration.
    No per-blob time alignment or independent time normalization is allowed.
    dB levels are converted to positive linear POWER weights (10**(dB/10));
    linear inputs must already be nonnegative power/energy weights.
    Negative labels remain unassigned. Sparse and directionally diffuse
    blobs are ineligible, rather than forcibly associated.
    """
    t = np.asarray(toa, dtype=float)
    lab = np.asarray(labels)
    if (t.ndim != 1 or lab.shape != t.shape or not np.all(np.isfinite(t))
            or not np.issubdtype(lab.dtype, np.integer)):
        raise ValueError("Require finite 1-D TOA and aligned integer blob labels")
    if (not isinstance(min_pulses, (int, np.integer)) or min_pulses < 2
            or not isinstance(bins, (int, np.integer)) or bins < 2
            or not 0 <= min_aoa_resultant <= 1):
        raise ValueError("Invalid count, bins or AOA concentration threshold")
    if amplitude_scale not in ("uniform", "db", "linear"):
        raise ValueError("Amplitude scale must be uniform, db or linear")
    amp = None if amplitude is None else np.asarray(amplitude, dtype=float)
    if amplitude_scale != "uniform" and (
            amp is None or amp.shape != t.shape or not np.all(np.isfinite(amp))):
        raise ValueError("Require aligned finite amplitude values")
    if amplitude_scale == "linear" and np.any(amp < 0):
        raise ValueError("Linear power weights cannot be negative")
    aoa = None if aoa_deg is None else np.asarray(aoa_deg, dtype=float)
    if aoa is not None and (aoa.shape != t.shape or not np.all(np.isfinite(aoa))):
        raise ValueError("Require aligned finite AOA values")
    if len(t) == 0:
        return []
    duration = float(np.ptp(t))
    if duration <= 0:
        raise ValueError("Observation window must have positive duration")
    relative = (t - t.min()) / duration
    edges = np.linspace(0, 1, bins + 1)
    summaries = {}
    for blob in np.unique(lab[lab >= 0]):
        keep = lab == blob
        n = int(keep.sum())
        if amplitude_scale == "db":
            # Per-blob scaling cancels when the histogram is normalized.
            weights = np.power(10.0, np.maximum((amp[keep] - amp[keep].max()) / 10, -300))
        elif amplitude_scale == "linear":
            values = amp[keep]
            weights = values / values.max() if values.max() > 0 else values
        else:
            weights = np.ones(n)
        hist, _ = np.histogram(relative[keep], bins=edges, weights=weights)
        mass = float(hist.sum())
        cdf = np.cumsum(hist / mass) if mass > 0 else None
        direction, concentration = _circular_summary(aoa[keep]) if aoa is not None else (None, None)
        summaries[int(blob)] = (n, cdf, direction, concentration)
    result = []
    for left, right in combinations(summaries, 2):
        n, u, a, ar = summaries[left]
        m, v, b, br = summaries[right]
        # Exact W1 for probability masses on equally spaced bin centres.
        distance = float(np.sum(np.abs(u[:-1] - v[:-1])) / bins) if u is not None and v is not None else float("nan")
        angle = abs((a - b + 180) % 360 - 180) if a is not None else float("nan")
        reason = ("sparse" if min(n, m) < min_pulses else
                  "zero_weight" if u is None or v is None else
                  "diffuse_aoa" if ar is not None and min(ar, br) < min_aoa_resultant else "ok")
        result.append(BlobPair(left, right, n, m, distance, float(angle), reason == "ok", reason))
    return result


def merge_blob_candidates(labels, pairs, *, max_distance=0.03, aoa_gate_deg=None):
    """Exploratory complete-link grouping; every cross-pair must pass.

    Avoids a chain of individually plausible edges silently joining two
    incompatible blobs. These are candidate groups, NOT emitter tracks.
    Without an AOA gate, a low OT distance alone can falsely merge emitters.
    """
    lab = np.asarray(labels)
    if lab.ndim != 1 or not np.issubdtype(lab.dtype, np.integer):
        raise ValueError("Require 1-D integer labels")
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("Distance threshold must be finite and nonnegative")
    if aoa_gate_deg is not None and (not np.isfinite(aoa_gate_deg) or not 0 < aoa_gate_deg <= 180):
        raise ValueError("AOA gate must be in (0, 180]")
    lookup = {tuple(sorted((p.left, p.right))): p for p in pairs}
    groups = [{int(b)} for b in np.unique(lab[lab >= 0])]

    def compatible(a, b):
        distances = []
        for i in a:
            for j in b:
                p = lookup.get(tuple(sorted((i, j))))
                if p is None or not p.eligible or not np.isfinite(p.wasserstein_fraction) or p.wasserstein_fraction > max_distance:
                    return None
                if aoa_gate_deg is not None and (not np.isfinite(p.aoa_distance_deg) or p.aoa_distance_deg > aoa_gate_deg):
                    return None
                distances.append(p.wasserstein_fraction)
        return max(distances)

    while True:
        candidates = []
        for i, j in combinations(range(len(groups)), 2):
            d = compatible(groups[i], groups[j])
            if d is not None:
                candidates.append((d, min(groups[i]), min(groups[j]), i, j))
        if not candidates:
            break
        _, _, _, i, j = min(candidates)
        groups[i] |= groups[j]
        del groups[j]
    out = lab.copy()
    for group in groups:
        out[np.isin(lab, list(group))] = min(group)
    return out
