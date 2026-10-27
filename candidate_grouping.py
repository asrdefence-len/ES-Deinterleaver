"""Canonical conservative PDW candidate grouping, baseline v0.1.

Unsupervised: no truth labels or emitter identities used. Labels are LOCAL to
each call/window and MUST NOT be interpreted as persistent emitter IDs.
"""
import numpy as np
from sklearn.cluster import DBSCAN


RF_FRACTION = 0.08
AOA_SCALE_DEG = 2.0
PW_ALPHA = 0.05
DBSCAN_EPS = 1.0
DBSCAN_MIN_SAMPLES = 8


def robust_scale(x):
    q = np.percentile(x, [25, 75])
    return max(float(q[1] - q[0]) / 3.0, 1e-9)


def group_pdws(data, feature_names):
    """Return window-local DBSCAN group IDs (-1 means ungrouped).

    Required columns: Frequency (positive), PulseWidth, AoA (degrees).
    RF distance uses log frequency / 0.08, not an 8% hard bandwidth gate.
    PW scale is estimated per supplied window. All ungrouped PDWs remain
    available for subsequent processing.
    """
    names = list(feature_names)
    rf = np.asarray(data[:, names.index("Frequency")], dtype=float)
    pw = np.asarray(data[:, names.index("PulseWidth")], dtype=float)
    aoa = np.deg2rad(np.asarray(data[:, names.index("AoA")], dtype=float))
    if not np.all(np.isfinite(rf)) or not np.all(np.isfinite(pw)) or not np.all(np.isfinite(aoa)):
        raise ValueError("RF/PW/AOA must be finite")
    if np.any(rf <= 0):
        raise ValueError("Frequency must be positive for log scaling")
    pw_scale = robust_scale(pw)
    features = np.column_stack((
        np.sin(aoa) / np.deg2rad(AOA_SCALE_DEG),
        np.cos(aoa) / np.deg2rad(AOA_SCALE_DEG),
        np.log(rf) / RF_FRACTION,
        np.log1p(PW_ALPHA * pw / pw_scale) / PW_ALPHA,
    ))
    return DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(features)
