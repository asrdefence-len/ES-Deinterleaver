"""Turing PDW exploratory baseline: label-blind RF/PW/AoA clustering.

Usage: python turing_baseline.py turing_data/config_100.h5 --limit 10000
This is an intentionally simple, non-temporal baseline, NOT a full deinterleaver.
Only the first N PDWs are used for clustering; labels are read afterwards for
scoring. HDBSCAN, timing association, and emitter discovery are future work.
"""
import argparse
import json

import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure


def feature_indices(names):
    aliases = {"Frequency": ("Frequency",), "PulseWidth": ("PulseWidth",),
               "AoA": ("AoA",)}
    result = {}
    for key, choices in aliases.items():
        hits = [i for i, name in enumerate(names) if name in choices]
        if len(hits) != 1:
            raise ValueError(f"Expected exactly one {key} feature, got {hits}")
        result[key] = hits[0]
    return result


def load_sample(path, limit):
    if limit < 2:
        raise ValueError("--limit must be >= 2")
    with h5py.File(path, "r") as f:
        if "data" not in f or "metadata/feature_names" not in f:
            raise ValueError("Expected data and metadata/feature_names")
        raw = f["metadata/feature_names"][:]
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
        n = min(limit, f["data"].shape[0])
        data = np.asarray(f["data"][:n], dtype=float)
        labels = (np.asarray(f["labels"][:n]).reshape(-1)
                  if "labels" in f else None)
        total = int(f["data"].shape[0])
    if data.ndim != 2 or len(names) != data.shape[1]:
        raise ValueError("Invalid PDW feature matrix or metadata")
    return data, labels, names, total


def build_features(data, names):
    idx = feature_indices(names)
    rf = data[:, idx["Frequency"]]
    pw = data[:, idx["PulseWidth"]]
    aoa = np.deg2rad(data[:, idx["AoA"]])
    # Periodic angular embedding avoids the -180/+180 discontinuity.
    # Robust scales reduce the effect of extreme pulse widths/frequencies.
    base = np.column_stack((rf, np.log1p(np.maximum(pw, 0))))
    center = np.median(base, axis=0)
    scale = np.percentile(base, 75, axis=0) - np.percentile(base, 25, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = np.clip((base - center) / scale, -20, 20)
    return np.column_stack((z, np.sin(aoa), np.cos(aoa)))


def run(path, limit=10000, eps=0.25, min_samples=8):
    data, truth, names, total = load_sample(path, limit)
    if not np.all(np.isfinite(data)):
        raise ValueError("Non-finite PDW measurements")
    x = build_features(data, names)
    # DBSCAN is a simple reference; results depend strongly on eps and scales.
    predicted = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    groups = sorted(set(predicted) - {-1})
    result = {"file": str(path), "total_pdws": total,
              "sample_pdws": len(data), "feature_names": names,
              "algorithm": "DBSCAN RF/logPW/sinAoA/cosAoA",
              "eps": eps, "min_samples": min_samples,
              "predicted_clusters": len(groups),
              "unassigned_pdws": int(np.count_nonzero(predicted == -1)),
              "unassigned_fraction": float(np.mean(predicted == -1))}
    if truth is not None:
        # Scores include DBSCAN noise as a single -1 cluster. Report the
        # unassigned fraction separately to avoid hiding rejected PDWs.
        h, c, v = homogeneity_completeness_v_measure(truth, predicted)
        values, counts = np.unique(truth, return_counts=True)
        result["truth"] = {"unique_labels_in_sample": len(values),
                           "largest_label_count": int(counts.max()),
                           "smallest_label_count": int(counts.min())}
        result["metrics"] = {"adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
                             "homogeneity": float(h), "completeness": float(c),
                             "v_measure": float(v)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--min-samples", type=int, default=8)
    a = p.parse_args()
    if a.eps <= 0 or a.min_samples < 1:
        p.error("--eps must be positive and --min-samples >= 1")
    print(json.dumps(run(a.file, a.limit, a.eps, a.min_samples), indent=2))


if __name__ == "__main__":
    main()
