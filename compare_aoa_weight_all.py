"""AOA-weight sensitivity and cross-bearing contamination audit for HDBSCAN.

Ground truth is used ONLY after clustering. Independent windows; no persistent IDs.
AOA is represented by sin/cos columns 0:2 in the existing feature matrix.
"""
import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
from compare_hdbscan_all import feature_matrix, score


def circular_distance(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def circular_center(degrees):
    radians = np.deg2rad(degrees)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)),
                                      np.mean(np.cos(radians)))) % 360.0)


def audit(pred, truth, aoa, separation_deg):
    """Count mixed-emitter PDWs involving clearly separated truth-emitter bearings.

    Emitter bearing is estimated from its observed PDWs in the same window.
    For each group, count minority-emitter pulses whose emitter bearing differs
    from the dominant emitter by >= separation_deg. Also report group angular
    spread (95th percentile distance from circular group center).
    """
    centers = {eid: circular_center(aoa[truth == eid])
               for eid in np.unique(truth)}
    distant_cross = 0
    cross = 0
    spread95 = []
    max_spread95 = 0.0
    for gid in np.unique(pred[pred >= 0]):
        mask = pred == gid
        counts = Counter(truth[mask].tolist())
        dominant = counts.most_common(1)[0][0]
        cross += sum(counts.values()) - counts[dominant]
        for eid, n in counts.items():
            if eid != dominant and circular_distance(centers[eid], centers[dominant]) >= separation_deg:
                distant_cross += n
        local_aoa = aoa[mask]
        p95 = float(np.percentile(circular_distance(local_aoa, circular_center(local_aoa)), 95))
        spread95.append(p95)
        max_spread95 = max(max_spread95, p95)
    return dict(distant_cross=distant_cross, cross_check=cross,
                groups_spread95_over_gate=sum(s >= separation_deg for s in spread95),
                max_group_aoa_spread95_deg=max_spread95)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--weights", nargs="+", type=float, default=[1, 2, 3, 5, 10])
    p.add_argument("--separation-deg", type=float, default=10.0)
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--output", default="turing_aoa_weight_results.csv")
    a = p.parse_args()
    if any(w < 0 for w in a.weights) or a.separation_deg <= 0:
        p.error("Weights must be nonnegative and separation must be positive")
    rows = []
    for path in a.files:
        with h5py.File(path) as f:
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in f["metadata/feature_names"][:]]
            n = len(f["data"])
            print(f"\\n{path}: {n:,} PDWs", flush=True)
            for wi, start in enumerate(range(0, n, a.window_size)):
                stop = min(start + a.window_size, n)
                if stop - start < 8:
                    continue
                x = np.asarray(f["data"][start:stop], dtype=float)
                truth = np.asarray(f["labels"][start:stop]).reshape(-1)
                aoa = x[:, names.index("AoA")]
                z = feature_matrix(x, names)
                for weight in a.weights:
                    zw = z.copy()
                    zw[:, :2] *= weight  # weight BOTH circular AOA components
                    t = time.perf_counter()
                    pred = HDBSCAN(min_cluster_size=a.min_cluster_size,
                                   min_samples=a.min_samples).fit_predict(zw)
                    seconds = time.perf_counter() - t
                    metrics = score(pred, truth)
                    diagnostic = audit(pred, truth, aoa, a.separation_deg)
                    assert diagnostic["cross_check"] == metrics["cross"]
                    del diagnostic["cross_check"]
                    rows.append(dict(file=path, window=wi, weight=weight,
                                     seconds=seconds, **metrics, **diagnostic))
                if (wi + 1) % 10 == 0 or stop == n:
                    print(f"  processed {stop:,}/{n:,}", flush=True)
    if not rows:
        raise SystemExit("No valid windows")
    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for title, subsets in [("OVERALL", [("ALL", rows)])] + [
            (Path(path).name, [(Path(path).name, [r for r in rows if r["file"] == path])])
            for path in a.files]:
        print(f"\\n{title}: cross = all cross-emitter PDWs; distant = cross-emitter "
              f"PDWs whose truth-emitter mean bearings differ by >= {a.separation_deg:g} deg")
        print("AOAweight  purity%   cross  distant   noise   groups   wide_groups  seconds")
        for _, subset in subsets:
            for w in a.weights:
                rr = [r for r in subset if r["weight"] == w]
                if not rr:
                    continue
                grouped = sum(r["grouped"] for r in rr)
                correct = sum(r["correct"] for r in rr)
                print(f"{w:9g} {100*correct/grouped if grouped else 0:8.3f} "
                      f"{sum(r['cross'] for r in rr):7d} "
                      f"{sum(r['distant_cross'] for r in rr):8d} "
                      f"{sum(r['noise'] for r in rr):7d} "
                      f"{sum(r['groups'] for r in rr):8d} "
                      f"{sum(r['groups_spread95_over_gate'] for r in rr):11d} "
                      f"{sum(r['seconds'] for r in rr):8.2f}", flush=True)
    print(f"\\nSaved {a.output}. Baseline unchanged; truth only used for evaluation.")
    print("Note: wide_groups is diagnostic, not automatically an error; emitter "
          "bearing may vary within a window.")


if __name__ == "__main__":
    main()
