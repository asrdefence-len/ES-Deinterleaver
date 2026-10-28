"""Experimental online AOA population initiation + RF/PW-only HDBSCAN.

AOA population != physical emitter track. A tentative population starts from
one PDW; subsequent compatible PDWs update its circular mean. Fixed gates are
tested before introducing an SNR-dependent uncertainty model.

Uses spatial lookup bins for efficient candidate search. PDWs are processed in
TOA order (stable ties); ambiguous matches go to nearest population. This is a
hard-assignment proof of concept, not a multi-hypothesis association algorithm.
"""
import argparse
import csv
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
from compare_hdbscan_all import feature_matrix, score
from compare_aoa_weight_all import audit
from compare_aoa_gate_all import cluster_gated


def online_aoa_populations(angles, toa, gate):
    """Online circular mean and fixed compatibility gate.

    Each track stores a unit-vector sum and count. Spatial hash bins accelerate
    nearest compatible track lookup. Moving means are reindexed after updates.
    """
    angles = np.asarray(angles, dtype=float) % 360.0
    order = np.argsort(toa, kind="stable")
    nbins = max(1, int(np.ceil(360.0 / gate)))
    bin_width = 360.0 / nbins
    bins = {}
    sums_s, sums_c, counts, centers, memberships = [], [], [], [], []
    labels = np.empty(len(angles), dtype=int)

    def bin_id(deg):
        return int(deg / bin_width) % nbins

    def delta(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    radius = int(np.ceil(gate / bin_width)) + 1
    for idx in order:
        angle = float(angles[idx])
        b = bin_id(angle)
        candidate_ids = set()
        for off in range(-radius, radius + 1):
            candidate_ids.update(bins.get((b + off) % nbins, ()))
        best, best_dist = None, gate + 1e-10
        for tid in candidate_ids:
            d = delta(angle, centers[tid])
            if d <= gate and d < best_dist:
                best, best_dist = tid, d
        rad = np.deg2rad(angle)
        if best is None:
            tid = len(centers)
            sums_s.append(float(np.sin(rad)))
            sums_c.append(float(np.cos(rad)))
            counts.append(1)
            centers.append(angle)
            memberships.append(bin_id(angle))
            bins.setdefault(memberships[-1], set()).add(tid)
        else:
            tid = best
            sums_s[tid] += np.sin(rad)
            sums_c[tid] += np.cos(rad)
            counts[tid] += 1
            new_center = float(np.rad2deg(np.arctan2(sums_s[tid], sums_c[tid])) % 360.0)
            new_bin = bin_id(new_center)
            if new_bin != memberships[tid]:
                bins[memberships[tid]].remove(tid)
                bins.setdefault(new_bin, set()).add(tid)
                memberships[tid] = new_bin
            centers[tid] = new_center
        labels[idx] = tid
    return labels, len(centers)


def cluster_online(x, names, gate, min_cluster_size, min_samples):
    angles = x[:, names.index("AoA")]
    toa = x[:, names.index("ToA")]
    populations, npop = online_aoa_populations(angles, toa, gate)
    features = feature_matrix(x, names)[:, 2:]
    pred = np.full(len(x), -1, dtype=int)
    gid = 0
    for pop in range(npop):
        idx = np.flatnonzero(populations == pop)
        if len(idx) < min_cluster_size:
            continue
        local = HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples).fit_predict(features[idx])
        for label in np.unique(local[local >= 0]):
            pred[idx[local == label]] = gid
            gid += 1
    return pred, npop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--gates", nargs="+", type=float, default=[2, 3, 5])
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--separation-deg", type=float, default=10)
    p.add_argument("--output", default="turing_aoa_online_results.csv")
    a = p.parse_args()
    if any(g <= 0 or g >= 180 for g in a.gates):
        p.error("Gates must be between 0 and 180 degrees")
    rows = []
    for path in a.files:
        with h5py.File(path) as f:
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in f["metadata/feature_names"][:]]
            n = len(f["data"])
            print(f"\\n{path}: {n:,} PDWs", flush=True)
            for wi, start in enumerate(range(0, n, a.window_size)):
                stop = min(start + a.window_size, n)
                if stop - start < a.min_cluster_size:
                    continue
                x = np.asarray(f["data"][start:stop], dtype=float)
                truth = np.asarray(f["labels"][start:stop]).reshape(-1)
                angles = x[:, names.index("AoA")]
                for gate in a.gates:
                    for method in ("seed", "online"):
                        t = time.perf_counter()
                        if method == "seed":
                            pred, npop = cluster_gated(x, names, gate,
                                                       a.min_cluster_size, a.min_samples)
                        else:
                            pred, npop = cluster_online(x, names, gate,
                                                        a.min_cluster_size, a.min_samples)
                        elapsed = time.perf_counter() - t
                        metrics = score(pred, truth)
                        diag = audit(pred, truth, angles, a.separation_deg)
                        assert diag.pop("cross_check") == metrics["cross"]
                        rows.append(dict(file=path, window=wi, method=method,
                                         gate_deg=gate, populations=npop,
                                         seconds=elapsed, **metrics, **diag))
                if (wi + 1) % 10 == 0 or stop == n:
                    print(f"  processed {stop:,}/{n:,}", flush=True)
    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\\nOVERALL: fixed-seed versus online circular-mean AOA populations")
    print("method gate purity% cross distant noise groups populations seconds")
    for gate in a.gates:
        for method in ("seed", "online"):
            rr = [r for r in rows if r["gate_deg"] == gate and r["method"] == method]
            grouped = sum(r["grouped"] for r in rr)
            correct = sum(r["correct"] for r in rr)
            print(f"{method:6s} {gate:4g} "
                  f"{100*correct/grouped if grouped else 0:7.3f} "
                  f"{sum(r['cross'] for r in rr):6d} "
                  f"{sum(r['distant_cross'] for r in rr):7d} "
                  f"{sum(r['noise'] for r in rr):7d} "
                  f"{sum(r['groups'] for r in rr):6d} "
                  f"{sum(r['populations'] for r in rr):11d} "
                  f"{sum(r['seconds'] for r in rr):7.2f}", flush=True)
    print(f"Saved {a.output}. Truth used only for scoring.")
    print("NOTE: online means can drift; hard association may merge nearby emitters. "
          "Gate width is fixed and no SNR model is used.")


if __name__ == "__main__":
    main()
