"""Compare fixed and narrowing AOA gates with unchanged RF/PW HDBSCAN.

Run: python -u compare_aoa_narrowing.py turing_data/config_101.h5
Ground truth is used only for evaluation. Each 10k-PDW window is independent.
"""
import argparse
import csv
import time
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score
from compare_hdbscan_all import feature_matrix, score
from compare_aoa_online_all import online_aoa_populations
from compare_aoa_weight_all import audit


def narrowing_populations(angles, toa, wide=3.0, narrow=1.5, mature=10):
    """Online circular means, gate linear in observation count.

    Gate at count=1 is wide; at count>=mature is narrow. Hard nearest
    compatible association; no emitter truth used, no SNR or motion model.
    """
    angles = np.asarray(angles, float) % 360.0
    order = np.argsort(toa, kind="stable")
    # Candidate lookup uses the maximum gate throughout.
    nbins = max(1, int(np.ceil(360.0 / wide)))
    bw = 360.0 / nbins
    radius = int(np.ceil(wide / bw)) + 1
    bins = {}
    ss, cc, counts, centers, memberships = [], [], [], [], []
    labels = np.empty(len(angles), dtype=int)

    def bid(angle):
        return int(angle / bw) % nbins

    for idx in order:
        angle = float(angles[idx])
        b = bid(angle)
        candidates = set()
        for offset in range(-radius, radius + 1):
            candidates.update(bins.get((b + offset) % nbins, ()))
        best, distance = None, float("inf")
        for tid in candidates:
            gate = wide - (wide - narrow) * min(counts[tid] - 1, mature - 1) / (mature - 1)
            d = abs((angle - centers[tid] + 180.0) % 360.0 - 180.0)
            if d <= gate and d < distance:
                best, distance = tid, d
        rad = np.deg2rad(angle)
        if best is None:
            tid = len(centers)
            ss.append(float(np.sin(rad)))
            cc.append(float(np.cos(rad)))
            counts.append(1)
            centers.append(angle)
            memberships.append(b)
            bins.setdefault(b, set()).add(tid)
        else:
            tid = best
            ss[tid] += np.sin(rad)
            cc[tid] += np.cos(rad)
            counts[tid] += 1
            new_center = float(np.rad2deg(np.arctan2(ss[tid], cc[tid])) % 360.0)
            new_bin = bid(new_center)
            if new_bin != memberships[tid]:
                bins[memberships[tid]].remove(tid)
                bins.setdefault(new_bin, set()).add(tid)
                memberships[tid] = new_bin
            centers[tid] = new_center
        labels[idx] = tid
    return labels, len(centers)


def cluster_features(x, names, populations, npop, mcs=8, ms=8):
    feat = feature_matrix(x, names)[:, 2:]
    pred = np.full(len(x), -1, dtype=int)
    gid = 0
    for pop in range(npop):
        idx = np.flatnonzero(populations == pop)
        if len(idx) < mcs:
            continue
        local = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(feat[idx])
        for label in np.unique(local[local >= 0]):
            pred[idx[local == label]] = gid
            gid += 1
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--wide", type=float, default=3.0)
    p.add_argument("--narrow", type=float, default=1.5)
    p.add_argument("--mature", type=int, default=10)
    p.add_argument("--output", default="turing_aoa_narrowing_results.csv")
    a = p.parse_args()
    if not (0 < a.narrow <= a.wide < 180 and a.mature >= 2):
        p.error("Require 0 < narrow <= wide < 180, mature >= 2")
    methods = ("fixed_3", "fixed_1p5", "narrowing")
    rows = []
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        n = len(f["data"])
        for wi, start in enumerate(range(0, n, a.window_size)):
            end = min(n, start + a.window_size)
            if end - start < 8:
                continue
            x = np.asarray(f["data"][start:end], float)
            truth = np.asarray(f["labels"][start:end]).reshape(-1)
            aoa = x[:, names.index("AoA")]
            toa = x[:, names.index("ToA")]
            for method in methods:
                t0 = time.perf_counter()
                if method == "fixed_3":
                    pops, npop = online_aoa_populations(aoa, toa, a.wide)
                elif method == "fixed_1p5":
                    pops, npop = online_aoa_populations(aoa, toa, a.narrow)
                else:
                    pops, npop = narrowing_populations(aoa, toa, a.wide, a.narrow, a.mature)
                pred = cluster_features(x, names, pops, npop)
                seconds = time.perf_counter() - t0
                metrics = score(pred, truth)
                diag = audit(pred, truth, aoa, 10.0)
                # Noise is a single sentinel cluster here; report metrics on
                # grouped PDWs separately to avoid falsely merging all noise.
                keep = pred >= 0
                if keep.sum() >= 2:
                    v = v_measure_score(truth[keep], pred[keep])
                    ari = adjusted_rand_score(truth[keep], pred[keep])
                    ami = adjusted_mutual_info_score(truth[keep], pred[keep])
                    hom = homogeneity_score(truth[keep], pred[keep])
                    comp = completeness_score(truth[keep], pred[keep])
                else:
                    v = ari = ami = hom = comp = float("nan")
                rows.append(dict(file=a.file, window=wi, pdws=len(x), method=method,
                                 populations=npop, seconds=seconds, **metrics,
                                 distant_cross=diag["distant_cross"], v_grouped=v,
                                 ari_grouped=ari, ami_grouped=ami,
                                 homogeneity_grouped=hom, completeness_grouped=comp))
            if (wi + 1) % 10 == 0 or end == n:
                print(f"processed {end:,}/{n:,} PDWs", flush=True)
    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nmethod purity% coverage% cross distant noise groups populations groups_per_10k seconds V_grouped completeness_grouped")
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        grouped = sum(r["grouped"] for r in rr)
        correct = sum(r["correct"] for r in rr)
        total = sum(r["pdws"] for r in rr)
        print(f"{method:12s} {100*correct/grouped:7.3f} "
              f"{100*grouped/total:8.3f} {sum(r['cross'] for r in rr):7d} "
              f"{sum(r['distant_cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} "
              f"{sum(r['groups'] for r in rr):7d} "
              f"{sum(r['populations'] for r in rr):7d} "
              f"{10000*sum(r['groups'] for r in rr)/total:8.1f} "
              f"{sum(r['seconds'] for r in rr):7.2f} "
              f"{np.mean([r['v_grouped'] for r in rr]):.3f} "
              f"{np.mean([r['completeness_grouped'] for r in rr]):.3f}", flush=True)
    print("V/ARI/AMI/homogeneity/completeness above are grouped-only diagnostics, NOT official challenge metrics.")
    print("All window-level results saved:", a.output)


if __name__ == "__main__":
    main()
