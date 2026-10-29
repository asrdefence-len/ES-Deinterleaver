"""Experimental adaptive RF connectivity, within batch AOA populations.

Usage: python -u compare_adaptive_rf.py turing_data/config_101.h5
No truth labels used in grouping. All thresholds are experimental.
"""
import argparse
import csv
import time
from collections import Counter
import h5py
import numpy as np
from sklearn.metrics import v_measure_score, completeness_score
from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_hdbscan_all import score


def adaptive_rf_groups(values, k=5, gap_factor=8.0, min_mass=8, guard=1e-9):
    """Find supported runs of RF samples with locally normal spacings.

    Each sorted adjacent gap is compared to k-neighbour distances on both
    sides, rather than the global RF span. A split requires a gap to be
    anomalously large relative to BOTH local scales. Singleton/outlier runs
    remain unassigned. This is connectivity, not emitter identification.
    """
    v = np.asarray(values, float)
    n = len(v)
    labels = np.full(n, -1, dtype=int)
    if n < min_mass:
        return labels
    order = np.argsort(v, kind="stable")
    s = v[order]
    gaps = np.diff(s)
    if not len(gaps):
        labels[:] = 0
        return labels
    cuts = []
    for i, g in enumerate(gaps):
        left = gaps[max(0, i-k):i]
        right = gaps[i+1:min(len(gaps),i+k+1)]
        # Robust local scale: a gap must be exceptional on both sides.
        if len(left) < 2 or len(right) < 2:
            continue
        scale = max(float(np.median(left)), float(np.median(right)), guard)
        if g > gap_factor * scale:
            cuts.append(i+1)
    bounds = [0] + cuts + [n]
    gid = 0
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end-start >= min_mass:
            labels[order[start:end]] = gid
            gid += 1
    return labels


def group_window(x, names, pops, npop, k, factor, min_mass):
    rf = x[:, names.index("Frequency")]
    result = np.full(len(x), -1, dtype=int)
    gid = 0
    for p in range(npop):
        ii = np.flatnonzero(pops == p)
        local = adaptive_rf_groups(rf[ii], k=k, gap_factor=factor, min_mass=min_mass)
        for lab in np.unique(local[local >= 0]):
            result[ii[local == lab]] = gid
            gid += 1
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--windows", type=int, default=0, help="0 means all")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gap-factor", type=float, default=8)
    ap.add_argument("--min-mass", type=int, default=8)
    ap.add_argument("--focus-emitters", nargs="+", type=int, default=[9,46,20,58])
    ap.add_argument("--output", default="adaptive_rf_comparison.csv")
    a = ap.parse_args()
    if a.k < 2 or a.gap_factor <= 1 or a.min_mass < 2:
        ap.error("Require k >= 2, gap-factor > 1, min-mass >= 2")
    rows=[]
    with h5py.File(a.file) as h:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in h["metadata/feature_names"][:]]
        n=len(h["data"])
        print(f"{a.file}: {n:,} PDWs",flush=True)
        for wi,start in enumerate(range(0,n,a.window_size)):
            if a.windows and wi>=a.windows:
                break
            end=min(start+a.window_size,n)
            x=np.asarray(h["data"][start:end],float)
            truth=np.asarray(h["labels"][start:end]).reshape(-1)
            if len(x)<8:
                continue
            pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
            for method in ("joint_RF_PW","adaptive_RF"):
                t0=time.perf_counter()
                pred=(rf_pw_groups(x,names,pops,npop) if method=="joint_RF_PW"
                      else group_window(x,names,pops,npop,a.k,a.gap_factor,a.min_mass))
                seconds=time.perf_counter()-t0
                m=score(pred,truth)
                keep=pred>=0
                rows.append(dict(window=wi,method=method,pdws=len(x),
                                 populations=npop,seconds=seconds,**m,
                                 v_grouped=v_measure_score(truth[keep],pred[keep]) if keep.sum()>1 else np.nan,
                                 completeness_grouped=completeness_score(truth[keep],pred[keep]) if keep.sum()>1 else np.nan))
                if wi==0:
                    print(f"WINDOW 0 {method}: {m['groups']} groups, {m['cross']} cross-emitter PDWs, {m['noise']} unassigned",flush=True)
                    for eid in a.focus_emitters:
                        ids=pred[truth==eid]
                        cnt=Counter(ids[ids>=0].tolist())
                        print(f"  emitter {eid}: {len(ids)} PDWs, {len(cnt)} groups, {sum(cnt.values())} grouped, largest={max(cnt.values(),default=0)}",flush=True)
            if (wi+1)%10==0 or end==n:
                print(f"Processed {end:,}/{n:,}",flush=True)
    if not rows:
        print("No windows processed")
        return
    with open(a.output,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nmethod purity% coverage% cross noise groups groups_per_10k seconds")
    for method in ("joint_RF_PW","adaptive_RF"):
        rr=[r for r in rows if r["method"]==method]
        n=sum(r["pdws"] for r in rr)
        assigned=sum(r["grouped"] for r in rr)
        correct=sum(r["correct"] for r in rr)
        print(f"{method:12s} {100*correct/assigned if assigned else 0:7.3f} "
              f"{100*assigned/n:8.3f} {sum(r['cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} {sum(r['groups'] for r in rr):7d} "
              f"{10000*sum(r['groups'] for r in rr)/n:8.1f} "
              f"{sum(r['seconds'] for r in rr):7.2f}")
    print("Caution: this is a local-gap connectivity experiment, not a calibrated gravitational attraction model.")
    print("The baseline and adaptive method both use identical batch AOA populations.")
    print("Saved",a.output)


if __name__=="__main__":
    main()
