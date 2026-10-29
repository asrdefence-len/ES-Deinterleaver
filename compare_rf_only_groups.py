"""Compare batch-AOA RF-only candidate groups with existing joint RF/PW groups.

Run: python -u compare_rf_only_groups.py turing_data/config_101.h5
Both methods use identical AOA populations and HDBSCAN (8/8).
PW is retained in input PDWs but excluded from RF-only clustering.
Truth labels are used only for evaluation. Groups are NOT emitter tracks.
"""
import argparse
import csv
import time
from collections import Counter
import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import completeness_score, v_measure_score
from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_hdbscan_all import score, feature_matrix


def rf_only_groups(x, names, populations, npop, mcs=8, ms=8):
    # Reuse the exact RF transform/scale from the joint-feature baseline.
    rf_feature = feature_matrix(x, names)[:, 2:3]
    pred = np.full(len(x), -1, dtype=int)
    gid = 0
    for pop in range(npop):
        ii = np.flatnonzero(populations == pop)
        if len(ii) < mcs:
            continue
        local = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(rf_feature[ii])
        for lab in np.unique(local[local >= 0]):
            pred[ii[local == lab]] = gid
            gid += 1
    return pred


def emitter_stats(pred, truth, emitter):
    ii = np.flatnonzero(truth == emitter)
    if not len(ii):
        return None
    counts = Counter(pred[ii][pred[ii] >= 0].tolist())
    return dict(pdws=len(ii), grouped=sum(counts.values()),
                groups_touched=len(counts),
                largest_group=max(counts.values(), default=0),
                largest_fraction=max(counts.values(), default=0) / len(ii))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--focus-window", type=int, default=0)
    p.add_argument("--focus-emitters", nargs="+", type=int, default=[9, 46, 20])
    p.add_argument("--output", default="turing_rf_only_comparison.csv")
    a = p.parse_args()
    rows = []
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        n = len(f["data"])
        print(f"{a.file}: {n:,} PDWs", flush=True)
        for wi, start in enumerate(range(0, n, a.window_size)):
            stop = min(start + a.window_size, n)
            if stop - start < max(8, a.min_cluster_size):
                continue
            x = np.asarray(f["data"][start:stop], float)
            truth = np.asarray(f["labels"][start:stop]).reshape(-1)
            pops, npop = batch_aoa_populations(x[:, names.index("AoA")])
            for method, fn in (("joint_RF_PW", rf_pw_groups),
                               ("RF_only", rf_only_groups)):
                t0 = time.perf_counter()
                pred = fn(x, names, pops, npop,
                          mcs=a.min_cluster_size, ms=a.min_samples)
                elapsed = time.perf_counter() - t0
                m = score(pred, truth)
                keep = pred >= 0
                rows.append(dict(window=wi, method=method, pdws=len(x),
                                 populations=npop, seconds=elapsed, **m,
                                 v_grouped=v_measure_score(truth[keep], pred[keep])
                                 if keep.sum() >= 2 else float("nan"),
                                 completeness_grouped=completeness_score(
                                     truth[keep], pred[keep])
                                 if keep.sum() >= 2 else float("nan")))
                if wi == a.focus_window:
                    print(f"\nWINDOW {wi} {method}: {npop} AOA populations, "
                          f"{m['groups']} groups, {m['cross']} cross-emitter PDWs")
                    for eid in a.focus_emitters:
                        s = emitter_stats(pred, truth, eid)
                        if s:
                            print(f"  emitter {eid}: {s['pdws']} PDWs, "
                                  f"{s['grouped']} grouped, "
                                  f"{s['groups_touched']} groups touched, "
                                  f"largest group {s['largest_group']} "
                                  f"({s['largest_fraction']:.1%} of pulses)")
            if (wi + 1) % 10 == 0 or stop == n:
                print(f"processed {stop:,}/{n:,}", flush=True)
    with open(a.output, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmethod purity% coverage% cross noise groups groups_per_10k "
          "seconds V_grouped completeness_grouped")
    for method in ("joint_RF_PW", "RF_only"):
        rr = [r for r in rows if r["method"] == method]
        total = sum(r["pdws"] for r in rr)
        grouped = sum(r["grouped"] for r in rr)
        correct = sum(r["correct"] for r in rr)
        print(f"{method:12s} {100*correct/grouped if grouped else 0:7.3f} "
              f"{100*grouped/total:8.3f} "
              f"{sum(r['cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} "
              f"{sum(r['groups'] for r in rr):7d} "
              f"{10000*sum(r['groups'] for r in rr)/total:8.1f} "
              f"{sum(r['seconds'] for r in rr):7.2f} "
              f"{np.nanmean([r['v_grouped'] for r in rr]):.3f} "
              f"{np.nanmean([r['completeness_grouped'] for r in rr]):.3f}")
    print("Note: purity can fall if unrelated emitters share RF and AOA. "
          "Grouped-only metrics exclude unassigned PDWs.")
    print("RF-only clustering does not automatically produce one group per "
          "nominal frequency; density structure may split or merge bands.")
    print("Saved", a.output)


if __name__ == "__main__":
    main()
