"""Compare HDBSCAN 8/8, 4/4 and 2/2 after online AOA gating.

Reports separately PDWs in AOA populations smaller than min_cluster_size,
and PDWs rejected by HDBSCAN in sufficiently large populations.
"""
import argparse
import time
import numpy as np
import h5py
from sklearn.cluster import HDBSCAN
from compare_hdbscan_all import feature_matrix, score
from compare_aoa_weight_all import audit
from compare_aoa_online_all import online_aoa_populations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--gate", type=float, default=3.0)
    p.add_argument("--window-size", type=int, default=10000)
    a = p.parse_args()
    settings = [(8, 8), (4, 4), (2, 2)]
    totals = {s: dict(grouped=0, correct=0, cross=0, distant=0,
                      noise=0, groups=0, small=0, hdb_noise=0,
                      populations=0, elapsed=0.0) for s in settings}
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        n = len(f["data"])
        print(f"{a.file}: {n:,} PDWs; online AOA gate +/-{a.gate:g} deg")
        for wi, start in enumerate(range(0, n, a.window_size)):
            end = min(n, start + a.window_size)
            x = np.asarray(f["data"][start:end], dtype=float)
            truth = np.asarray(f["labels"][start:end]).reshape(-1)
            aoa = x[:, names.index("AoA")]
            toa = x[:, names.index("ToA")]
            pops, npop = online_aoa_populations(aoa, toa, a.gate)
            feat = feature_matrix(x, names)[:, 2:]
            indices = [np.flatnonzero(pops == pop) for pop in range(npop)]
            for s in settings:
                minimum, samples = s
                t0 = time.perf_counter()
                pred = np.full(len(x), -1, dtype=int)
                small = 0
                hdb_noise = 0
                gid = 0
                for idx in indices:
                    if len(idx) < minimum:
                        small += len(idx)
                        continue
                    labels = HDBSCAN(min_cluster_size=minimum,
                                     min_samples=samples).fit_predict(feat[idx])
                    hdb_noise += int(np.sum(labels < 0))
                    for label in np.unique(labels[labels >= 0]):
                        pred[idx[labels == label]] = gid
                        gid += 1
                elapsed = time.perf_counter() - t0
                metrics = score(pred, truth)
                diag = audit(pred, truth, aoa, 10.0)
                assert metrics["noise"] == small + hdb_noise
                t = totals[s]
                t["grouped"] += metrics["grouped"]
                t["correct"] += metrics["correct"]
                t["cross"] += metrics["cross"]
                t["distant"] += diag["distant_cross"]
                t["noise"] += metrics["noise"]
                t["groups"] += metrics["groups"]
                t["small"] += small
                t["hdb_noise"] += hdb_noise
                t["populations"] += npop
                t["elapsed"] += elapsed
            if (wi + 1) % 10 == 0 or end == n:
                print(f"  processed {end:,}/{n:,}", flush=True)
    print("\\nRESULTS (truth for scoring only)")
    print("mcs/ms purity% coverage% cross distant noise small_AOA HDB_noise groups seconds")
    for s in settings:
        t = totals[s]
        print(f"{s[0]}/{s[1]} {100*t['correct']/t['grouped'] if t['grouped'] else 0:8.3f} "
              f"{100*t['grouped']/n:9.3f} {t['cross']:6d} {t['distant']:7d} "
              f"{t['noise']:6d} {t['small']:9d} {t['hdb_noise']:9d} "
              f"{t['groups']:6d} {t['elapsed']:7.2f}", flush=True)
    print("small_AOA: PDWs in AOA populations below min_cluster_size.")
    print("HDB_noise: PDWs rejected by HDBSCAN within eligible populations.")


if __name__ == "__main__":
    main()
