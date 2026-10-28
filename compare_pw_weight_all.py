"""Compare HDBSCAN pulse-width weights across complete Turing datasets.

Blind grouping: truth labels are used only for post-fit scoring. Each 10k-PDW
window is independent; counts of groups across windows are NOT persistent tracks.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
from compare_hdbscan_all import feature_matrix, score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--weights", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--output", default="turing_pw_weight_full_results.csv")
    a = p.parse_args()
    if any(w <= 0 for w in a.weights):
        p.error("All weights must be positive")
    rows = []
    for path in a.files:
        with h5py.File(path) as f:
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in f["metadata/feature_names"][:]]
            n = len(f["data"])
            print(f"\n{path}: {n:,} PDWs", flush=True)
            for wi, start in enumerate(range(0, n, a.window_size)):
                stop = min(n, start + a.window_size)
                if stop - start < 8:
                    continue
                x = np.asarray(f["data"][start:stop], dtype=float)
                truth = np.asarray(f["labels"][start:stop]).reshape(-1)
                z = feature_matrix(x, names)
                for weight in a.weights:
                    zw = z.copy()
                    zw[:, 3] *= weight
                    t = time.perf_counter()
                    pred = HDBSCAN(min_cluster_size=a.min_cluster_size,
                                   min_samples=a.min_samples).fit_predict(zw)
                    elapsed = time.perf_counter() - t
                    metrics = score(pred, truth)
                    # Fragmentation diagnostic: number of groups dominated by
                    # each truth emitter in THIS window (not a mode count).
                    dominant = Counter()
                    for gid in np.unique(pred[pred >= 0]):
                        members = Counter(truth[pred == gid].tolist())
                        dominant[members.most_common(1)[0][0]] += 1
                    emitters = set(truth.tolist())
                    fragmentation = sum(dominant.values()) / len(emitters) if emitters else 0.0
                    rows.append(dict(file=path, window=wi, start=start, stop=stop,
                                     weight=weight, seconds=elapsed, **metrics,
                                     truth_emitters=len(emitters),
                                     groups_per_truth_emitter=fragmentation))
                if (wi + 1) % 10 == 0 or stop == n:
                    print(f"  processed {stop:,}/{n:,} PDWs", flush=True)
    if not rows:
        raise SystemExit("No windows with >=8 PDWs")
    with open(a.output, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nFULL-DATA HDBSCAN WEIGHT COMPARISON (purity weighted by grouped PDWs)")
    for path in a.files:
        for weight in a.weights:
            subset = [r for r in rows if r["file"] == path and r["weight"] == weight]
            grouped = sum(r["grouped"] for r in subset)
            correct = sum(r["correct"] for r in subset)
            cross = sum(r["cross"] for r in subset)
            noise = sum(r["noise"] for r in subset)
            groups = sum(r["groups"] for r in subset)
            seconds = sum(r["seconds"] for r in subset)
            emitter_windows = sum(r["truth_emitters"] for r in subset)
            purity = 100 * correct / grouped if grouped else 0.0
            fragmentation = groups / emitter_windows if emitter_windows else 0.0
            print(f"{Path(path).name:18s} PW={weight:3.1f} purity={purity:7.3f}% "
                  f"cross={cross:7d} noise={noise:7d} groups={groups:7d} "
                  f"groups/emitter-window={fragmentation:6.2f} seconds={seconds:7.2f}", flush=True)
    print("\nOVERALL (all files)")
    for weight in a.weights:
        subset = [r for r in rows if r["weight"] == weight]
        grouped = sum(r["grouped"] for r in subset)
        correct = sum(r["correct"] for r in subset)
        cross = sum(r["cross"] for r in subset)
        noise = sum(r["noise"] for r in subset)
        groups = sum(r["groups"] for r in subset)
        seconds = sum(r["seconds"] for r in subset)
        emitter_windows = sum(r["truth_emitters"] for r in subset)
        print(f"PW={weight:3.1f} purity={100*correct/grouped if grouped else 0:7.3f}% "
              f"cross={cross:8d} noise={noise:8d} groups={groups:8d} "
              f"groups/emitter-window={groups/emitter_windows if emitter_windows else 0:6.2f} "
              f"seconds={seconds:8.2f}", flush=True)
    print(f"Saved {a.output}. Ground truth used for scoring only; baseline unchanged.", flush=True)


if __name__ == "__main__":
    main()
