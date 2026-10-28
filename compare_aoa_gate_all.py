"""Compare fixed AOA population gates + RF/PW-only HDBSCAN against baseline.

Greedy circular AOA populations use a fixed seed centre and complete-width gate:
every member is within +/- gate degrees of its population seed. This is a
simple experimental spatial partition, NOT a probabilistic emitter tracker.
Truth labels are used only after clustering for evaluation.
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


def angle_difference(a, b):
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def spatial_populations(aoa, gate):
    """Assign each PDW once to a compatible seed-bearing population.

    Sort bearings first to reduce input-order sensitivity. Fixed seed centres
    prevent chaining, but can split populations near gate boundaries.
    """
    aoa = np.asarray(aoa, dtype=float) % 360.0
    order = np.argsort(aoa, kind="stable")
    labels = np.full(len(aoa), -1, dtype=int)
    centers = []
    for idx in order:
        compatible = [j for j, c in enumerate(centers)
                      if angle_difference(aoa[idx], c) <= gate]
        if compatible:
            chosen = min(compatible, key=lambda j: angle_difference(aoa[idx], centers[j]))
        else:
            chosen = len(centers)
            centers.append(float(aoa[idx]))
        labels[idx] = chosen
    return labels, len(centers)


def cluster_gated(x, names, gate, min_cluster_size, min_samples):
    aoa = x[:, names.index("AoA")]
    populations, n_populations = spatial_populations(aoa, gate)
    # Same RF/PW feature scaling as the baseline; AOA dimensions removed.
    features = feature_matrix(x, names)[:, 2:]
    predictions = np.full(len(x), -1, dtype=int)
    next_gid = 0
    for pop in range(n_populations):
        indices = np.flatnonzero(populations == pop)
        if len(indices) < min_cluster_size:
            continue
        local = HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples).fit_predict(features[indices])
        for local_gid in np.unique(local[local >= 0]):
            predictions[indices[local == local_gid]] = next_gid
            next_gid += 1
    return predictions, n_populations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--gates", nargs="+", type=float, default=[1, 2, 3, 5, 10])
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--separation-deg", type=float, default=10.0)
    p.add_argument("--output", default="turing_aoa_gate_results.csv")
    a = p.parse_args()
    if any(g <= 0 or g >= 180 for g in a.gates):
        p.error("Gate half-widths must be in (0,180) degrees")
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
                aoa = x[:, names.index("AoA")]
                variants = [("baseline", None)] + [(f"gate_{g:g}", g) for g in a.gates]
                for method, gate in variants:
                    t = time.perf_counter()
                    if gate is None:
                        pred = HDBSCAN(min_cluster_size=a.min_cluster_size,
                                       min_samples=a.min_samples).fit_predict(feature_matrix(x, names))
                        n_pop = 0
                    else:
                        pred, n_pop = cluster_gated(x, names, gate,
                                                    a.min_cluster_size, a.min_samples)
                    elapsed = time.perf_counter() - t
                    metrics = score(pred, truth)
                    diag = audit(pred, truth, aoa, a.separation_deg)
                    assert diag["cross_check"] == metrics["cross"]
                    del diag["cross_check"]
                    rows.append(dict(file=path, window=wi, method=method,
                                     gate_deg=gate if gate is not None else "",
                                     aoa_populations=n_pop, seconds=elapsed,
                                     **metrics, **diag))
                if (wi + 1) % 10 == 0 or stop == n:
                    print(f"  processed {stop:,}/{n:,}", flush=True)
    if not rows:
        raise SystemExit("No valid windows")
    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\\nOVERALL (truth used for scoring only)")
    print("method       purity%   cross  distant    noise   groups   aoa_pops  seconds")
    for method in ["baseline"] + [f"gate_{g:g}" for g in a.gates]:
        rr = [r for r in rows if r["method"] == method]
        grouped = sum(r["grouped"] for r in rr)
        correct = sum(r["correct"] for r in rr)
        print(f"{method:12s} {100*correct/grouped if grouped else 0:8.3f} "
              f"{sum(r['cross'] for r in rr):7d} "
              f"{sum(r['distant_cross'] for r in rr):8d} "
              f"{sum(r['noise'] for r in rr):8d} "
              f"{sum(r['groups'] for r in rr):8d} "
              f"{sum(r['aoa_populations'] for r in rr):9d} "
              f"{sum(r['seconds'] for r in rr):8.2f}", flush=True)
    print("\\nPER FILE")
    for path in a.files:
        print(Path(path).name)
        for method in ["baseline"] + [f"gate_{g:g}" for g in a.gates]:
            rr = [r for r in rows if r["file"] == path and r["method"] == method]
            grouped = sum(r["grouped"] for r in rr)
            correct = sum(r["correct"] for r in rr)
            print(f"  {method:12s} purity={100*correct/grouped if grouped else 0:7.3f}% "
                  f"cross={sum(r['cross'] for r in rr):7d} "
                  f"distant={sum(r['distant_cross'] for r in rr):6d} "
                  f"noise={sum(r['noise'] for r in rr):7d} "
                  f"groups={sum(r['groups'] for r in rr):7d}", flush=True)
    print(f"Saved {a.output}. Baseline unchanged.")
    print("Caveat: seed-based angular partitions may fragment near boundaries; "
          "this experiment tests the coarse-gate concept, not final tracking.")


if __name__ == "__main__":
    main()
