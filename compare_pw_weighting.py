"""Controlled HDBSCAN pulse-width weighting experiment; truth only for scoring."""
import argparse
from collections import Counter
import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
from compare_hdbscan_all import feature_matrix, score

def run(path, window, size, weights):
    start = window * size
    with h5py.File(path) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["metadata/feature_names"][:]]
        x = np.asarray(f["data"][start:start+size], dtype=float)
        truth = np.asarray(f["labels"][start:start+len(x)]).reshape(-1)
    if len(x) < 8:
        raise ValueError("Window has fewer than 8 PDWs")
    base = feature_matrix(x, names)
    print(f"{path} window={window} PDWs={len(x)}")
    print("PW weight | purity % | cross | noise | groups | E19/E20 mixed | E43 groups | E8 groups")
    for weight in weights:
        z = base.copy()
        z[:, 3] *= weight
        labels = HDBSCAN(min_cluster_size=8, min_samples=8).fit_predict(z)
        m = score(labels, truth)
        mix = 0
        for gid in np.unique(labels[labels >= 0]):
            c = Counter(truth[labels == gid].tolist())
            mix += min(c.get(19, 0), c.get(20, 0))
        fragmentation = {}
        for eid in (43, 8):
            fragmentation[eid] = sum(
                1 for gid in np.unique(labels[labels >= 0])
                if Counter(truth[labels == gid].tolist()).most_common(1)[0][0] == eid
            )
        print(f"{weight:9.2f} | {m['purity_pct']:8.3f} | {m['cross']:5d} | {m['noise']:5d} | "
              f"{m['groups']:6d} | {mix:13d} | {fragmentation[43]:10d} | {fragmentation[8]:9d}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="turing_data/config_115.h5")
    p.add_argument("--window", type=int, default=29)
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--weights", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    a = p.parse_args()
    run(a.path, a.window, a.window_size, a.weights)

if __name__ == "__main__":
    main()
