"""Diagnose mixed HDBSCAN groups in a Turing PDW window.

Truth labels are used ONLY for post-clustering diagnostics and plot colouring.
HDBSCAN features and parameters match compare_hdbscan_all.py.
"""
import argparse
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from compare_hdbscan_all import feature_matrix, score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    p.add_argument("--window", type=int, default=29)
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--top", type=int, default=5, help="Number of worst mixed groups to plot")
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    start = a.window * a.window_size
    with h5py.File(a.path) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        x = np.asarray(f["data"][start:start+a.window_size], dtype=float)
        truth = np.asarray(f["labels"][start:start+len(x)]).reshape(-1)
    if len(x) < 8:
        raise SystemExit("Window missing or too small")
    labels = HDBSCAN(min_cluster_size=a.min_cluster_size,
                     min_samples=a.min_samples).fit_predict(feature_matrix(x, names))
    print(f"{Path(a.path).name} window {a.window}, rows {start}:{start+len(x)}")
    print("HDBSCAN:", score(labels, truth))
    groups = []
    for gid in np.unique(labels[labels >= 0]):
        idx = labels == gid
        counts = Counter(truth[idx].tolist())
        cross = int(idx.sum()) - counts.most_common(1)[0][1]
        groups.append((cross, int(idx.sum()), int(gid), counts))
    groups.sort(reverse=True)
    print("\nWORST MIXED GROUPS (cross, size, group, emitter counts)")
    for cross, size, gid, counts in groups[:15]:
        print(f"  cross={cross:4d} size={size:4d} G{gid}: {counts.most_common(8)}")
    print("\nEMITTER FRAGMENTATION (dominant-group assignments only)")
    emitter_rows = []
    for emitter, count in Counter(truth.tolist()).most_common():
        dominant = [(gid, size) for cross, size, gid, counts in groups
                    if counts.most_common(1)[0][0] == emitter]
        assigned = sum(int(np.sum((labels == gid) & (truth == emitter)))
                       for gid, _ in dominant)
        emitter_rows.append((len(dominant), count, emitter, assigned, dominant))
    for ng, count, emitter, assigned, dominant in sorted(emitter_rows, reverse=True)[:15]:
        print(f"  emitter={emitter} PDWs={count} dominant_groups={ng} "
              f"covered={assigned}/{count} largest={sorted(dominant,key=lambda z:-z[1])[:6]}")
    rf = x[:, names.index("Frequency")]
    pw = x[:, names.index("PulseWidth")]
    aoa = x[:, names.index("AoA")]
    toa = x[:, names.index("ToA")]
    worst = [r for r in groups if r[0] > 0][:a.top]
    if not worst:
        print("No mixed groups to plot.")
        return
    fig, axes = plt.subplots(len(worst), 3, figsize=(16, 3.7*len(worst)), squeeze=False, layout="constrained")
    for row, (cross, size, gid, counts) in enumerate(worst):
        mask = labels == gid
        unique = [eid for eid, _ in counts.most_common()]
        colors = plt.get_cmap("tab10")
        for col, (xx, yy, xlabel, ylabel) in enumerate((
                (rf, pw, "RF (dataset units)", "PW (dataset units)"),
                (aoa, pw, "AOA (degrees)", "PW (dataset units)"),
                (toa, rf, "TOA (dataset units)", "RF (dataset units)"))):
            ax = axes[row, col]
            for j, eid in enumerate(unique):
                sel = mask & (truth == eid)
                ax.scatter(xx[sel], yy[sel], s=8, alpha=.7,
                           color=colors(j % 10), label=f"E{eid} ({sel.sum()})",
                           rasterized=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            ax.set_title(f"G{gid}: {size} PDWs, {cross} cross")
            if col == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{Path(a.path).name} window {a.window} — mixed HDBSCAN groups; colours are TRUTH for diagnosis only")
    out = a.output or f"{Path(a.path).stem}_window{a.window}_mixed_groups.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
