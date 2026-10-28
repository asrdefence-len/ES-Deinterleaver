"""Diagnose RF/PW/AOA fragmentation and pulse timing for selected truth emitters.

Run: python -u diagnose_fragmented_emitters.py turing_data/config_101.h5
Truth is used ONLY to select and colour evaluation examples; grouping is blind.
"""
import argparse
import csv
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_aoa_batch import batch_aoa_populations, rf_pw_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--emitters", nargs="+", type=int, default=[9, 46, 20])
    p.add_argument("--prefix", default="fragmented_emitter")
    a = p.parse_args()
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        start = a.window * a.window_size
        stop = min(start + a.window_size, len(f["data"]))
        if start >= stop:
            p.error("window out of range")
        x = np.asarray(f["data"][start:stop], float)
        truth = np.asarray(f["labels"][start:stop]).reshape(-1)
    aoa = x[:, names.index("AoA")]
    toa = x[:, names.index("ToA")]
    rf = x[:, names.index("Frequency")]
    pw = x[:, names.index("PulseWidth")]
    pops, npop = batch_aoa_populations(aoa)
    groups = rf_pw_groups(x, names, pops, npop)
    print(f"window {a.window}: {len(x)} PDWs, {npop} AOA populations, "
          f"{len(np.unique(groups[groups >= 0]))} candidate groups")
    rows = []
    for eid in a.emitters:
        ii = np.flatnonzero(truth == eid)
        if not len(ii):
            print(f"Emitter {eid} absent in window {a.window}")
            continue
        ii = ii[np.argsort(toa[ii], kind="stable")]
        g = groups[ii]
        t = toa[ii]
        # Preserve raw ToA units: no assumptions about dataset time scaling.
        delta = np.diff(t)
        valid = delta > 0
        positive = delta[valid]
        counts = Counter(g[g >= 0].tolist())
        print(f"\nEmitter {eid}: {len(ii)} PDWs, {len(counts)} candidate groups, "
              f"{int(np.sum(g < 0))} unassigned, "
              f"{len(np.unique(pops[ii]))} AOA populations")
        if len(positive):
            print(f"  adjacent truth-emitter ToA gaps (raw units): "
                  f"p10={np.percentile(positive,10):.7g} "
                  f"median={np.median(positive):.7g} "
                  f"p90={np.percentile(positive,90):.7g}")
        print("  top group sizes:", counts.most_common(12))
        for gid, size in counts.most_common():
            jj = ii[g == gid]
            tg = np.sort(toa[jj])
            dt = np.diff(tg)
            dt = dt[dt > 0]
            rows.append(dict(window=a.window, emitter=eid, group=gid,
                             pdws=size, aoa_population=int(Counter(pops[jj]).most_common(1)[0][0]),
                             rf_median=float(np.median(rf[jj])),
                             pw_median=float(np.median(pw[jj])),
                             aoa_median=float(np.median(aoa[jj])),
                             toa_min=float(tg.min()), toa_max=float(tg.max()),
                             within_group_gap_median=float(np.median(dt)) if len(dt) else float("nan")))
        # Plots: group colour consistent within this emitter. Noise in grey.
        unique = sorted(counts)
        colour = {gid: plt.get_cmap("tab20")(j % 20) for j, gid in enumerate(unique)}
        colours = [colour.get(int(gid), (0.65, 0.65, 0.65, 0.35)) for gid in g]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, values, title in (
            (axes[0,0], rf[ii], "RF vs ToA"),
            (axes[0,1], pw[ii], "Pulse width vs ToA"),
            (axes[1,0], aoa[ii], "AOA vs ToA")):
            ax.scatter(t, values, c=colours, s=8, linewidths=0)
            ax.set_xlabel("ToA (dataset units)")
            ax.set_ylabel(title.split(" vs ")[0])
            ax.set_title(title)
        axes[1,1].scatter(rf[ii], pw[ii], c=colours, s=8, linewidths=0)
        axes[1,1].set_xlabel("RF (dataset units)")
        axes[1,1].set_ylabel("Pulse width (dataset units)")
        axes[1,1].set_title("RF vs pulse width")
        fig.suptitle(f"Window {a.window}, truth emitter {eid}: {len(ii)} PDWs, "
                     f"{len(counts)} candidate groups; grey = unassigned")
        fig.tight_layout()
        path = f"{a.prefix}_w{a.window}_e{eid}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        print("  saved", path)
        # Compare within-group gaps to the combined truth-emitter gaps.
        # This is a diagnostic upper bound, not an operational PRI estimator.
        fig, ax = plt.subplots(figsize=(11, 4))
        if len(positive):
            ax.hist(positive, bins=70, alpha=0.6, label="All truth-emitter consecutive gaps")
        for gid, size in counts.most_common(5):
            tg = np.sort(toa[ii[g == gid]])
            d = np.diff(tg)
            d = d[d > 0]
            if len(d):
                ax.hist(d, bins=70, histtype="step", label=f"group {gid} ({size} PDWs)")
        ax.set_xlabel("Consecutive ToA gap (dataset units)")
        ax.set_ylabel("Count")
        ax.set_title(f"Emitter {eid}: consecutive ToA gaps (not inferred PRI)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = f"{a.prefix}_w{a.window}_e{eid}_gaps.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        print("  saved", path)
    if rows:
        path = f"{a.prefix}_w{a.window}_groups.csv"
        with open(path, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("\nSaved", path)
    print("NOTE: plots use ground truth for emitter selection only. ToA gap "
          "histograms are not PRI estimates and cannot prove mode changes.")


if __name__ == "__main__":
    main()
