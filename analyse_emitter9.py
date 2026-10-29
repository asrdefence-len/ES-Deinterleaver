"""Inspect timing and RF/PW structure of a truth-selected emitter (offline diagnostic only).

Run: python -u analyse_emitter9.py turing_data/config_101.h5
Truth selects the example ONLY; no operational classification is made.
"""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def describe(label, a):
    a = np.asarray(a, float)
    if not len(a):
        print(label + ": none")
        return
    print(f"{label}: n={len(a)}, min={a.min():.12g}, p01={np.percentile(a,1):.12g}, "
          f"p10={np.percentile(a,10):.12g}, median={np.median(a):.12g}, "
          f"p90={np.percentile(a,90):.12g}, p99={np.percentile(a,99):.12g}, max={a.max():.12g}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--emitter", type=int, default=9)
    p.add_argument("--n", type=int, default=100, help="consecutive PDWs to zoom")
    p.add_argument("--prefix", default="emitter9")
    a = p.parse_args()
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["metadata/feature_names"][:]]
        start = a.window * a.window_size
        stop = min(start + a.window_size, len(f["data"]))
        x = np.asarray(f["data"][start:stop], float)
        truth = np.asarray(f["labels"][start:stop]).reshape(-1)
    idx = np.flatnonzero(truth == a.emitter)
    if len(idx) < 2:
        raise SystemExit("Fewer than two matching PDWs")
    t = x[:, names.index("ToA")][idx]
    rf = x[:, names.index("Frequency")][idx]
    pw = x[:, names.index("PulseWidth")][idx]
    aoa = x[:, names.index("AoA")][idx]
    order = np.argsort(t, kind="stable")
    idx, t, rf, pw, aoa = (v[order] for v in (idx, t, rf, pw, aoa))
    dt = np.diff(t)
    print(f"Window {a.window}, truth emitter {a.emitter}: {len(t)} PDWs; raw dataset units")
    for name, v in (("ToA",t),("RF",rf),("PW",pw),("AOA",aoa),("adjacent ToA gaps",dt)):
        describe(name,v)
    print(f"Exact duplicate ToAs (adjacent pairs): {np.count_nonzero(dt == 0)}")
    print(f"Nonpositive gaps: {np.count_nonzero(dt <= 0)}")
    positive = dt[dt > 0]
    if len(positive):
        for fraction in (0.001, 0.01, 0.1):
            threshold = fraction * np.median(positive)
            print(f"Gaps <= {fraction:g} x median positive gap ({threshold:.12g}): "
                  f"{np.count_nonzero((dt >= 0) & (dt <= threshold))}")
    print(f"RF-PW Pearson correlation: {np.corrcoef(rf,pw)[0,1]:.5f}")
    if len(rf) > 2:
        print(f"RF lag-1 correlation: {np.corrcoef(rf[:-1],rf[1:])[0,1]:.5f}")
        print(f"PW lag-1 correlation: {np.corrcoef(pw[:-1],pw[1:])[0,1]:.5f}")
    n = min(a.n, len(t))
    path = f"{a.prefix}_w{a.window}_first{n}.csv"
    with open(path,"w",newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["local_index","ToA_raw","delta_ToA_raw","Frequency_raw","PulseWidth_raw","AoA_raw"])
        for j in range(n):
            w.writerow([int(idx[j]),format(t[j],".17g"),format(dt[j-1],".17g") if j else "",
                        format(rf[j],".17g"),format(pw[j],".17g"),format(aoa[j],".17g")])
    print("Saved",path)
    fig,axs = plt.subplots(3,2,figsize=(14,11))
    for row, (y,label) in enumerate(((rf,"RF"),(pw,"PW"),(aoa,"AOA"))):
        axs[row,0].scatter(t[:n]-t[0],y[:n],s=15)
        axs[row,0].set(xlabel="ToA relative to first pulse (raw units)",ylabel=label,title=f"First {n}: {label} vs ToA")
    axs[0,1].hist(positive,bins=100)
    axs[0,1].set(xlabel="Positive adjacent ToA gap (raw units)",ylabel="Count",title="All adjacent positive gaps")
    axs[1,1].scatter(rf[:n],pw[:n],s=16)
    axs[1,1].set(xlabel="RF (raw units)",ylabel="PW (raw units)",title=f"First {n}: RF vs PW")
    axs[2,1].scatter(np.arange(1,n),dt[:n-1],s=15)
    axs[2,1].set(xlabel="Pulse index",ylabel="Adjacent ToA gap (raw units)",title="First pulses: arrival intervals")
    fig.suptitle(f"Emitter {a.emitter} diagnostic; truth used only to select observations")
    fig.tight_layout()
    png=f"{a.prefix}_w{a.window}_timing.png"
    fig.savefig(png,dpi=170)
    plt.close(fig)
    print("Saved",png)
    print("Interpretation: exact/near-coincident ToAs may suggest multiple detections; "
          "neither PDWs nor these plots alone establish OFDM, QAM, or communications.")


if __name__ == "__main__":
    main()
