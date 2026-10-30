"""Plot a PRI interval/residual histogram for one existing RF blob.
Run: python -u pri_residual_histogram.py turing_data/config_101.h5 --window 0 --blob 9
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from deinterleaver_v0 import discover

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--blob", type=int, default=9)
    p.add_argument("--period", type=float)
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument("--min-count", type=int, default=5)
    args = p.parse_args()
    with h5py.File(args.dataset) as h:
        lo = args.window * 10000
        hi = min(lo + 10000, len(h["data"]))
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in h["metadata/feature_names"][:]]
        x = np.asarray(h["data"][lo:hi], float)
    populations, n_pop = batch_aoa_populations(x[:, names.index("AoA")])
    groups = group_window(x, names, populations, n_pop, 8, .20, 6., 8)
    t = np.sort(x[groups == args.blob, names.index("ToA")])
    if len(t) < 3:
        p.error("Too few pulses in requested blob")
    if args.period is None:
        tracks, _ = discover(t, 20., 200., 2., 20, 4, .1)
        if not tracks:
            p.error("No candidate period; supply --period")
        period = tracks[0].period
    else:
        period = args.period
    if period <= 0:
        p.error("Period must be positive")
    gaps = np.diff(t)
    k = np.rint(gaps / period).astype(int)
    residual = gaps - k * period
    use = (gaps > 0) & (k >= 1) & (k <= args.max_k)
    k, residual = k[use], residual[use]
    counts = np.array([(k == j).sum() for j in range(1, args.max_k + 1)])
    variances = np.array([
        np.var(residual[k == j], ddof=1) if counts[j - 1] >= 2 else np.nan
        for j in range(1, args.max_k + 1)
    ])
    eligible = counts >= args.min_count
    print(f"Blob {args.blob}: PDWs={len(t)}, PRI={period:.4f}, gaps_used={len(k)}")
    print(" k  count  residual_variance")
    for j, (n, v) in enumerate(zip(counts, variances), 1):
        print(f"{j:2d} {n:6d} {v:18.4f}")
    slope = None
    if eligible.sum() >= 3:
        ks = np.arange(1, args.max_k + 1)[eligible]
        slope, intercept = np.polyfit(
            ks, variances[eligible], 1, w=np.sqrt(counts[eligible] - 1)
        )
        print(f"Weighted variance slope={slope:.4f}; intercept={intercept:.4f}")
        if slope > 0:
            print(f"Conditional jitter estimate={100 * np.sqrt(slope) / period:.3f}%")
        else:
            print("Nonpositive slope: no jitter estimate")
    else:
        print("Insufficient populated k bins: jitter estimate inconclusive")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    hist, xedge, yedge = np.histogram2d(
        k, residual,
        bins=[np.arange(.5, args.max_k + 1.5), np.linspace(-period / 2, period / 2, 49)]
    )
    m = ax.pcolormesh(xedge, yedge, hist.T, cmap="viridis")
    fig.colorbar(m, ax=ax, label="Observed intervals")
    ax.set(xlabel="Inferred PRI intervals (k)", ylabel="Timing residual (raw TOA units)",
           title="PRI interval / residual histogram")
    kval = np.arange(1, args.max_k + 1)
    bx.scatter(kval[np.isfinite(variances)], variances[np.isfinite(variances)])
    if slope is not None:
        bx.plot(kval, intercept + slope * kval, linestyle="--")
    bx.set(xlabel="Inferred PRI intervals (k)",
           ylabel="Residual variance (raw TOA units squared)",
           title="Residual variance versus k")
    filename = f"pri_residual_histogram_w{args.window}_b{args.blob}.png"
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print("Saved:", filename)
    print("Caution: estimated k may be wrong; mixed trains, harmonics, and PRI changes "
          "can invalidate the slope. This is an exploratory estimate, not ground truth.")

if __name__ == "__main__":
    main()
