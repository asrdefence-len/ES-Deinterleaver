"""Diagnose a timing lattice and long gaps for one truth-selected emitter.

Run: python -u analyse_emitter9_pri.py turing_data/config_101.h5
Truth labels select the example only. Time units remain raw dataset units.
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--emitter", type=int, default=9)
    ap.add_argument("--prefix", default="emitter9_pri")
    a = ap.parse_args()
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        start = a.window * a.window_size
        stop = min(start + a.window_size, len(f["data"]))
        x = np.asarray(f["data"][start:stop], float)
        labels = np.asarray(f["labels"][start:stop]).reshape(-1)
    ii = np.flatnonzero(labels == a.emitter)
    if len(ii) < 3:
        raise SystemExit("Not enough emitter PDWs")
    t = np.sort(x[ii, names.index("ToA")])
    d = np.diff(t)
    positive = d[d > 0]
    if not len(positive):
        raise SystemExit("No positive intervals")
    # Exclude exceptional gaps when estimating a local lattice. A global
    # pulse timing lattice is NOT assumed to persist through burst breaks.
    cutoff = np.percentile(positive, 95)
    local = positive[positive <= cutoff]
    # Search for the fundamental spacing near the shortest commonly seen
    # intervals; evaluate multiples 1..16 and penalize over-small candidates
    # by limiting search to the vicinity of the lower interval mode.
    lower = np.percentile(local, 1)
    candidates = np.linspace(0.90 * lower, 1.10 * lower, 2001)
    scores = []
    for p in candidates:
        k = np.maximum(1, np.rint(local / p))
        residual = local - k * p
        scores.append(np.median(np.abs(residual)))
    p0 = float(candidates[int(np.argmin(scores))])
    k = np.maximum(1, np.rint(local / p0)).astype(int)
    # Least-squares refinement for gaps close to integer multiples.
    good = np.abs(local - k * p0) < 0.12 * p0
    p = float(np.dot(k[good], local[good]) / np.dot(k[good], k[good])) if good.any() else p0
    multiples = np.maximum(1, np.rint(local / p)).astype(int)
    residuals = local - multiples * p
    print(f"Emitter {a.emitter}, window {a.window}: {len(t)} PDWs, {len(d)} intervals")
    print(f"Local interval cutoff (p95): {cutoff:.9f} raw units")
    print(f"Estimated candidate fundamental interval: {p:.9f} raw units")
    print(f"Local residual |error|: median={np.median(np.abs(residuals)):.6f}, "
          f"p90={np.percentile(np.abs(residuals),90):.6f}, "
          f"p99={np.percentile(np.abs(residuals),99):.6f}")
    print(f"Within 5% of a multiple: {100*np.mean(np.abs(residuals) < .05*p):.2f}%")
    print(f"Within 10% of a multiple: {100*np.mean(np.abs(residuals) < .10*p):.2f}%")
    print("Observed multiples (local intervals):")
    vals, counts = np.unique(multiples, return_counts=True)
    for v, c in zip(vals, counts):
        if v <= 16:
            print(f"  {v:2d} x candidate interval: {c:5d} ({100*c/len(local):5.1f}%)")
    # Report large breaks and their positions, including temporal context.
    threshold = max(10 * p, np.percentile(positive, 99))
    breaks = np.flatnonzero(d > threshold)
    print(f"Large gaps > {threshold:.6f} raw units: {len(breaks)}")
    for j in breaks[np.argsort(d[breaks])[::-1][:15]]:
        print(f"  pulse {j} -> {j+1}: gap={d[j]:.9f}, ToA={t[j]:.9f} -> {t[j+1]:.9f}")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0,0].hist(local, bins=120)
    ax[0,0].set(xlabel="Adjacent interval (raw units)",ylabel="Count",title="Local arrival intervals (<= p95)")
    ax[0,1].hist(residuals, bins=100)
    ax[0,1].set(xlabel="Gap minus nearest multiple (raw units)",ylabel="Count",title="Candidate timing-lattice residual")
    ax[1,0].scatter(np.arange(len(d)), d, s=5)
    ax[1,0].set_yscale("log")
    ax[1,0].set(xlabel="Consecutive pulse index",ylabel="Interval (raw units, log scale)",title="Intervals and burst breaks")
    ax[1,1].scatter(np.arange(len(d)), d, s=5)
    ax[1,1].set_ylim(0, min(max(100, 10*p), np.percentile(d,99.5)*1.2))
    ax[1,1].set(xlabel="Consecutive pulse index",ylabel="Interval (raw units)",title="Ordinary intervals across observation")
    fig.suptitle(f"Emitter {a.emitter}: timing structure, not a confirmed PRI")
    fig.tight_layout()
    path = f"{a.prefix}_w{a.window}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("Saved", path)
    print("Caution: multiples may reflect missed observations, intentional pulse omissions, "
          "or a true repeated timing grid. The method does not distinguish these causes.")


if __name__ == "__main__":
    main()
