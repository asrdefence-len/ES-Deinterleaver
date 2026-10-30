"""Truth-blind timing diagnostics inside frozen multiscale RF blobs.

Truth labels select interesting blobs for reporting ONLY; never enter grouping
or PRI candidate generation. PRI peaks are not confirmed pulse trains.
"""
import argparse
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


def candidates(times, max_lag=12, bins=300, max_delta=None):
    t = np.sort(np.asarray(times, float))
    if len(t) < 4:
        return [], np.array([]), np.array([])
    positive = np.diff(t)
    positive = positive[positive > 0]
    if not len(positive):
        return [], np.array([]), np.array([])
    # Restrict search to a robust time scale; user may override it.
    limit = float(max_delta) if max_delta is not None else float(
        np.quantile(positive, .75) * 20)
    if limit <= 0:
        return [], np.array([]), np.array([])
    diffs = []
    for lag in range(1, min(max_lag, len(t) - 1) + 1):
        d = t[lag:] - t[:-lag]
        diffs.append(d[(d > 0) & (d <= limit)])
    d = np.concatenate(diffs) if diffs else np.array([])
    hist, edges = np.histogram(d, bins=bins, range=(0, limit))
    peaks, _ = find_peaks(hist, prominence=max(3, hist.max() * .06),
                          distance=2)
    peaks = sorted(peaks, key=lambda p: hist[p], reverse=True)[:8]
    result = [(float((edges[p] + edges[p+1])/2), int(hist[p]))
              for p in peaks]
    return result, hist, edges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--emitters", type=int, nargs="+", default=[46, 58, 8, 22])
    p.add_argument("--top", type=int, default=3,
                   help="Most populated blobs for each evaluation emitter")
    p.add_argument("--max-lag", type=int, default=12)
    p.add_argument("--bins", type=int, default=300)
    p.add_argument("--max-delta", type=float, default=None,
                   help="Maximum TOA difference, in dataset TOA units")
    p.add_argument("--out", default="rf_blob_timing")
    a = p.parse_args()
    if a.window < 0 or a.window_size < 8 or a.top < 1:
        p.error("Invalid window/top settings")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.file) as h:
        start = a.window * a.window_size
        stop = min(start + a.window_size, len(h["data"]))
        if stop - start < 8:
            p.error("Window empty or too small")
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in h["metadata/feature_names"][:]]
        x = np.asarray(h["data"][start:stop], float)
        truth = np.asarray(h["labels"][start:stop]).reshape(-1)
    rf = x[:, names.index("Frequency")]
    toa = x[:, names.index("ToA")]
    aoa = x[:, names.index("AoA")]
    pops, npop = batch_aoa_populations(aoa)
    labels = group_window(x, names, pops, npop, 8, .20, 6.0, 8)
    print(f"Window {a.window}: {len(x)} PDWs, {npop} AOA populations, "
          f"{len(set(labels[labels >= 0]))} RF blobs", flush=True)
    chosen = []
    for emitter in a.emitters:
        ids = labels[truth == emitter]
        counts = Counter(ids[ids >= 0])
        print(f"Evaluation emitter {emitter}: {len(ids)} PDWs, "
              f"{len(counts)} blobs; largest {counts.most_common(a.top)}",
              flush=True)
        chosen.extend(g for g, _ in counts.most_common(a.top))
    for gid in dict.fromkeys(chosen):
        idx = np.flatnonzero(labels == gid)
        if len(idx) < 4:
            continue
        peaks, hist, edges = candidates(
            toa[idx], max_lag=a.max_lag, bins=a.bins,
            max_delta=a.max_delta)
        mix = Counter(truth[idx].tolist())
        print(f"Blob {gid}: {len(idx)} PDWs, RF [{rf[idx].min():.3f}, "
              f"{rf[idx].max():.3f}], truth composition {mix.most_common(5)}",
              flush=True)
        print(f"  Candidate TOA differences (raw TOA units; count): "
              f"{peaks if peaks else 'none'}", flush=True)
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        axs[0].scatter(toa[idx], rf[idx], s=5, alpha=.65)
        axs[0].set(xlabel="TOA (dataset units)", ylabel="RF (dataset units)",
                   title=f"Window {a.window}, RF blob {gid}: RF versus TOA")
        if len(hist):
            centers = (edges[:-1] + edges[1:]) / 2
            axs[1].plot(centers, hist)
            for value, _ in peaks[:5]:
                axs[1].axvline(value, linestyle=":", linewidth=1)
        axs[1].set(xlabel="TOA difference (dataset units)",
                   ylabel="Pair count (lag-limited)",
                   title="Candidate timing periodicities (not verified PRIs)")
        fig.tight_layout()
        path = out / f"window_{a.window}_blob_{gid}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  Plot: {path}", flush=True)
    print("No truth labels used in AOA/RF grouping or timing candidate detection.")
    print("Warning: histogram peaks include harmonics, missed pulses and "
          "cross-emitter coincidences; no pulse-train extraction yet.")


if __name__ == "__main__":
    main()
