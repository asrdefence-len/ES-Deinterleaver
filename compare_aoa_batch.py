"""Batch circular AOA density versus online AOA grouping, followed by RF/PW HDBSCAN.

Usage: python -u compare_aoa_batch.py turing_data/config_101.h5
Truth labels are ONLY used for evaluation. Independent 10,000-PDW windows.
This tests batch density estimation, NOT persistent emitter identification.
"""
import argparse
import csv
import time
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import HDBSCAN
from sklearn.metrics import v_measure_score, completeness_score, adjusted_rand_score
from compare_hdbscan_all import feature_matrix, score
from compare_aoa_online_all import online_aoa_populations


def batch_aoa_populations(aoa, bin_deg=0.25, smooth_deg=0.5,
                          min_peak_count=8, min_separation_deg=1.5):
    """Estimate circular AOA density peaks from a complete PDW window.

    Circular histogram and wrapped Gaussian smoothing. Peak heights are
    counts per bin, not probability densities. Every PDW is assigned to
    its nearest peak (no truth labels; no AOA hard rejection). RF/PW
    HDBSCAN can separate different emitters within the same AOA population.
    """
    nbin = int(round(360.0 / bin_deg))
    if nbin < 12:
        raise ValueError("AOA bin width too large")
    width = 360.0 / nbin
    a = np.asarray(aoa, float) % 360.0
    bins = np.floor(a / width).astype(int) % nbin
    hist = np.bincount(bins, minlength=nbin).astype(float)
    density = gaussian_filter1d(hist, sigma=max(0.01, smooth_deg / width),
                                mode="wrap")
    # Circular peak detection: pad one full revolution on either side.
    extended = np.tile(density, 3)
    separation_bins = max(1, int(round(min_separation_deg / width)))
    peaks, _ = find_peaks(extended, distance=separation_bins)
    peaks = peaks[(peaks >= nbin) & (peaks < 2 * nbin)] - nbin
    peaks = peaks[density[peaks] >= min_peak_count * width /
                  max(2.5066 * smooth_deg, width)]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(density))])
    # Resolve equal-height or near-duplicate circular peaks explicitly.
    peaks = sorted(peaks, key=lambda p: (-density[p], p))
    selected = []
    for p in peaks:
        if all(min(abs(p - q), nbin - abs(p - q)) >= separation_bins
               for q in selected):
            selected.append(int(p))
    centers = (np.array(sorted(selected)) + 0.5) * width
    # Broadcast in chunks to keep memory bounded for many peaks.
    labels = np.empty(len(a), dtype=int)
    for start in range(0, len(a), 10000):
        aa = a[start:start + 10000, None]
        distance = np.abs((aa - centers[None, :] + 180) % 360 - 180)
        labels[start:start + len(aa)] = np.argmin(distance, axis=1)
    return labels, len(centers)


def rf_pw_groups(x, names, populations, npop, mcs=8, ms=8):
    z = feature_matrix(x, names)[:, 2:]
    pred = np.full(len(x), -1, dtype=int)
    gid = 0
    for pop in range(npop):
        ii = np.flatnonzero(populations == pop)
        if len(ii) < mcs:
            continue
        local = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(z[ii])
        for lab in np.unique(local[local >= 0]):
            pred[ii[local == lab]] = gid
            gid += 1
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--gate", type=float, default=3.0)
    p.add_argument("--bin-deg", type=float, default=0.25)
    p.add_argument("--smooth-deg", type=float, default=0.5)
    p.add_argument("--min-peak-count", type=int, default=8)
    p.add_argument("--min-separation-deg", type=float, default=1.5)
    p.add_argument("--output", default="turing_aoa_batch_results.csv")
    a = p.parse_args()
    if min(a.window_size, a.gate, a.bin_deg, a.smooth_deg,
           a.min_peak_count, a.min_separation_deg) <= 0:
        p.error("All tuning parameters must be positive")
    rows = []
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        n = len(f["data"])
        print(f"{a.file}: {n:,} PDWs", flush=True)
        for wi, start in enumerate(range(0, n, a.window_size)):
            stop = min(start + a.window_size, n)
            if stop - start < 8:
                continue
            x = np.asarray(f["data"][start:stop], float)
            truth = np.asarray(f["labels"][start:stop]).reshape(-1)
            aoa, toa = x[:, names.index("AoA")], x[:, names.index("ToA")]
            for method in ("online_fixed3", "batch_density"):
                t0 = time.perf_counter()
                if method == "online_fixed3":
                    pops, npop = online_aoa_populations(aoa, toa, a.gate)
                else:
                    pops, npop = batch_aoa_populations(
                        aoa, a.bin_deg, a.smooth_deg,
                        a.min_peak_count, a.min_separation_deg)
                pred = rf_pw_groups(x, names, pops, npop)
                elapsed = time.perf_counter() - t0
                metrics = score(pred, truth)
                keep = pred >= 0
                rows.append(dict(window=wi, pdws=len(x), method=method,
                                 populations=npop, seconds=elapsed, **metrics,
                                 v_grouped=v_measure_score(truth[keep], pred[keep])
                                 if keep.sum() >= 2 else float("nan"),
                                 completeness_grouped=completeness_score(
                                     truth[keep], pred[keep]) if keep.sum() >= 2 else float("nan"),
                                 ari_grouped=adjusted_rand_score(
                                     truth[keep], pred[keep]) if keep.sum() >= 2 else float("nan")))
            if (wi + 1) % 10 == 0 or stop == n:
                print(f"processed {stop:,}/{n:,}", flush=True)
    with open(a.output, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nmethod purity% coverage% cross noise groups populations groups_per_10k seconds V_grouped completeness_grouped")
    for method in ("online_fixed3", "batch_density"):
        rr = [r for r in rows if r["method"] == method]
        total = sum(r["pdws"] for r in rr)
        grouped = sum(r["grouped"] for r in rr)
        print(f"{method:14s} {100*sum(r['correct'] for r in rr)/grouped:7.3f} "
              f"{100*grouped/total:8.3f} {sum(r['cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} "
              f"{sum(r['groups'] for r in rr):7d} "
              f"{sum(r['populations'] for r in rr):7d} "
              f"{10000*sum(r['groups'] for r in rr)/total:8.1f} "
              f"{sum(r['seconds'] for r in rr):7.2f} "
              f"{np.mean([r['v_grouped'] for r in rr]):.3f} "
              f"{np.mean([r['completeness_grouped'] for r in rr]):.3f}")
    print("Grouped-only metrics exclude unassigned PDWs and are NOT official challenge scores.")
    print("Batch density uses a full window; it introduces window latency and may split moving emitters.")
    print("Saved", a.output)


if __name__ == "__main__":
    main()
