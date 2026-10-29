"""Compare conservative RF density-blob segmentation against joint RF/PW HDBSCAN.

Run: python -u compare_rf_blobs.py turing_data/config_101.h5
Truth labels are used ONLY for scoring; never for segmentation.
A blob is an observable AOA/RF population, NOT an emitter identity.
"""
import argparse
import csv
import time
from collections import Counter
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import completeness_score, v_measure_score
from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_hdbscan_all import score


def segment_rf(values, min_mass=8, valley_ratio=0.35, max_bins=2048):
    """Split 1D RF only at deep density valleys between supported peaks.

    Adaptive bin width uses the narrower of robust IQR-based and range-based
    resolutions; the peak-mass and valley tests reject noisy micro-peaks.
    Returns one label per observation. Does not inspect PW or truth.
    """
    f = np.asarray(values, float)
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=int)
    if n < 2 * min_mass or np.ptp(f) <= 1e-12:
        return np.zeros(n, dtype=int)
    lo, hi = np.min(f), np.max(f)
    q25, q75 = np.percentile(f, [25, 75])
    iqr = q75 - q25
    # Fine enough to resolve separated narrow hopping bands, while smoothing
    # enough to avoid mistaking a broad unimodal fuzz cloud for many modes.
    fd_width = 2 * iqr / np.cbrt(n) if iqr > 0 else (hi - lo) / 128
    width = max((hi - lo) / max_bins, min(fd_width / 2, (hi - lo) / 128))
    nb = int(np.clip(np.ceil((hi - lo) / max(width, 1e-12)), 32, max_bins))
    counts, edges = np.histogram(f, bins=nb, range=(lo, hi))
    sm = gaussian_filter1d(counts.astype(float), sigma=2.0, mode="nearest")
    peaks, _ = find_peaks(sm)
    if not len(peaks):
        return np.zeros(n, dtype=int)
    peaks = peaks[np.argsort(peaks)]
    # Keep only peaks with at least min_mass observations in their local
    # support, approximated by +/- 3 smoothing bandwidths.
    support = np.array([counts[max(0,p-6):min(nb,p+7)].sum() for p in peaks])
    peaks = peaks[support >= min_mass]
    if len(peaks) < 2:
        return np.zeros(n, dtype=int)
    cuts = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 6:
            continue
        valley = left + int(np.argmin(sm[left:right+1]))
        # Split only at deep valleys relative to BOTH adjacent peak heights.
        if sm[valley] > valley_ratio * min(sm[left], sm[right]):
            continue
        cut = edges[valley+1]
        mass_left = np.count_nonzero(f <= cut)
        mass_right = n - mass_left
        if mass_left >= min_mass and mass_right >= min_mass:
            cuts.append(cut)
    if not cuts:
        return np.zeros(n, dtype=int)
    # Reject segments smaller than min_mass by merging into neighbours.
    labels = np.searchsorted(cuts, f, side="right")
    while True:
        sizes = np.bincount(labels)
        small = np.flatnonzero(sizes < min_mass)
        if not len(small) or len(sizes) == 1:
            break
        j = int(small[0])
        target = j-1 if j == len(sizes)-1 else j+1
        labels[labels == j] = target
        _, labels = np.unique(labels, return_inverse=True)
    return labels


def rf_blob_groups(x, names, populations, npop, min_mass=8, valley_ratio=0.35):
    rf = x[:, names.index("Frequency")]
    pred = np.full(len(x), -1, dtype=int)
    gid = 0
    for pop in range(npop):
        ii = np.flatnonzero(populations == pop)
        if not len(ii):
            continue
        local = segment_rf(rf[ii], min_mass=min_mass, valley_ratio=valley_ratio)
        for label in np.unique(local):
            pred[ii[local == label]] = gid
            gid += 1
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--min-mass", type=int, default=8)
    p.add_argument("--valley-ratio", type=float, default=0.35)
    p.add_argument("--focus-window", type=int, default=0)
    p.add_argument("--focus-emitters", nargs="+", type=int, default=[9,46,20])
    p.add_argument("--output", default="turing_rf_blob_comparison.csv")
    a = p.parse_args()
    if not (0 < a.valley_ratio < 1) or a.min_mass < 2:
        p.error("Require 0 < valley-ratio < 1 and min-mass >= 2")
    rows = []
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        total = len(f["data"])
        print(f"{a.file}: {total:,} PDWs", flush=True)
        for wi, start in enumerate(range(0, total, a.window_size)):
            x = np.asarray(f["data"][start:min(start+a.window_size,total)], float)
            truth = np.asarray(f["labels"][start:min(start+a.window_size,total)]).reshape(-1)
            if len(x) < 8:
                continue
            pops, npop = batch_aoa_populations(x[:, names.index("AoA")])
            for method, fn in (("joint_RF_PW", rf_pw_groups), ("RF_blobs", rf_blob_groups)):
                t0 = time.perf_counter()
                if method == "RF_blobs":
                    pred = fn(x, names, pops, npop, a.min_mass, a.valley_ratio)
                else:
                    pred = fn(x, names, pops, npop)
                elapsed = time.perf_counter() - t0
                m = score(pred, truth)
                keep = pred >= 0
                rows.append(dict(window=wi, method=method, pdws=len(x),
                                 populations=npop, seconds=elapsed, **m,
                                 v_grouped=v_measure_score(truth[keep], pred[keep]) if keep.sum() > 1 else np.nan,
                                 completeness_grouped=completeness_score(truth[keep], pred[keep]) if keep.sum() > 1 else np.nan))
                if wi == a.focus_window:
                    print(f"WINDOW {wi} {method}: {npop} AOA populations, "
                          f"{m['groups']} groups, {m['cross']} cross-emitter PDWs")
                    for eid in a.focus_emitters:
                        ids = pred[truth == eid]
                        counts = Counter(ids[ids >= 0].tolist())
                        print(f"  emitter {eid}: {len(ids)} PDWs, {len(counts)} groups, "
                              f"{sum(counts.values())} grouped, largest={max(counts.values(),default=0)}")
            if (wi+1) % 10 == 0 or start + len(x) == total:
                print(f"Processed {start+len(x):,}/{total:,}", flush=True)
    with open(a.output,"w",newline="") as fh:
        w = csv.DictWriter(fh,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmethod purity% coverage% cross noise groups groups_per_10k seconds V_grouped completeness_grouped")
    for method in ("joint_RF_PW","RF_blobs"):
        rr = [r for r in rows if r["method"] == method]
        n = sum(r["pdws"] for r in rr)
        grouped = sum(r["grouped"] for r in rr)
        correct = sum(r["correct"] for r in rr)
        print(f"{method:12s} {100*correct/grouped if grouped else 0:7.3f} "
              f"{100*grouped/n:8.3f} {sum(r['cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} {sum(r['groups'] for r in rr):7d} "
              f"{10000*sum(r['groups'] for r in rr)/n:8.1f} "
              f"{sum(r['seconds'] for r in rr):7.2f} "
              f"{np.nanmean([r['v_grouped'] for r in rr]):.3f} "
              f"{np.nanmean([r['completeness_grouped'] for r in rr]):.3f}")
    print("Caution: blob coverage includes all assigned pulses; a single mixed blob "
          "may have high coverage but poor purity. Inspect cross-emitter errors.")
    print("Saved", a.output)


if __name__ == "__main__":
    main()
