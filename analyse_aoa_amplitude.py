"""Diagnostic AOA residuals versus amplitude for Turing PDWs.

Truth labels are used ONLY for offline receiver characterisation, never clustering.
The -100 dB ambient level is a simulator parameter, NOT calibrated per-PDW SNR.
Uses local robust circular bearing references to reduce moving-emitter bias.
"""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def reference_bearings(toa, aoa, labels, block, minimum):
    """Within-emitter, within-time-block circular mean, leave-one-out.

    The emitter labels are evaluation-only. Small blocks are excluded rather
    than assigning a noisy reference. This is not absolute AOA accuracy.
    """
    ref = np.full(len(aoa), np.nan)
    for emitter in np.unique(labels):
        idx = np.flatnonzero(labels == emitter)
        if len(idx) < minimum:
            continue
        idx = idx[np.argsort(toa[idx], kind="stable")]
        for start in range(0, len(idx), block):
            ii = idx[start:start + block]
            if len(ii) < minimum:
                continue
            r = np.deg2rad(aoa[ii])
            ss = np.sin(r).sum() - np.sin(r)
            cc = np.cos(r).sum() - np.cos(r)
            strength = np.hypot(ss, cc) / np.maximum(len(ii) - 1, 1)
            ok = strength >= 0.5
            ref[ii[ok]] = np.rad2deg(np.arctan2(ss[ok], cc[ok]))
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--minimum", type=int, default=20)
    p.add_argument("--bin-width", type=float, default=5.0)
    p.add_argument("--noise-floor-db", type=float, default=-100.0)
    p.add_argument("--prefix", default=None)
    a = p.parse_args()
    if a.minimum < 3 or a.block < a.minimum or a.bin_width <= 0:
        p.error("Require block >= minimum >= 3 and bin-width > 0")
    prefix = a.prefix or Path(a.file).stem + "_aoa_amplitude"
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        x = np.asarray(f["data"][:], dtype=float)
        labels = np.asarray(f["labels"][:]).reshape(-1)
    if len(labels) != len(x):
        raise ValueError("Label and PDW lengths differ")
    toa = x[:, names.index("ToA")]
    aoa = x[:, names.index("AoA")]
    amp = x[:, names.index("Amplitude")]
    valid = np.isfinite(toa) & np.isfinite(aoa) & np.isfinite(amp)
    ref = reference_bearings(toa[valid], aoa[valid], labels[valid],
                             a.block, a.minimum)
    residual = np.abs(wrap(aoa[valid] - ref))
    av = amp[valid]
    usable = np.isfinite(residual)
    print(f"PDWs: {len(x):,}; reference-valid: {usable.sum():,} "
          f"({100 * usable.mean():.1f}%)")
    print("Residual is relative to local same-emitter bearing, not known true AOA.")
    print(f"Simulator ambient noise parameter: {a.noise_floor_db:g} dB "
          "(not a calibrated SNR).")
    if not np.any(usable):
        raise SystemExit("No valid local bearing references; lower --minimum")
    av, residual = av[usable], residual[usable]
    low = np.floor(av.min() / a.bin_width) * a.bin_width
    high = np.ceil(av.max() / a.bin_width) * a.bin_width + a.bin_width
    edges = np.arange(low, high + 0.5 * a.bin_width, a.bin_width)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (av >= left) & (av < right)
        v = residual[mask]
        rows.append(dict(amp_low_db=float(left), amp_high_db=float(right),
                         relative_to_ambient_db=float((left + right) / 2 - a.noise_floor_db),
                         n=int(len(v)), median_abs_deg=float(np.median(v)) if len(v) else "",
                         p68_abs_deg=float(np.percentile(v, 68)) if len(v) else "",
                         p95_abs_deg=float(np.percentile(v, 95)) if len(v) else ""))
    with open(prefix + ".csv", "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    populated = [r for r in rows if r["n"] >= 100]
    print("Amplitude bins with >=100 observations:")
    for r in populated:
        print(f" {r['amp_low_db']:7.1f} to {r['amp_high_db']:7.1f} dB "
              f"n={r['n']:7d} median={r['median_abs_deg']:6.2f} "
              f"p68={r['p68_abs_deg']:6.2f} p95={r['p95_abs_deg']:6.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(av, bins=edges)
    ax.axvline(a.noise_floor_db, linestyle="--", label="Simulator ambient level")
    ax.set(xlabel="Reported amplitude (dB)", ylabel="PDW count",
           title="Turing received amplitude distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix + "_amplitude.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    if populated:
        centers = [(r["amp_low_db"] + r["amp_high_db"]) / 2 for r in populated]
        for key, label in (("median_abs_deg", "Median"), ("p68_abs_deg", "68th percentile"),
                           ("p95_abs_deg", "95th percentile")):
            ax.plot(centers, [r[key] for r in populated], marker=".", label=label)
    ax.axvline(a.noise_floor_db, linestyle="--", label="Simulator ambient level")
    ax.set(xlabel="Reported amplitude (dB)", ylabel="Absolute local AOA residual (degrees)",
           title="AOA dispersion versus amplitude (truth-assisted diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix + "_aoa_residual.png", dpi=160)
    plt.close(fig)
    print(f"Saved {prefix}.csv, {prefix}_amplitude.png, {prefix}_aoa_residual.png")
    print("CAUTION: moving emitters, correlated OU perturbations, and local-reference "
          "error can affect these residuals. Do not infer receiver sigma or SNR directly.")


if __name__ == "__main__":
    main()
