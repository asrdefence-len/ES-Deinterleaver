"""Inspect an existing AOA/multiscale-RF blob for a frequency-density step.

Example:
 python diagnose_rf_histogram_split.py turing_data/config_101.h5 \
     --window 40 --blob 40 --low 9000 --high 12000

The split is fitted using observed frequencies ONLY; truth is read afterward
for evaluation. A density boundary proposes subgroups, not physical emitter
identities. This diagnostic does not modify the grouping baseline.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


def fit_density_step(frequency, low, high, bin_width=25., min_count=30):
    """Two constant RF densities separated at a candidate histogram edge.

    Observed support bounds are common to the null and alternative. All
    pulses are retained, including those just outside the displayed range.
    This models a density CHANGE rather than an empty gap or two peaks.
    Delta BIC penalizes one density-proportion and one boundary parameter.
    It is a descriptive model comparison, not a calibrated significance test.
    """
    f = np.asarray(frequency, float)
    if (f.ndim != 1 or not np.all(np.isfinite(f))
            or not np.all(np.isfinite([low, high, bin_width]))
            or not low < high or bin_width <= 0
            or not isinstance(min_count, int) or min_count < 2):
        raise ValueError("Invalid frequencies or density-step parameters")
    if len(f) < 2 * min_count or np.ptp(f) <= 0:
        return dict(supported=False, reason="insufficient_support")
    lo, hi, n = float(f.min()), float(f.max()), len(f)
    null_ll = -n * np.log(hi - lo)
    options = []
    for cut in np.arange(low, high + bin_width * 1e-8, bin_width):
        if not lo < cut < hi:
            continue
        left = int(np.count_nonzero(f < cut))
        right = n - left
        if min(left, right) < min_count:
            continue
        ll = left * np.log(left / n / (cut - lo)) + right * np.log(right / n / (hi - cut))
        options.append((float(ll), float(cut), left, right))
    if not options:
        return dict(supported=False, reason="no_supported_histogram_edge")
    ll, cut, left, right = max(options)
    improvement = float(2 * (ll - null_ll) - 2 * np.log(n))
    return dict(supported=improvement > 10., reason="density_step" if improvement > 10. else "weak_evidence",
                cut=cut, left_count=left, right_count=right,
                delta_bic=improvement, density_left=left / (cut - lo),
                density_right=right / (hi - cut), observation_min=lo, observation_max=hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--blob", type=int, default=40)
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--low", type=float, default=9000.)
    ap.add_argument("--high", type=float, default=12000.)
    ap.add_argument("--bin-width", type=float, default=25.)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--output", default="results/rf_histogram_split")
    a = ap.parse_args()
    if a.window < 0 or a.window_size < 8:
        ap.error("Invalid window")
    with h5py.File(a.file) as h:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in h["metadata/feature_names"][:]]
        start = a.window * a.window_size
        x = np.asarray(h["data"][start:start + a.window_size], float)
        if len(x) < 8:
            ap.error("Window unavailable")
        pops, npop = batch_aoa_populations(x[:, names.index("AoA")])
        blob = group_window(x, names, pops, npop, 8, .20, 6., 8)
        idx = np.flatnonzero(blob == a.blob)
        if not len(idx):
            ap.error("Blob unavailable")
        f = x[idx, names.index("Frequency")]
        t = x[idx, names.index("ToA")]
        fitted = fit_density_step(f, a.low, a.high, a.bin_width, a.min_count)
        if "cut" not in fitted:
            ap.error(f"No fit: {fitted}")
        cut = fitted["cut"]
        children = (f >= cut).astype(int)
        # Evaluation boundary: no truth has been read before this point.
        truth = np.asarray(h["labels"][start:start + len(x)]).reshape(-1)[idx] if "labels" in h else None
    sensitivity = [dict(bin_width=bw, origin_shift=shift,
        **fit_density_step(f, a.low + shift, a.high, bw, a.min_count))
        for bw in (10., 25., 50., 100.) for shift in (0., bw / 2)]
    result = dict(file=Path(a.file).name, window=a.window, blob=a.blob, pdws=len(f),
                  source="AOA + multiscale RF grouping", fitted=fitted, sensitivity=sensitivity,
                  histogram_low=a.low, histogram_high=a.high,
                  histogram_outside_count=int(np.sum((f < a.low) | (f > a.high))))
    if truth is not None:
        original = Counter(truth.tolist())
        components = [Counter(truth[children == c].tolist()) for c in (0, 1)]
        result["evaluation"] = dict(original_counts=dict(original),
            child_counts=[dict(c) for c in components],
            original_purity_pct=100 * max(original.values()) / len(f),
            split_purity_pct=100 * sum(max(c.values()) for c in components) / len(f),
            minority_after_split=sum(sum(c.values()) - max(c.values()) for c in components))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    edges = np.arange(a.low, a.high + a.bin_width * 1e-8, a.bin_width)
    hist, edges = np.histogram(f, bins=edges)
    np.savetxt(out / "histogram.csv", np.column_stack((edges[:-1], edges[1:], hist)),
               delimiter=",", header="lower_edge,upper_edge,pulse_count", comments="", fmt=["%.6f", "%.6f", "%d"])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    ink, orange, blue = "#334155", "#d55e00", "#0072b2"
    for ax in (axes[0, 0], axes[1, 0]):
        ax.stairs(hist, edges, fill=True, color=ink, alpha=.8)
        ax.axvline(cut, color=orange, lw=2, ls="--", label=f"Fitted split: {cut:,.0f}")
        ax.set(xlabel="Frequency (dataset units)", ylabel=f"PDWs per {a.bin_width:g}-unit bin")
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].set(xlim=(a.low, a.high), title="Frequency histogram — labels not used")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set(xlim=(cut - 200, cut + 200), title="Boundary detail — abrupt density drop")
    origin = float(t.min())
    axes[0, 1].scatter(t - origin, f, s=10, color=ink, alpha=.65)
    axes[0, 1].axhline(cut, color=orange, lw=1.5, ls="--")
    axes[0, 1].set(title="RF versus TOA — labels not used", ylabel="Frequency (dataset units)",
                   xlabel="TOA minus common origin (dataset units)", ylim=(a.low - 30, a.high + 30))
    for child, color in ((0, blue), (1, orange)):
        k = children == child
        axes[1, 1].scatter(t[k] - origin, f[k], s=10, alpha=.65, color=color,
                          label=f"Child {'A' if child == 0 else 'B'}: {k.sum()} PDWs")
    axes[1, 1].axhline(cut, color=ink, lw=1, ls="--")
    axes[1, 1].set(title="Split by frequency alone", ylabel="Frequency (dataset units)",
                   xlabel="TOA minus common origin (dataset units)", ylim=(a.low - 30, a.high + 30))
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f"Window {a.window}, blob {a.blob}: {len(f)} PDWs | blind density-step split", fontsize=16)
    note = f"Display range {a.low:g}–{a.high:g}; {result['histogram_outside_count']} pulse(s) outside it are retained in fitting and splitting."
    if truth is not None:
        note += f"  Post-fit truth check: purity {result['evaluation']['split_purity_pct']:.2f}%."
    fig.supxlabel(note, fontsize=9)
    dest = out / f"rf_histogram_w{a.window}_blob{a.blob}.png"
    fig.savefig(dest, dpi=160, facecolor="white")
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print(f"Plot: {dest}")


if __name__ == "__main__":
    main()
