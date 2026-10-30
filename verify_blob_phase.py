"""Experimental fixed-PRI phase-coherence verification in frozen RF blobs.

Does NOT identify emitters or claim a PRI from histogram peaks alone.
Truth labels are used only for post-hoc reporting.
"""
import argparse
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


def fit_phase(t, period, tolerance):
    phase = np.mod(t, period)
    angles = 2 * np.pi * phase / period
    # Search phase centers from observed phases; efficient for small blobs.
    best = (np.array([], dtype=int), 0.0)
    for center in phase:
        residual = (phase - center + period / 2) % period - period / 2
        idx = np.flatnonzero(np.abs(residual) <= tolerance)
        if len(idx) > len(best[0]):
            best = (idx, float(center))
    idx, center = best
    residual = (phase[idx] - center + period / 2) % period - period / 2
    return idx, center, float(np.sqrt(np.mean(residual**2))) if len(idx) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--blob", type=int, default=11)
    p.add_argument("--period-min", type=float, default=82)
    p.add_argument("--period-max", type=float, default=89)
    p.add_argument("--period-step", type=float, default=.025)
    p.add_argument("--tolerance", type=float, default=2.0,
                   help="Absolute phase residual tolerance in raw TOA units")
    p.add_argument("--extend", action="store_true",
                   help="Evaluate fitted phase against other RF blobs in same AOA population")
    p.add_argument("--out", default="phase_verify")
    a = p.parse_args()
    if not 0 < a.period_min < a.period_max or a.period_step <= 0 or a.tolerance <= 0:
        p.error("Invalid search interval/tolerance")
    with h5py.File(a.file) as h:
        start = a.window * 10000
        stop = min(start + 10000, len(h["data"]))
        if stop <= start:
            p.error("Window outside dataset")
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in h["metadata/feature_names"][:]]
        x = np.asarray(h["data"][start:stop], float)
        truth = np.asarray(h["labels"][start:stop]).reshape(-1)
    toa = x[:, names.index("ToA")]
    rf = x[:, names.index("Frequency")]
    pops, npop = batch_aoa_populations(x[:, names.index("AoA")])
    blob = group_window(x, names, pops, npop, 8, .20, 6.0, 8)
    selected = np.flatnonzero(blob == a.blob)
    if len(selected) < 5:
        p.error("Selected blob has fewer than five PDWs")
    periods = np.arange(a.period_min, a.period_max + a.period_step/2, a.period_step)
    # Center TOAs to avoid large absolute phase numerical offsets.
    origin = float(toa[selected].min())
    t = toa[selected] - origin
    results = [fit_phase(t, float(T), a.tolerance) for T in periods]
    ranked = sorted(range(len(periods)), key=lambda j:
                    (-len(results[j][0]), results[j][2]))
    best = ranked[0]
    T = float(periods[best])
    local, phase, rms = results[best]
    matched = selected[local]
    print(f"Window {a.window} blob {a.blob}: {len(selected)} PDWs; "
          f"RF range {rf[selected].min():.3f} to {rf[selected].max():.3f}")
    print(f"Best candidate period {T:.4f} raw TOA units; "
          f"phase {phase:.3f}; matches {len(matched)}/{len(selected)}; "
          f"RMS residual {rms:.3f}")
    print("Top candidate periods (period, matched PDWs, RMS residual):")
    for j in ranked[:10]:
        print(f"  {periods[j]:.4f}, {len(results[j][0])}, {results[j][2]:.3f}")
    print(f"Evaluation-only truth for matched PDWs: {Counter(truth[matched].tolist())}")
    # Quantify an elementary uniform-phase chance expectation; not a p-value,
    # because we optimize period and phase on the same observations.
    fraction = min(1.0, 2*a.tolerance/T)
    print(f"Uniform-phase fixed-window expectation: ~{len(selected)*fraction:.1f} "
          f"matches (NOT a significance test; period/phase were searched).")
    if a.extend:
        pop = Counter(pops[selected]).most_common(1)[0][0]
        other = np.flatnonzero((pops == pop) & (blob != a.blob))
        residual = (np.mod(toa[other] - origin - phase + T/2, T) - T/2)
        extra = other[np.abs(residual) <= a.tolerance]
        print(f"Other blobs in same AOA population: {len(other)} PDWs; "
              f"{len(extra)} phase matches; evaluation truth "
              f"{Counter(truth[extra].tolist()).most_common(10)}")
        print("Matches across other blobs are candidate associations only, "
              "not validated extensions of the pulse train.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(periods, [len(r[0]) for r in results])
    ax.axvline(T, linestyle=":")
    ax.set(xlabel="Candidate PRI (raw TOA units)", ylabel="PDWs within phase gate",
           title=f"Window {a.window} blob {a.blob}: phase consistency scan")
    fig.tight_layout()
    path = out / f"window_{a.window}_blob_{a.blob}_phase_scan.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    resid = (np.mod(t-phase+T/2, T)-T/2)
    ax.scatter(toa[selected], resid, s=12)
    ax.axhline(a.tolerance, linestyle=":")
    ax.axhline(-a.tolerance, linestyle=":")
    ax.set(xlabel="TOA (raw units)", ylabel="Phase residual (raw units)",
           title=f"Blob {a.blob} phase residuals, candidate PRI {T:.4f}")
    fig.tight_layout()
    path2 = out / f"window_{a.window}_blob_{a.blob}_phase_residuals.png"
    fig.savefig(path2, dpi=140)
    plt.close(fig)
    print(f"Plots: {path}, {path2}")
    print("Caution: fixed-PRI phase fitting can also match harmonics, "
          "subharmonics and unrelated pulses. No sequence continuity test yet.")


if __name__ == "__main__":
    main()
