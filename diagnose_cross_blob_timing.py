"""Diagnostic: test cross-blob timing and frequency relationships without truth leakage.

Usage: python -u diagnose_cross_blob_timing.py turing_data/config_101.h5 --window 0 --blobs 9 11 12
Uses frozen AOA/RF grouping. Does not merge blobs or assign emitter identities.
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


def circular_residual(t, period, phase):
    return (t-phase+period/2)%period-period/2


def best_phase(t, period, tol):
    phases=np.mod(t,period)
    s=np.sort(phases)
    ext=np.r_[s,s+period]
    right=np.searchsorted(ext,ext+2*tol,side="right")
    start=int(np.argmax(right[:len(s)]-np.arange(len(s))))
    center=float((ext[start]+tol)%period)
    r=circular_residual(t,period,center)
    return center,int(np.count_nonzero(np.abs(r)<=tol))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window",type=int,default=0)
    p.add_argument("--blobs",type=int,nargs="+",default=[9,11,12])
    p.add_argument("--period",type=float,default=85.1)
    p.add_argument("--tolerance",type=float,default=2.)
    p.add_argument("--out",default="cross_blob_timing")
    a=p.parse_args()
    if a.period<=0 or a.tolerance<=0 or a.tolerance>=a.period/4:
        p.error("Invalid period or tolerance")
    with h5py.File(a.file) as h:
        start=a.window*10000
        end=min(start+10000,len(h["data"]))
        if start<0 or end<=start: p.error("Window out of bounds")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:end],float)
        truth=np.asarray(h["labels"][start:end]).reshape(-1)
    toa=x[:,names.index("ToA")]
    rf=x[:,names.index("Frequency")]
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    blob=group_window(x,names,pops,npop,8,.20,6.,8)
    present=[b for b in a.blobs if np.any(blob==b)]
    if not present: p.error("No requested blobs found")
    selected=np.flatnonzero(np.isin(blob,present))
    origin=float(toa[selected].min())
    relative=toa-origin
    print(f"Window {a.window}, period hypothesis {a.period:.4f} raw TOA units; "
          f"common origin {origin:.3f}")
    phases={}
    for b in present:
        idx=np.flatnonzero(blob==b)
        phase,n=best_phase(relative[idx],a.period,a.tolerance)
        phases[b]=phase
        print(f"Blob {b}: PDWs={len(idx)} RF=[{rf[idx].min():.2f},{rf[idx].max():.2f}] "
              f"phase={phase:.3f} aligned={n}/{len(idx)} "
              f"truth_eval={Counter(truth[idx].tolist())}")
    print("Pairwise phase offsets (wrapped, raw TOA units):")
    for i,b in enumerate(present):
        for c in present[i+1:]:
            delta=float(circular_residual(np.array([phases[c]]),a.period,phases[b])[0])
            print(f"  {b} -> {c}: {delta:+.3f}")
    # Compare two models: one common phase vs independently fitted phases.
    common_phase,common_n=best_phase(relative[selected],a.period,a.tolerance)
    independent_n=sum(best_phase(relative[blob==b],a.period,a.tolerance)[1] for b in present)
    print(f"Common-phase fit: {common_n}/{len(selected)}; "
          f"independent-phase fits: {independent_n}/{len(selected)}")
    print("Independent phases can be consistent with distinct emitters, mode resets, "
          "or structured frequency timing. This diagnostic does not distinguish them.")
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5))
    for b in present:
        idx=np.flatnonzero(blob==b)
        ax.scatter(relative[idx],np.mod(relative[idx],a.period),s=13,label=f"Blob {b}")
    ax.set(xlabel="TOA minus common origin (raw units)",
           ylabel=f"TOA modulo {a.period:.4f} (raw units)",
           title="Cross-blob timing phase; no truth used")
    ax.legend()
    fig.tight_layout()
    dest=out/f"window_{a.window}_blobs_{'_'.join(map(str,present))}.png"
    fig.savefig(dest,dpi=160)
    plt.close(fig)
    print(f"Plot: {dest}")
    print("Note: a 170.2-unit fit may be a harmonic of 85.1; this plot alone "
          "cannot establish a hop schedule or distinguish simultaneous streams.")


if __name__=="__main__":
    main()
