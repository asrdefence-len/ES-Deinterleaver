"""Fold observed RF-blob TOAs at a candidate PRI; measure circular coherence.

Run:
 python -u compare_phase_folding_jitter.py turing_data/config_101.h5 --window 0 --blob 9

Truth is not used for estimation. Circular phase dispersion is NOT PRI jitter.
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--window",type=int,default=0)
    p.add_argument("--blob",type=int,default=9)
    p.add_argument("--period",type=float,default=None)
    p.add_argument("--bins",type=int,default=40)
    a=p.parse_args()
    with h5py.File(a.dataset) as h:
        start=a.window*10000
        end=min(start+10000,len(h["data"]))
        if start<0 or end<=start: p.error("Window out of range")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:end],float)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    groups=group_window(x,names,pops,npop,8,.20,6.,8)
    t=np.sort(x[groups==a.blob,names.index("ToA")])
    if len(t)<3: p.error("Blob has fewer than three PDWs")
    if a.period is None:
        tracks,_=discover(t,20.,200.,2.,20,4,.1)
        if not tracks: p.error("No V0 candidate period; supply --period")
        period=float(tracks[0].period)
    else:
        period=a.period
    if period<=0 or a.bins<4: p.error("Invalid period or bins")
    # Subtract an origin before modulo to reduce floating point precision loss.
    phase=np.mod(t-t[0],period)
    theta=2*np.pi*phase/period
    c=np.mean(np.cos(theta))
    s=np.mean(np.sin(theta))
    R=float(np.hypot(c,s))
    circular_sd=period/(2*np.pi)*np.sqrt(max(0.,-2*np.log(max(R,1e-15))))
    dispersion_pct=100*circular_sd/period
    centre=(np.arctan2(s,c)%(2*np.pi))*period/(2*np.pi)
    centered=(phase-centre+period/2)%period-period/2
    print(f"Window={a.window} blob={a.blob} PDWs={len(t)} "
          f"PRI={period:.4f} raw_TOA_units")
    print(f"Circular coherence R={R:.6f}")
    print(f"Circular phase SD={circular_sd:.4f} raw_TOA_units")
    print(f"Phase dispersion={dispersion_pct:.3f}% of PRI (NOT PRI jitter)")
    print(f"Folded phase centre relative to first TOA={centre:.4f}")
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    ax.hist(centered,bins=np.linspace(-period/2,period/2,a.bins+1),
            color="steelblue",edgecolor="white")
    ax.set(xlabel="Folded timing residual about circular mean (raw TOA units)",
           ylabel="PDW count",
           title=f"Phase fold: window {a.window}, blob {a.blob}, "
                 f"PRI {period:.3f}; R={R:.3f}; dispersion={dispersion_pct:.2f}%")
    out=f"phase_fold_w{a.window}_b{a.blob}.png"
    fig.savefig(out,dpi=170)
    plt.close(fig)
    print(f"Plot: {out}")
    print("Caution: folding naturally tolerates missing pulses but can mix "
          "multiple timing populations. Phase spread depends on PRI accuracy, "
          "observation duration, drift, and jitter mechanism. It is not an "
          "intrinsic transmitter PRI jitter estimate.")

if __name__=="__main__":
    main()
