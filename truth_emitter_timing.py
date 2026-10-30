"""Truth-labelled timing characterisation (evaluation only; never feed truth to deinterleaver).

Usage:
 python -u truth_emitter_timing.py turing_data/config_101.h5 --window 0 --emitter 46
Optional: --period 85.1 --out truth_timing
Plots observed adjacent gaps, gap multiples, RF/time, and timing residuals.
The data may omit transmitted pulses: this is NOT transmitted-PRI ground truth.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def robust_stats(a):
    if not len(a): return "none"
    return (f"n={len(a)} median={np.median(a):.4f} "
            f"mean={np.mean(a):.4f} std={np.std(a,ddof=1) if len(a)>1 else 0:.4f} "
            f"p05={np.percentile(a,5):.4f} p95={np.percentile(a,95):.4f}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window",type=int,default=0)
    ap.add_argument("--emitter",type=int,default=46)
    ap.add_argument("--period",type=float,default=85.1,
                    help="Candidate period from V0, NOT inferred from truth")
    ap.add_argument("--gap-tolerance",type=float,default=.12,
                    help="Maximum relative deviation from nearest period multiple")
    ap.add_argument("--out",default="truth_timing")
    a=ap.parse_args()
    if a.period<=0 or not 0<a.gap_tolerance<.5: ap.error("Invalid timing parameters")
    with h5py.File(a.file) as h:
        start=a.window*10000
        end=min(start+10000,len(h["data"]))
        if start<0 or start>=end: ap.error("Window outside dataset")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:end],float)
        labels=np.asarray(h["labels"][start:end]).reshape(-1)
    t=x[labels==a.emitter,names.index("ToA")]
    rf=x[labels==a.emitter,names.index("Frequency")]
    pw=x[labels==a.emitter,names.index("PulseWidth")]
    if len(t)<3: ap.error("Fewer than three labelled pulses for this emitter")
    order=np.argsort(t,kind="stable")
    t,rf,pw=t[order],rf[order],pw[order]
    d=np.diff(t)
    positive=d>0
    k=np.maximum(1,np.rint(d/a.period)).astype(int)
    residual=d-k*a.period
    close=positive & (np.abs(residual)<=a.gap_tolerance*a.period)
    single=close & (k==1)
    # Critical: these are gap statistics, not verified transmitted PRI jitter.
    print(f"Truth-labelled emitter={a.emitter} window={a.window} "
          f"observed_PDW={len(t)} RF_range=[{rf.min():.3f},{rf.max():.3f}] "
          f"TOA_span={np.ptp(t):.3f}")
    print(f"All positive adjacent observed gaps: {robust_stats(d[positive])}")
    print(f"Candidate period={a.period:.4f}; positive gaps={np.count_nonzero(positive)} "
          f"near integer multiples={np.count_nonzero(close)} "
          f"({100*np.mean(close):.1f}% of all gaps); "
          f"candidate single-period gaps={np.count_nonzero(single)}")
    print(f"Candidate single-period gaps: {robust_stats(d[single])}")
    if np.count_nonzero(single)>2:
        cv=100*np.std(d[single],ddof=1)/np.mean(d[single])
        print(f"Observed single-gap CV={cv:.3f}% (NOT validated transmitter jitter)")
    print(f"Residuals from k * candidate period, matching gaps: {robust_stats(residual[close])}")
    unique,counts=np.unique(k[close],return_counts=True)
    print("Inferred period-multiple gap counts (not confirmed missed-pulse counts):")
    print(" ".join(f"k={int(u)}:{int(c)}" for u,c in zip(unique[:30],counts[:30])))
    if len(unique)>30: print(f"... {len(unique)-30} additional k values")
    print(f"Non-multiple gaps={np.count_nonzero(positive & ~close)}; "
          f"nonpositive/duplicate TOA gaps={np.count_nonzero(~positive)}")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    prefix=out/f"emitter_{a.emitter}_window_{a.window}"
    fig,ax=plt.subplots(figsize=(11,4))
    ax.scatter(t-t[0],rf,s=8)
    ax.set(xlabel="TOA since first observed pulse (raw units)",
           ylabel="RF (raw dataset units)",title=f"Truth-labelled emitter {a.emitter}: RF vs TOA")
    fig.tight_layout();fig.savefig(str(prefix)+"_rf_time.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.scatter(t[1:]-t[0],d,s=8)
    ax.axhline(a.period,color="gray",linestyle="--",label="candidate period")
    ax.set(xlabel="TOA since first observed pulse (raw units)",
           ylabel="Adjacent observed TOA gap (raw units)",
           title="Observed gaps; may span missing pulses or multiple emission streams")
    ax.legend();fig.tight_layout()
    fig.savefig(str(prefix)+"_gaps_time.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.hist(d[positive],bins=min(150,max(15,int(np.sqrt(np.count_nonzero(positive))*3))))
    ax.set(xlabel="Adjacent observed TOA gap (raw units)",ylabel="Count",
           title="Truth-labelled adjacent observed gaps")
    fig.tight_layout();fig.savefig(str(prefix)+"_gap_hist.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.scatter(k[close],residual[close],s=12,alpha=.65)
    ax.axhline(0,color="gray",linestyle="--")
    ax.set(xlabel="Nearest candidate-period multiple k",
           ylabel="Gap residual (raw units)",
           title="Residual vs skipped-period hypothesis; not a jitter estimate")
    fig.tight_layout();fig.savefig(str(prefix)+"_gap_residuals.png",dpi=160);plt.close(fig)
    print(f"Plots saved to {prefix}_*.png")
    print("CAUTION: Labels identify received PDWs by emitter; they do not prove "
          "every transmitted pulse was observed, nor that all pulses form one stream. "
          "Stagger, modes, frequency hopping, TOA noise and missing pulses remain "
          "possible. Truth is used ONLY in this evaluation script.")


if __name__=="__main__":
    main()
