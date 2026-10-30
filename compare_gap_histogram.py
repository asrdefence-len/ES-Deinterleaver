"""Successive-TOA-gap histogram: dominant peak mean as provisional PRI.

Run: python -u compare_gap_histogram.py turing_data/config_101.h5 --window 0 --emitter 46

Uses truth labels ONLY to select blobs for reporting, not to calculate gaps.
A peak of observed gaps need not be the fundamental PRI when pulses are
missing or multiple trains overlap.
"""
import argparse
import csv
import h5py
import numpy as np
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


def analyse(toa, bin_width=2.0, min_gap=10.0, max_gap=400.0):
    t=np.sort(np.asarray(toa, dtype=float))
    gaps=np.diff(t)
    gaps=gaps[(gaps>=min_gap)&(gaps<=max_gap)]
    if len(gaps)<3:
        return None, gaps, None, None
    edges=np.arange(min_gap,max_gap+bin_width*1.0001,bin_width)
    counts,_=np.histogram(gaps,bins=edges)
    # Modal bin plus immediate neighbours: estimate peak mean and spread.
    # This is a deliberately simple fixed-bandwidth peak measurement.
    peak=int(np.argmax(counts))
    low=edges[max(0,peak-1)]
    high=edges[min(len(edges)-1,peak+2)]
    in_peak=gaps[(gaps>=low)&(gaps<high)]
    if not len(in_peak): return None,gaps,counts,edges
    mean=float(np.mean(in_peak))
    sd=float(np.std(in_peak,ddof=1)) if len(in_peak)>1 else float("nan")
    return dict(pri=mean,jitter_sd=sd,jitter_pct=100*sd/mean,
                peak_n=len(in_peak),gap_n=len(gaps),
                peak_lo=float(low),peak_hi=float(high),
                modal_bin_n=int(counts[peak])),gaps,counts,edges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window",type=int,default=0)
    p.add_argument("--emitter",type=int,default=46)
    p.add_argument("--bin-width",type=float,default=2.0)
    p.add_argument("--min-gap",type=float,default=10.0)
    p.add_argument("--max-gap",type=float,default=400.0)
    p.add_argument("--out",default="gap_histogram_comparison.csv")
    p.add_argument("--plot",action="store_true")
    a=p.parse_args()
    if a.bin_width<=0 or a.min_gap<0 or a.max_gap<=a.min_gap:
        p.error("Invalid histogram range or bin width")
    with h5py.File(a.file) as h:
        lo=a.window*10000
        hi=min(lo+10000,len(h["data"]))
        if lo<0 or lo>=hi: p.error("Window outside dataset")
        names=[n.decode() if isinstance(n,bytes) else str(n)
               for n in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][lo:hi],float)
        truth=np.asarray(h["labels"][lo:hi]).reshape(-1)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    blobs=group_window(x,names,pops,npop,8,.20,6.,8)
    toa=x[:,names.index("ToA")]
    rf=x[:,names.index("Frequency")]
    chosen=np.flatnonzero(truth==a.emitter)
    if not len(chosen): p.error("Emitter not found in window")
    ids=sorted(set(int(v) for v in blobs[chosen] if v>=0))
    print(f"Emitter={a.emitter} window={a.window} blobs={len(ids)}")
    print("blob RF_median PDWs valid_gaps peak_n peak_range     mean_PRI   gap_SD   SD_pct")
    rows=[]
    for bid in ids:
        ix=np.flatnonzero(blobs==bid)
        result,gaps,counts,edges=analyse(toa[ix],a.bin_width,a.min_gap,a.max_gap)
        if result is None:
            print(f"{bid:4d} {np.median(rf[ix]):9.2f} {len(ix):4d} NO_PEAK")
            continue
        row=dict(blob=bid,rf_median=float(np.median(rf[ix])),pdws=len(ix),**result)
        rows.append(row)
        print(f"{bid:4d} {row['rf_median']:9.2f} {len(ix):4d} "
              f"{row['gap_n']:10d} {row['peak_n']:6d} "
              f"{row['peak_lo']:5.1f}-{row['peak_hi']:<5.1f} "
              f"{row['pri']:10.4f} {row['jitter_sd']:8.4f} "
              f"{row['jitter_pct']:8.3f}")
        if a.plot:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(9,4))
            ax.hist(gaps,bins=edges,edgecolor="none")
            ax.axvspan(row["peak_lo"],row["peak_hi"],alpha=.18,label="Selected peak")
            ax.axvline(row["pri"],linestyle="--",label=f"Peak mean {row['pri']:.3f}")
            ax.set(xlabel="Successive observed TOA gap (raw units)",
                   ylabel="Count",title=f"Window {a.window}, RF blob {bid}")
            ax.legend()
            fig.tight_layout()
            path=f"gap_hist_w{a.window}_b{bid}.png"
            fig.savefig(path,dpi=150)
            plt.close(fig)
            print("  Plot:",path)
    if rows:
        with open(a.out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("CSV:",a.out)
    print("CAUTION: mean_PRI is the mean of the dominant observed-gap peak, "
          "not necessarily the fundamental PRI. SD describes ONLY that "
          "selected peak, and is selection-biased. Missing pulses can "
          "produce multiples; interleaved emitters can create other peaks. "
          "No truth labels are used in estimation.")


if __name__=="__main__":
    main()
