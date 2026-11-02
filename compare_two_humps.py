"""Explore one/two dominant neighbouring gap humps, excluding distant gaps.

Run: python -u compare_two_humps.py turing_data/config_101.h5 --window 0 --emitter 46 --plot

Peak selection uses only observed TOAs. Emitter truth selects reporting blobs only.
This is a diagnostic, not a validated multi-emitter pulse association method.
"""
import argparse
import csv
import h5py
import numpy as np
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


from testbed.pri_histogram import measure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window",type=int,default=0)
    p.add_argument("--emitter",type=int,default=46)
    p.add_argument("--bin-width",type=float,default=2.)
    p.add_argument("--neighbourhood",type=float,default=.15,
                   help="Second hump within this fraction of first hump (default 0.15)")
    p.add_argument("--min-gap",type=float,default=10.)
    p.add_argument("--max-gap",type=float,default=400.)
    p.add_argument("--plot",action="store_true")
    p.add_argument("--out",default="two_humps_comparison.csv")
    a=p.parse_args()
    if a.bin_width<=0 or a.min_gap<0 or a.max_gap<=a.min_gap or not 0<a.neighbourhood<1:
        p.error("Invalid bin width, range or neighbourhood")
    with h5py.File(a.file) as h:
        start=a.window*10000
        end=min(start+10000,len(h["data"]))
        if start<0 or end<=start:p.error("Window outside dataset")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:end],float)
        truth=np.asarray(h["labels"][start:end]).reshape(-1)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    blobs=group_window(x,names,pops,npop,8,.20,6.,8)
    indices=np.flatnonzero(truth==a.emitter)
    if not len(indices):p.error("Emitter absent in window")
    ids=sorted({int(v) for v in blobs[indices] if v>=0})
    toa=x[:,names.index("ToA")]
    rf=x[:,names.index("Frequency")]
    print(f"Emitter={a.emitter} window={a.window} blobs={len(ids)}")
    print("blob RF_median PDWs gaps used hump1 hump2 mean_PRI SD pct")
    rows=[]
    for b in ids:
        ix=np.flatnonzero(blobs==b)
        result,gaps,counts,edges,selected=measure(
            toa[ix],a.bin_width,a.min_gap,a.max_gap,a.neighbourhood)
        if result is None:
            print(f"{b:4d} {np.median(rf[ix]):9.2f} {len(ix):4d} NO_ESTIMATE")
            continue
        row=dict(blob=b,rf_median=float(np.median(rf[ix])),pdws=len(ix),**result)
        rows.append(row)
        hump2=f"{result['peak2']:.1f}" if result["n_humps"]==2 else "-"
        print(f"{b:4d} {row['rf_median']:9.2f} {len(ix):4d} "
              f"{result['n_gaps']:4d} {result['n_used']:4d} "
              f"{result['peak1']:5.1f} {hump2:>5} "
              f"{result['pri']:8.3f} {result['sd']:6.3f} {result['pct']:5.2f}%")
        if a.plot:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(9,4))
            ax.bar(edges[:-1],counts,width=a.bin_width,align="edge",
                   label="All observed gaps",alpha=.45)
            ax.bar(edges[:-1][selected],counts[selected],width=a.bin_width,
                   align="edge",label="Selected hump bins",alpha=.85)
            ax.axvline(result["pri"],linestyle="--",
                       label=f"Mean {result['pri']:.2f}")
            ax.set(xlabel="Successive observed TOA gap (raw units)",
                   ylabel="Count",title=f"Window {a.window}, blob {b}")
            ax.legend()
            fig.tight_layout()
            filename=f"two_humps_w{a.window}_b{b}.png"
            fig.savefig(filename,dpi=150)
            plt.close(fig)
            print("  Plot:",filename)
    if rows:
        with open(a.out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("CSV:",a.out)
    print("Diagnostic: one or two local histogram humps, within 15% of "
          "dominant peak; each uses contiguous >=half-height bins. "
          "Distant missed-pulse multiples are discarded. The selected "
          "width is still truncated, so SD is not an unbiased jitter "
          "estimate. Inspect plots before using in a streaming tracker.")


if __name__=="__main__":
    main()
