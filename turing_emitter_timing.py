"""Inspect observed pulse-to-pulse timing without assuming a PRI range.

Truth labels are used ONLY for this diagnostic. No claim is made that
successive PDWs with one emitter label necessarily form one pulse train.

Usage:
 python turing_emitter_timing.py turing_data/config_100.h5 --emitter 80
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from turing_baseline import load_sample


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--emitter",default="80",help="Ground-truth label for diagnostic only")
    ap.add_argument("--limit",type=int,default=10000)
    ap.add_argument("--prefix",default="emitter_timing")
    ap.add_argument("--max-lag",type=int,default=50)
    args=ap.parse_args()
    data,truth,names,total=load_sample(args.file,args.limit)
    if truth is None:
        ap.error("This diagnostic requires ground-truth labels")
    if "ToA" not in names:
        ap.error("ToA feature is absent")
    mask=np.asarray([str(x)==args.emitter for x in truth])
    toa=np.sort(data[mask,names.index("ToA")])
    if len(toa)<3:
        ap.error("Emitter label absent or fewer than three pulses")
    dt=np.diff(toa)
    positive=dt[dt>0]
    if len(positive)==0:
        ap.error("No positive TOA differences")
    q=[0,1,5,10,25,50,75,90,95,99,100]
    quant={str(k):float(np.percentile(dt,k)) for k in q}
    # ACF of intervals by index, not a model of constant PRI.
    centered=dt-np.mean(dt)
    var=np.dot(centered,centered)
    acf=[]
    for lag in range(1,min(args.max_lag,len(dt)-1)+1):
        acf.append(float(np.dot(centered[:-lag],centered[lag:])/var) if var>0 else None)
    # Local timing distribution reveals changes in behaviour over the observation.
    windows=np.array_split(np.arange(len(dt)),min(10,len(dt)))
    local=[{"index_first":int(w[0]),"index_last":int(w[-1]),
            "median_interval":float(np.median(dt[w])),
            "p10_interval":float(np.percentile(dt[w],10)),
            "p90_interval":float(np.percentile(dt[w],90))}
           for w in windows if len(w)]
    summary={"file":args.file,"label":args.emitter,"pdws":int(len(toa)),
             "toa_span_native_units":float(toa[-1]-toa[0]),
             "observed_intervals_native_units":quant,
             "nonpositive_intervals":int(np.count_nonzero(dt<=0)),
             "interval_acf_lags_1_to_n":acf,
             "local_windows":local,
             "caveat":"Consecutive same-label pulses may span multiple modes or trains; intervals are not asserted to be PRI.",
             "units":"Native dataset TOA units; verify metadata for physical units."}
    fig,axes=plt.subplots(4,1,figsize=(13,14),constrained_layout=True)
    axes[0].plot(np.arange(len(dt)),dt,lw=.55)
    axes[0].set(xlabel="Pulse index",ylabel="Adjacent TOA difference (native units)",
                title=f"Emitter {args.emitter}: observed intervals (all, no PRI bounds)")
    axes[1].hist(positive,bins=120)
    axes[1].set(xlabel="Positive adjacent TOA difference (native units)",ylabel="Count",
                title="Full-range interval histogram")
    axes[2].plot(np.arange(len(dt)),dt,lw=.5)
    axes[2].set_yscale("log")
    axes[2].set(xlabel="Pulse index",ylabel="Adjacent TOA difference (log scale)",
                title="Log view of interval variation")
    axes[3].plot(np.arange(1,len(acf)+1),acf,marker="o",ms=3)
    axes[3].axhline(0,color="gray",lw=.8)
    axes[3].set(xlabel="Interval-sequence lag (pulses)",ylabel="Autocorrelation",
                title="Observed interval sequence autocorrelation")
    png=args.prefix+"_"+args.emitter+".png"
    out=args.prefix+"_"+args.emitter+".json"
    fig.savefig(png,dpi=150)
    plt.close(fig)
    with open(out,"w") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
    print("Saved",png,"and",out)


if __name__=="__main__":
    main()
