"""Truth-only diagnostic for emitter 30; no changes to deinterleaver.

python turing_emitter30_diagnostic.py turing_data/config_100.h5
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from turing_baseline import load_sample
from turing_grouping_comparison import robust_scale, groups
from turing_timing_fast import fast_extract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--emitter",type=int,default=30)
    p.add_argument("--prefix",default="turing_emitter30")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Truth labels required")
    mask=truth==a.emitter
    if not mask.any():
        p.error("Emitter not present in selected sample")
    toa=data[:,names.index("ToA")]
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    aoa=data[:,names.index("AoA")]
    grouping=groups(data,names,"C",2.0,8,
                    robust_scale(rf),robust_scale(pw))
    assigned,trains=fast_extract(toa,grouping)
    ids=np.flatnonzero(mask)
    ids=ids[np.argsort(toa[ids])]
    dt=np.diff(toa[ids])
    group_counts=Counter(int(x) for x in grouping[ids])
    group_rows=[]
    for g,n in group_counts.most_common():
        members=ids[grouping[ids]==g]
        whole=np.flatnonzero(grouping==g) if g>=0 else np.array([],dtype=int)
        group_rows.append({"group":g,"emitter_pdws":n,
                           "all_pdws_in_group":int(len(whole)),
                           "other_emitter_pdws":int(len(whole)-n),
                           "recovered_emitter_pdws":int(np.count_nonzero(assigned[members]>=0)),
                           "rf_min":float(rf[members].min()),
                           "rf_max":float(rf[members].max()),
                           "pw_min":float(pw[members].min()),
                           "pw_max":float(pw[members].max())})
    selected_trains=[]
    for i,t in enumerate(trains):
        member=np.asarray(t["pdw_indices"],dtype=int)
        count=int(np.count_nonzero(mask[member]))
        if count:
            selected_trains.append({"train":i,"emitter_pdws":count,
                                    "total_train_pdws":len(member),
                                    "candidate_interval":t["candidate_interval"]})
    out={"emitter":a.emitter,"total_pdws":len(ids),
         "recovered_pdws":int(np.count_nonzero(assigned[ids]>=0)),
         "toa_span":float(np.ptp(toa[ids])),
         "rf_range":[float(rf[ids].min()),float(rf[ids].max())],
         "pw_range":[float(pw[ids].min()),float(pw[ids].max())],
         "aoa_range":[float(aoa[ids].min()),float(aoa[ids].max())],
         "interarrival_percentiles_1_10_25_50_75_90_99":
            np.percentile(dt,[1,10,25,50,75,90,99]).tolist() if len(dt) else [],
         "groups":group_rows,"recovered_trains":selected_trains,
         "note":"Interarrival times are between observed same-emitter pulses; not necessarily transmitted PRI."}
    with open(a.prefix+".json","w") as f:
        json.dump(out,f,indent=2)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=False,constrained_layout=True)
    t=toa[ids]-toa[ids[0]]
    recovered=assigned[ids]>=0
    for ax,values,title in [(axes[0],rf[ids],"Frequency (MHz)"),
                            (axes[1],pw[ids],"Pulse width (microseconds)"),
                            (axes[2],aoa[ids],"AOA (degrees)")]:
        ax.scatter(t[~recovered],values[~recovered],s=12,label="Unassigned")
        ax.scatter(t[recovered],values[recovered],s=22,label="Recovered")
        ax.set_ylabel(title)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[3].scatter(t[1:],dt,s=12)
    axes[3].set_xlabel("Relative TOA (microseconds)")
    axes[3].set_ylabel("Observed interval (microseconds)")
    axes[3].grid(alpha=.2)
    fig.suptitle("Emitter %s: truth-labelled PDWs and Approach C recovery"%a.emitter)
    fig.savefig(a.prefix+".png",dpi=160)
    plt.close(fig)
    print(json.dumps(out,indent=2))
    print("Saved",a.prefix+".json and",a.prefix+".png")


if __name__=="__main__":
    main()
