"""Compare population, fixed and fractional RF scaling for candidate PDW groups.

Run:
  python turing_fractional_rf_comparison.py turing_data/config_100.h5 turing_data/testconfig_100.h5

Ground truth is used for scoring only. Groups are window-local, not emitter tracks.
"""
import argparse
import csv
from collections import Counter
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from turing_grouping_comparison import robust_scale


METHODS=("Population adaptive","Fixed 1000","Fractional 10%","Fractional 8%","Fractional 5%")


def grouping(x,names,method):
    rf=x[:,names.index("Frequency")]
    pw=x[:,names.index("PulseWidth")]
    angle=np.deg2rad(x[:,names.index("AoA")])
    pw_scale=robust_scale(pw)
    alpha=.05
    angular=np.column_stack((np.sin(angle)/np.deg2rad(2.),
                             np.cos(angle)/np.deg2rad(2.)))
    if method=="Population adaptive":
        rf_z=(rf-np.median(rf))/robust_scale(rf)
    elif method=="Fixed 1000":
        rf_z=(rf-np.median(rf))/1000.
    else:
        frac={"Fractional 10%":.10,"Fractional 8%":.08,"Fractional 5%":.05}[method]
        if np.any(rf<=0):
            raise ValueError("Fractional RF scaling requires positive frequency")
        rf_z=np.log(rf)/frac
    pw_z=np.log1p(alpha*pw/pw_scale)/alpha
    z=np.column_stack((angular,rf_z,pw_z))
    return DBSCAN(eps=1.,min_samples=8).fit_predict(z)


def metrics(pred,truth,rf):
    valid=pred>=0
    grouped=int(valid.sum())
    correct=0
    groups=0
    emitter_groups=Counter()
    wrong=np.zeros(len(pred),dtype=bool)
    for gid in np.unique(pred[valid]):
        indices=np.flatnonzero(pred==gid)
        counts=Counter(truth[indices].tolist())
        lead,n=counts.most_common(1)[0]
        correct+=n
        wrong[indices[truth[indices]!=lead]]=True
        groups+=1
        emitter_groups[lead]+=1
    return dict(groups=groups,grouped=grouped,correct=correct,
                cross=grouped-correct,noise=len(pred)-grouped,
                purity_pct=100*correct/grouped if grouped else 0.,
                dominant_emitters=len(emitter_groups),
                excess_dominant_groups=sum(max(n-1,0) for n in emitter_groups.values()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files",nargs="+")
    ap.add_argument("--window-size",type=int,default=10000)
    ap.add_argument("--output",default="turing_fractional_rf_results.csv")
    ap.add_argument("--focus-window",type=int,default=26)
    args=ap.parse_args()
    if args.window_size<8:ap.error("window-size must be at least 8")
    rows=[]
    for path in args.files:
        with h5py.File(path,"r") as f:
            names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
            if "labels" not in f:raise ValueError(f"Missing truth labels: {path}")
            n=len(f["data"])
            print(f"\n{path}: {n} PDWs")
            for start in range(0,n,args.window_size):
                stop=min(start+args.window_size,n)
                if stop-start<8:continue
                x=np.asarray(f["data"][start:stop],dtype=float)
                truth=np.asarray(f["labels"][start:stop]).reshape(-1)
                if not np.all(np.isfinite(x)):raise ValueError(f"Non-finite values at {path}:{start}")
                for method in METHODS:
                    pred=grouping(x,names,method)
                    m=metrics(pred,truth,x[:,names.index("Frequency")])
                    row=dict(file=path,start=start,stop=stop,method=method,**m)
                    rows.append(row)
                    if start==args.focus_window*args.window_size:
                        pairs=[]
                        for gid in np.unique(pred[pred>=0]):
                            members=truth[pred==gid]
                            if np.any(members==46) and np.any(members==80):
                                pairs.append((int(gid),int(np.sum(members==46)),int(np.sum(members==80))))
                        print(f"  FOCUS {start}:{stop} {method:20s} grouped={m['grouped']:5d} cross={m['cross']:4d} purity={m['purity_pct']:7.3f}% groups={m['groups']:3d} 46+80 groups={pairs}")
    with open(args.output,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print("\nFULL DATASET RESULTS")
    for path in args.files:
        for method in METHODS:
            subset=[r for r in rows if r["file"]==path and r["method"]==method]
            grouped=sum(r["grouped"] for r in subset)
            correct=sum(r["correct"] for r in subset)
            cross=sum(r["cross"] for r in subset)
            noise=sum(r["noise"] for r in subset)
            groups=sum(r["groups"] for r in subset)
            print(f"{path} | {method:20s} grouped={grouped:7d} correct={correct:7d} cross={cross:6d} noise={noise:5d} groups={groups:5d} purity={100*correct/grouped:.3f}%")
    print(f"\nSaved {args.output}")
    print("Frequency percentages are distance scales, NOT hard bandwidth limits.")
    print("Group counts and dominant-emitter counts are window-local. Labels never influence clustering.")


if __name__=="__main__":
    main()
