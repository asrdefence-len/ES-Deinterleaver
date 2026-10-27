"""Windowed PDW grouping stress test; truth used only after grouping.

Run: python turing_windowed_stress.py turing_data/config_100.h5 turing_data/testconfig_100.h5
"""
import argparse
import csv
from collections import Counter
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from turing_grouping_comparison import groups, robust_scale


def score(labels, truth):
    valid=labels>=0
    grouped=int(valid.sum())
    correct=0
    dominant=set()
    for g in np.unique(labels[valid]):
        counts=Counter(truth[labels==g].tolist())
        leader,n=counts.most_common(1)[0]
        correct+=n
        dominant.add(leader)
    return dict(groups=len(np.unique(labels[valid])),grouped=grouped,
                ungrouped=len(labels)-grouped,correct=correct,cross=grouped-correct,
                purity=correct/grouped if grouped else 0.,
                dominant_emitters=len(dominant),truth_emitters=len(np.unique(truth)))


def process(data,names,truth,aoa,min_samples):
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    sr=robust_scale(rf)
    sp=robust_scale(pw)
    base=groups(data,names,"C",aoa,min_samples,sr,sp)
    theta=np.deg2rad(data[:,names.index("AoA")])
    alpha=.05
    z=np.column_stack((np.sin(theta)/np.deg2rad(aoa),
                       np.cos(theta)/np.deg2rad(aoa),
                       (rf-np.median(rf))/sr,
                       np.log1p(alpha*pw/sp)/alpha))
    adaptive=DBSCAN(eps=1.,min_samples=min_samples).fit_predict(z)
    return sr,sp,score(base,truth),score(adaptive,truth)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("files",nargs="+")
    p.add_argument("--window",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--output",default="turing_windowed_stress.csv")
    args=p.parse_args()
    if args.window<args.min_samples or args.aoa_deg<=0:
        p.error("Window must be >= min-samples; aoa-deg must be positive")
    rows=[]
    for path in args.files:
        with h5py.File(path,"r") as f:
            raw=f["metadata/feature_names"][:]
            names=[v.decode() if isinstance(v,bytes) else str(v) for v in raw]
            if "labels" not in f:
                print(f"SKIP {path}: no ground truth labels")
                continue
            total=f["data"].shape[0]
            print(f"\n{path}: {total} PDWs, window={args.window}")
            for start in range(0,total,args.window):
                stop=min(start+args.window,total)
                if stop-start<args.min_samples:
                    print(f"  Skipping final {stop-start} PDWs (below min-samples)")
                    continue
                data=np.asarray(f["data"][start:stop],dtype=float)
                truth=np.asarray(f["labels"][start:stop]).reshape(-1)
                if not np.all(np.isfinite(data)):
                    print(f"  SKIP window {start}:{stop}: non-finite PDWs")
                    continue
                if np.any(data[:,names.index("PulseWidth")]<=0):
                    print(f"  SKIP window {start}:{stop}: non-positive PW")
                    continue
                sr,sp,c,a=process(data,names,truth,args.aoa_deg,args.min_samples)
                for method,m in (("C",c),("Adaptive 5%",a)):
                    rows.append(dict(file=path,start=start,stop=stop,window_pdws=stop-start,
                                     method=method,rf_scale=sr,pw_scale=sp,**m))
                print(f"  {start:7d}:{stop:<7d} C {c['purity']*100:6.2f}%/{c['grouped']:5d} grouped/{c['cross']:4d} cross | A {a['purity']*100:6.2f}%/{a['grouped']:5d} grouped/{a['cross']:4d} cross | RF scale {sr:.3g} PW scale {sp:.3g}")
    if not rows:
        p.error("No windows processed")
    with open(args.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nSUMMARY (window-level groups; emitter counts cannot be summed across windows)")
    for path in args.files:
        file_rows=[r for r in rows if r["file"]==path]
        if not file_rows:continue
        for method in ("C","Adaptive 5%"):
            rs=[r for r in file_rows if r["method"]==method]
            grouped=sum(r["grouped"] for r in rs)
            correct=sum(r["correct"] for r in rs)
            cross=sum(r["cross"] for r in rs)
            ungrouped=sum(r["ungrouped"] for r in rs)
            print(f"{path} | {method}: windows={len(rs)}, grouped={grouped}, correct={correct}, cross={cross}, ungrouped={ungrouped}, micro-purity={100*correct/grouped:.3f}%")
        print("  RF scales min/max:",min(r["rf_scale"] for r in file_rows),max(r["rf_scale"] for r in file_rows))
        print("  PW scales min/max:",min(r["pw_scale"] for r in file_rows),max(r["pw_scale"] for r in file_rows))
    print(f"Saved {args.output}")
    print("These are offline, non-overlapping PDW-count windows, not real-time operation.")
    print("Purity is within-window majority-label purity, not correct emitter identity across windows.")
    print("No labels are used to form groups.")


if __name__=="__main__":
    main()
