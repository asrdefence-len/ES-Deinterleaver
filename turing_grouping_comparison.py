"""Compare three blind PDW grouping strategies followed by the SAME timing extractor.

A: AOA; B: AOA+RF; C: AOA+RF+PW.
Truth is used ONLY for post-extraction scoring. All time/frequency/width
thresholds are derived from observed measurements, not assumed units.

Usage:
 python turing_grouping_comparison.py turing_data/config_100.h5
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample


def robust_scale(x):
    q=np.percentile(x,[25,75])
    spread=float(q[1]-q[0])
    return max(spread/3.,1e-9)


def groups(data,names,mode,aoa_deg,min_samples,rf_scale,pw_scale):
    a=np.deg2rad(data[:,names.index("AoA")])
    z=[np.sin(a)/np.deg2rad(aoa_deg),np.cos(a)/np.deg2rad(aoa_deg)]
    if mode in ("B","C"):
        rf=data[:,names.index("Frequency")]
        z.append((rf-np.median(rf))/rf_scale)
    if mode=="C":
        pw=data[:,names.index("PulseWidth")]
        z.append((pw-np.median(pw))/pw_scale)
    return DBSCAN(eps=1.0,min_samples=min_samples).fit_predict(np.column_stack(z))


def periods(times, max_candidates=12):
    # Difference spectrum over nearby pulses. No fixed PRI range.
    if len(times)<3:
        return []
    d=np.concatenate([times[k:]-times[:-k] for k in range(1,min(10,len(times)-1)+1)])
    d=d[np.isfinite(d)&(d>0)]
    if len(d)<3:
        return []
    low,high=np.percentile(d,[1,99])
    if high<=low:
        return [float(np.median(d))]
    # Log bins capture sub-unit through large intervals without arbitrary cutoffs.
    edges=np.geomspace(max(low*.75,np.min(d)),high*1.25,161)
    hist,_=np.histogram(d,bins=edges)
    selected=np.argsort(hist)[::-1][:max_candidates]
    return [float(np.sqrt(edges[i]*edges[i+1])) for i in selected if hist[i]>0]


def chain(times,period,relative_tolerance,max_skip=6,max_seeds=36):
    # Seed only a bounded sample; searchsorted avoids O(n^2) greedy scans.
    n=len(times)
    seeds=np.unique(np.linspace(0,min(n-1,500),min(max_seeds,n)).astype(int))
    tol=max(period*relative_tolerance,0.02*np.median(np.diff(times)[np.diff(times)>0]))
    best=[]
    for seed in seeds:
        found=[int(seed)]
        last=float(times[seed])
        pos=seed+1
        while pos<n:
            next_best=None
            for skip in range(1,max_skip+1):
                expected=last+skip*period
                k=int(np.searchsorted(times,expected,side="left"))
                for j in (k-1,k):
                    if j>=pos and j<n:
                        error=abs(times[j]-expected)
                        if error<=tol and (next_best is None or error<next_best[0]):
                            next_best=(error,j)
            if next_best is None:
                break
            pos=next_best[1]+1
            last=float(times[next_best[1]])
            found.append(next_best[1])
        if len(found)>len(best):
            best=found
    return np.asarray(best,dtype=int)


def extract(toa,labels,min_train,max_trains,relative_tolerance):
    assigned=np.full(len(toa),-1,dtype=int)
    trains=[]
    for group in sorted(set(labels)-{-1}):
        ids=np.flatnonzero(labels==group)
        ids=ids[np.argsort(toa[ids])]
        for _ in range(max_trains):
            if len(ids)<min_train:
                break
            times=toa[ids]
            best=np.array([],dtype=int)
            best_pri=None
            for pri in periods(times):
                found=chain(times,pri,relative_tolerance)
                if len(found)>len(best):
                    best,best_pri=found,pri
            if len(best)<min_train:
                break
            chosen=ids[best]
            assigned[chosen]=len(trains)
            trains.append({"group":int(group),"pdw_indices":chosen.tolist(),
                           "pulse_count":len(chosen),"candidate_interval":best_pri,
                           "time_span":float(toa[chosen[-1]]-toa[chosen[0]])})
            keep=np.ones(len(ids),dtype=bool)
            keep[best]=False
            ids=ids[keep]
    return assigned,trains


def evaluate(truth,assigned,trains):
    if truth is None:
        return None
    dominant=0
    contaminated=0
    label_to_trains={}
    for train in trains:
        idx=np.asarray(train["pdw_indices"],dtype=int)
        labels,counts=np.unique(truth[idx],return_counts=True)
        top=int(np.argmax(counts))
        dominant+=int(counts[top])
        contaminated+=int(len(idx)-counts[top])
        for label in labels:
            key=str(label)
            label_to_trains.setdefault(key,set()).add(len(label_to_trains.get(key,set()))) if False else None
    # Count distinct recovered train IDs containing each ground-truth emitter.
    label_to_trains={}
    for i,train in enumerate(trains):
        for label in np.unique(truth[np.asarray(train["pdw_indices"],dtype=int)]):
            label_to_trains.setdefault(str(label),set()).add(i)
    assigned_n=int(np.count_nonzero(assigned>=0))
    return {"weighted_purity":float(dominant/assigned_n) if assigned_n else None,
            "cross_emitter_assigned_pdws":contaminated,
            "truth_emitters_represented":len(label_to_trains),
            "trains_per_truth_emitter":{"median":float(np.median([len(v) for v in label_to_trains.values()])) if label_to_trains else None,
                                        "maximum":max((len(v) for v in label_to_trains.values()),default=0)},
            "note":"Truth labels identify emitters, NOT individual pulse trains; fragmentation is descriptive only."}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--limit",type=int,default=10000)
    ap.add_argument("--aoa-deg",type=float,default=2.0)
    ap.add_argument("--min-samples",type=int,default=8)
    ap.add_argument("--rf-scale",type=float,default=None,
                    help="Optional RF compatibility scale in native units; default robust data scale")
    ap.add_argument("--pw-scale",type=float,default=None,
                    help="Optional PW compatibility scale in native units; default robust data scale")
    ap.add_argument("--min-train",type=int,default=8)
    ap.add_argument("--max-trains-per-group",type=int,default=20)
    ap.add_argument("--relative-tolerance",type=float,default=.05)
    ap.add_argument("--output",default="turing_grouping_comparison.json")
    a=ap.parse_args()
    if a.aoa_deg<=0 or a.min_samples<2 or a.min_train<2 or a.relative_tolerance<=0:
        ap.error("Invalid grouping or extraction parameters")
    data,truth,names,total=load_sample(a.file,a.limit)
    toa=data[:,names.index("ToA")]
    rf_scale=a.rf_scale or robust_scale(data[:,names.index("Frequency")])
    pw_scale=a.pw_scale or robust_scale(data[:,names.index("PulseWidth")])
    results={}
    for mode,description in [("A","AOA"),("B","AOA + RF"),("C","AOA + RF + PW")]:
        labels=groups(data,names,mode,a.aoa_deg,a.min_samples,rf_scale,pw_scale)
        assigned,trains=extract(toa,labels,a.min_train,a.max_trains_per_group,a.relative_tolerance)
        results[mode]={"grouping":description,
                       "candidate_groups":len(set(labels)-{-1}),
                       "grouped_pdws":int(np.count_nonzero(labels>=0)),
                       "recovered_trains":len(trains),
                       "assigned_pdws":int(np.count_nonzero(assigned>=0)),
                       "coverage":float(np.mean(assigned>=0)),
                       "train_sizes":[t["pulse_count"] for t in trains],
                       "evaluation_only":evaluate(truth,assigned,trains)}
        print(description,":",results[mode]["recovered_trains"],"trains,",
              results[mode]["assigned_pdws"],"assigned PDWs")
    output={"file":a.file,"sample_pdws":len(data),
            "rf_scale_native_units":rf_scale,"pw_scale_native_units":pw_scale,
            "method":"Identical approximate constant-interval chain extractor for all three groupings",
            "caveats":["A common emitter can produce multiple valid trains.",
                       "The same emitter truth label is not a train label.",
                       "This is a controlled grouping ablation, not yet an agile-PRI deinterleaver.",
                       "Log-binned interval proposals and bounded greedy seeds can miss real trains.",
                       "A/B/C groups are alternative hypotheses; do not combine their assignments."],
            "results":results}
    with open(a.output,"w") as f:
        json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))
    print("Saved",a.output)


if __name__=="__main__":
    main()
