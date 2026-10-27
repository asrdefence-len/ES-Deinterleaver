"""Controlled timing-extractor comparison with Approach C grouping held fixed.

Baseline is the original extractor. Experimental extractor uses more global
seed coverage, wider skip search, and multiple candidate periods. Truth is
used only after extraction for evaluation.

python turing_timing_upgrade.py turing_data/config_100.h5
"""
import argparse
import json
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import robust_scale, groups, extract, evaluate, periods


def extended_chain(times, period, tolerance, max_skip=16, max_seeds=160, max_misses=3):
    """Greedy timing chain, with seeds across entire observation and gap recovery."""
    n=len(times)
    if n<2 or period<=0:
        return np.empty(0,dtype=int)
    seeds=np.unique(np.linspace(0,n-1,min(n,max_seeds)).astype(int))
    positive=np.diff(times)
    positive=positive[positive>0]
    tol=max(period*tolerance,0.02*np.median(positive) if len(positive) else 0)
    best=np.empty(0,dtype=int)
    for seed in seeds:
        found=[int(seed)]
        last=float(times[seed])
        pos=int(seed)+1
        misses=0
        while pos<n:
            match=None
            for skip in range(1,max_skip+1):
                target=last+skip*period
                k=int(np.searchsorted(times,target,side="left"))
                for j in (k-1,k):
                    if j>=pos and j<n:
                        error=abs(times[j]-target)
                        if error<=tol and (match is None or error<match[0]):
                            match=(error,j)
            if match is not None:
                j=match[1]
                found.append(j)
                last=float(times[j])
                pos=j+1
                misses=0
            else:
                misses+=1
                if misses>max_misses:
                    break
                last+=max_skip*period
                pos=max(pos,int(np.searchsorted(times,last-tol,side="left")))
        if len(found)>len(best):
            best=np.asarray(found,dtype=int)
    return best


def extract_extended(toa,labels,min_train=8,max_trains=60,tolerance=.05):
    assigned=np.full(len(toa),-1,dtype=int)
    trains=[]
    for group in sorted(set(labels)-{-1}):
        ids=np.flatnonzero(labels==group)
        ids=ids[np.argsort(toa[ids])]
        for _ in range(max_trains):
            if len(ids)<min_train:
                break
            times=toa[ids]
            best=np.empty(0,dtype=int)
            best_period=None
            for period in periods(times,max_candidates=24):
                found=extended_chain(times,period,tolerance)
                if len(found)>len(best):
                    best,best_period=found,period
            if len(best)<min_train:
                break
            chosen=ids[best]
            assigned[chosen]=len(trains)
            trains.append({"group":int(group),"pdw_indices":chosen.tolist(),
                           "pulse_count":len(chosen),"candidate_interval":best_period})
            keep=np.ones(len(ids),dtype=bool)
            keep[best]=False
            ids=ids[keep]
    return assigned,trains


def summary(truth,assigned,trains):
    ev=evaluate(truth,assigned,trains)
    represented=set()
    if truth is not None:
        for train in trains:
            represented.update(np.unique(truth[np.asarray(train["pdw_indices"],dtype=int)]).tolist())
    return {"trains":len(trains),"assigned_pdws":int((assigned>=0).sum()),
            "coverage":float(np.mean(assigned>=0)),
            "weighted_purity":ev["weighted_purity"] if ev else None,
            "cross_emitter_pdws":ev["cross_emitter_assigned_pdws"] if ev else None,
            "emitters_represented":len(represented) if truth is not None else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--output",default="turing_timing_upgrade.json")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    toa=data[:,names.index("ToA")]
    rf_scale=robust_scale(data[:,names.index("Frequency")])
    pw_scale=robust_scale(data[:,names.index("PulseWidth")])
    labels=groups(data,names,"C",2.0,8,rf_scale,pw_scale)
    old_assigned,old_trains=extract(toa,labels,8,20,.05)
    new_assigned,new_trains=extract_extended(toa,labels)
    output={"method":"Approach C grouping fixed; compare baseline vs expanded timing search",
            "baseline":summary(truth,old_assigned,old_trains),
            "experimental":summary(truth,new_assigned,new_trains),
            "caveats":["More candidate searches can also create chance-aligned false trains.",
                       "Constant-interval hypothesis remains; this is not an agile-PRI solution.",
                       "Ground truth is evaluation-only, never used for extraction."]}
    if truth is not None:
        per_emitter=[]
        for label in np.unique(truth):
            m=truth==label
            per_emitter.append({"emitter":str(label),"total":int(m.sum()),
                                "baseline":int((m&(old_assigned>=0)).sum()),
                                "experimental":int((m&(new_assigned>=0)).sum())})
        output["per_emitter"]=sorted(per_emitter,key=lambda x:-x["total"])
    print(json.dumps({k:v for k,v in output.items() if k!="per_emitter"},indent=2))
    print("Top 15 emitters by population:")
    for row in output.get("per_emitter",[])[:15]:
        print(row)
    with open(a.output,"w") as f:
        json.dump(output,f,indent=2)
    print("Saved",a.output)


if __name__=="__main__":
    main()
