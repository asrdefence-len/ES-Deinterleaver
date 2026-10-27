"""Fast, bounded timing experiment with unchanged Approach C grouping.

Compare original extractor against vectorized phase-lattice proposals and
bounded greedy chains. Truth labels are used ONLY for post-hoc evaluation.

python turing_timing_fast.py turing_data/config_100.h5
"""
import argparse
import json
import time
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import robust_scale, groups, extract, evaluate, periods


def phase_chain(times, period, tolerance):
    """Choose a dense phase band, then require monotonic distinct pulse slots."""
    if len(times)<2 or period<=0:
        return np.empty(0,dtype=int)
    phase=np.mod(times-times[0],period)
    # Circular histogram: each pulse votes for a phase neighbourhood.
    bins=64
    h,_=np.histogram(phase,bins=bins,range=(0,period))
    smooth=np.roll(h,1)+h+np.roll(h,-1)
    centre=(int(np.argmax(smooth))+.5)*period/bins
    residual=np.mod(phase-centre+period/2,period)-period/2
    tol=period*tolerance
    eligible=np.flatnonzero(np.abs(residual)<=tol)
    if len(eligible)<2:
        return np.empty(0,dtype=int)
    # A pulse train cannot have two detections assigned to the same PRI slot.
    slots=np.rint((times[eligible]-times[0]-centre)/period).astype(np.int64)
    order=np.lexsort((np.abs(residual[eligible]),slots))
    best_for_slot={}
    for i in order:
        s=int(slots[i])
        if s not in best_for_slot:
            best_for_slot[s]=int(eligible[i])
    chosen=np.asarray(sorted(best_for_slot.values()),dtype=int)
    return chosen


def fast_extract(toa,labels,min_train=8,max_trains=20,tolerance=.05):
    assigned=np.full(len(toa),-1,dtype=int)
    trains=[]
    for group in sorted(set(labels)-{-1}):
        ids=np.flatnonzero(labels==group)
        ids=ids[np.argsort(toa[ids])]
        # Compute interval candidates ONCE per group, not every iteration.
        candidates=periods(toa[ids],max_candidates=12)
        for _ in range(max_trains):
            if len(ids)<min_train or not candidates:
                break
            times=toa[ids]
            best=np.empty(0,dtype=int)
            best_period=None
            for period in candidates:
                candidate=phase_chain(times,period,tolerance)
                if len(candidate)>len(best):
                    best,best_period=candidate,period
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


def metrics(truth,assigned,trains,seconds):
    ev=evaluate(truth,assigned,trains)
    return {"seconds":round(seconds,3),"trains":len(trains),
            "assigned":int((assigned>=0).sum()),
            "coverage":round(float(np.mean(assigned>=0)),5),
            "weighted_purity":ev["weighted_purity"] if ev else None,
            "cross_emitter_pdws":ev["cross_emitter_assigned_pdws"] if ev else None,
            "emitters_represented":ev["truth_emitters_represented"] if ev else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--output",default="turing_timing_fast.json")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    toa=data[:,names.index("ToA")]
    labels=groups(data,names,"C",2.0,8,
                  robust_scale(data[:,names.index("Frequency")]),
                  robust_scale(data[:,names.index("PulseWidth")]))
    print("Grouping complete; running baseline...",flush=True)
    start=time.perf_counter()
    old_a,old_t=extract(toa,labels,8,20,.05)
    baseline=metrics(truth,old_a,old_t,time.perf_counter()-start)
    print("Baseline:",json.dumps(baseline),flush=True)
    print("Running vectorized phase experiment...",flush=True)
    start=time.perf_counter()
    new_a,new_t=fast_extract(toa,labels)
    experimental=metrics(truth,new_a,new_t,time.perf_counter()-start)
    print("Experimental:",json.dumps(experimental),flush=True)
    result={"grouping":"Approach C unchanged","baseline":baseline,
            "experimental":experimental,
            "caveats":["Phase alignment does not prove a physical pulse train.",
                       "The experimental method may select chance-coincident pulses.",
                       "Constant-interval hypotheses only; not agile-PRI extraction.",
                       "Truth labels used for evaluation only."]}
    if truth is not None:
        result["per_emitter"]=[]
        for label in np.unique(truth):
            m=truth==label
            result["per_emitter"].append({"emitter":str(label),"total":int(m.sum()),
                "baseline":int((m&(old_a>=0)).sum()),
                "experimental":int((m&(new_a>=0)).sum())})
        result["per_emitter"].sort(key=lambda r:-r["total"])
        print("Top 15 emitters:",flush=True)
        for row in result["per_emitter"][:15]:
            print(row,flush=True)
    with open(a.output,"w") as f:
        json.dump(result,f,indent=2)
    print("Saved",a.output,flush=True)


if __name__=="__main__":
    main()
