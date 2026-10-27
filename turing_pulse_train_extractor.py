"""Experimental AOA-gated constant-PRI pulse train extractor.

No truth labels, emitter IDs, or known train count enter extraction. This is
a deliberately simple first baseline, not a production deinterleaver.
It uses angular candidate groups, finds repeated TOA differences, grows
non-overlapping sequences, and leaves unclaimed pulses unassigned.

Usage:
 python turing_pulse_train_extractor.py turing_data/config_100.h5
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample


def bearing_distance(a,b):
    return np.abs((a-b+180)%360-180)


def grow_train(times,period,tolerance,max_skip):
    """Greedy forward timing chain; choose strongest seed from first 1000 TOAs."""
    best=[]
    seeds=np.arange(min(len(times),1000))
    for seed in seeds:
        chain=[int(seed)]
        last=float(times[seed])
        for k in range(seed+1,len(times)):
            dt=float(times[k]-last)
            if dt<tolerance:
                continue
            n=int(np.rint(dt/period))
            if n<1:
                continue
            if n>max_skip:
                break
            if abs(dt-n*period)<=tolerance:
                chain.append(int(k))
                last=float(times[k])
        if len(chain)>len(best):
            best=chain
    return np.asarray(best,dtype=int)


def candidate_periods(times,minimum,maximum,bin_width,max_neighbours):
    # Sample nearby forward differences, not all O(N^2) differences.
    diffs=[]
    for shift in range(1,min(max_neighbours,len(times)-1)+1):
        d=times[shift:]-times[:-shift]
        diffs.append(d[(d>=minimum)&(d<=maximum)])
    if not diffs:
        return []
    diffs=np.concatenate(diffs)
    if not len(diffs):
        return []
    edges=np.arange(minimum,maximum+bin_width*1.01,bin_width)
    counts,edges=np.histogram(diffs,bins=edges)
    # Largest peaks; duplicates and harmonics remain possible.
    peaks=np.argsort(counts)[::-1]
    return [float((edges[i]+edges[i+1])/2) for i in peaks if counts[i]>0]


def extract(data,names,args):
    toa=data[:,names.index("ToA")]
    aoa=data[:,names.index("AoA")]
    rad=np.deg2rad(aoa)
    angular=np.column_stack([np.sin(rad),np.cos(rad)])
    eps=2*np.sin(np.deg2rad(args.aoa_eps)/2)
    sectors=DBSCAN(eps=eps,min_samples=args.aoa_min_samples).fit_predict(angular)
    trains=[]
    assigned=np.full(len(toa),-1,dtype=int)
    for sector in sorted(set(sectors)-{-1}):
        indices=np.flatnonzero(sectors==sector)
        indices=indices[np.argsort(toa[indices])]
        available=indices.copy()
        for iteration in range(args.max_trains_per_sector):
            if len(available)<args.min_pulses:
                break
            times=toa[available]
            periods=candidate_periods(times,args.pri_min,args.pri_max,
                                      args.pri_bin,args.neighbours)[:args.top_periods]
            best_idx=np.array([],dtype=int)
            best_pri=None
            for pri in periods:
                chain=grow_train(times,pri,args.tolerance,args.max_skip)
                if len(chain)>len(best_idx):
                    best_idx=chain
                    best_pri=pri
            if len(best_idx)<args.min_pulses:
                break
            chosen=available[best_idx]
            # A train should be directionally coherent, even within a broad
            # DBSCAN sector. This does not predict moving bearings yet.
            angle=np.angle(np.mean(np.exp(1j*rad[chosen])))
            center=float(np.rad2deg(angle))
            consistent=bearing_distance(aoa[chosen],center)<=args.train_aoa_gate
            chosen=chosen[consistent]
            if len(chosen)<args.min_pulses:
                break
            train_id=len(trains)
            assigned[chosen]=train_id
            diffs=np.diff(toa[chosen])
            trains.append({"train_id":train_id,"sector":int(sector),
                           "pulse_count":int(len(chosen)),
                           "estimated_pri":best_pri,
                           "aoa_center_degrees":center,
                           "toa_first":float(toa[chosen[0]]),
                           "toa_last":float(toa[chosen[-1]]),
                           "pdw_indices":[int(i) for i in chosen],
                           "median_observed_interval":float(np.median(diffs)) if len(diffs) else None})
            available=available[~np.isin(available,chosen)]
    return trains,assigned,sectors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-eps",type=float,default=2.)
    p.add_argument("--aoa-min-samples",type=int,default=8)
    p.add_argument("--train-aoa-gate",type=float,default=3.)
    p.add_argument("--pri-min",type=float,default=25.)
    p.add_argument("--pri-max",type=float,default=1600.)
    p.add_argument("--pri-bin",type=float,default=1.)
    p.add_argument("--neighbours",type=int,default=12)
    p.add_argument("--top-periods",type=int,default=12)
    p.add_argument("--tolerance",type=float,default=1.)
    p.add_argument("--max-skip",type=int,default=8)
    p.add_argument("--min-pulses",type=int,default=12)
    p.add_argument("--max-trains-per-sector",type=int,default=30)
    p.add_argument("--output",default="turing_pulse_trains.json")
    a=p.parse_args()
    if not (0<a.pri_min<a.pri_max and a.pri_bin>0 and a.tolerance>0
            and a.max_skip>=1 and a.min_pulses>=2):
        p.error("Invalid PRI, tolerance, skip or pulse count parameters")
    data,truth,names,total=load_sample(a.file,a.limit)
    trains,assigned,sectors=extract(data,names,a)
    summary={"file":a.file,"sample_pdws":len(data),
             "angular_sectors":int(len(set(sectors)-{-1})),
             "recovered_trains":len(trains),
             "assigned_pdws":int(np.sum(assigned>=0)),
             "unassigned_pdws":int(np.sum(assigned<0)),
             "train_sizes":[t["pulse_count"] for t in trains],
             "limitations":[
                 "Constant-PRI hypothesis with greedy chaining; no jitter/stagger model.",
                 "Greedy search may join different emitters at PRI harmonics.",
                 "Fixed AOA coherence gate; moving bearing tracks are not yet supported.",
                 "No explicit missed-detection or collision probability model.",
                 "Ground truth is evaluation-only, not used in extraction."]}
    if truth is not None:
        purities=[]
        for t in trains:
            labels,counts=np.unique(truth[t["pdw_indices"]],return_counts=True)
            purity=float(counts.max()/counts.sum())
            purities.append(purity)
            t["evaluation_only"]={"purity":purity,"truth_emitters":int(len(labels))}
        summary["evaluation_only"]={
            "mean_train_purity":float(np.mean(purities)) if purities else None,
            "weighted_train_purity":float(np.average(purities,weights=[t["pulse_count"] for t in trains])) if trains else None}
    with open(a.output,"w") as f:
        json.dump({"summary":summary,"trains":trains},f,indent=2)
    print(json.dumps(summary,indent=2))
    print("Full train memberships written to",a.output)


if __name__=="__main__":
    main()
