"""Diagnose why the greedy pulse-train extractor misses dense AOA group 5.

No truth is used to discover groups or timing candidates. Truth, when present,
is used only to describe the resulting group and pulse train purity.

Usage:
 python turing_train_diagnostics.py turing_data/config_100.h5 --group 5
"""
import argparse
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample
from turing_pulse_train_extractor import candidate_periods, grow_train


def quantiles(x):
    if not len(x):
        return None
    return {str(p):float(np.percentile(x,p)) for p in (0,10,25,50,75,90,99,100)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--group",type=int,default=5)
    p.add_argument("--aoa-eps",type=float,default=2.)
    p.add_argument("--aoa-min-samples",type=int,default=8)
    p.add_argument("--pri-min",type=float,default=25.)
    p.add_argument("--pri-max",type=float,default=1600.)
    p.add_argument("--pri-bin",type=float,default=1.)
    p.add_argument("--neighbours",type=int,default=12)
    p.add_argument("--top-periods",type=int,default=20)
    p.add_argument("--tolerance",type=float,default=1.)
    p.add_argument("--max-skip",type=int,default=8)
    p.add_argument("--train-file",default="turing_pulse_trains.json")
    a=p.parse_args()
    if a.aoa_eps<=0 or a.pri_min<=0 or a.pri_max<=a.pri_min:
        p.error("Invalid angular or PRI range")
    data,truth,names,total=load_sample(a.file,a.limit)
    toa=data[:,names.index("ToA")]
    aoa=np.deg2rad(data[:,names.index("AoA")])
    angles=np.column_stack((np.sin(aoa),np.cos(aoa)))
    eps=2*np.sin(np.deg2rad(a.aoa_eps)/2)
    sectors=DBSCAN(eps=eps,min_samples=a.aoa_min_samples).fit_predict(angles)
    ids=np.flatnonzero(sectors==a.group)
    if not len(ids):
        p.error("No pulses in requested group")
    ids=ids[np.argsort(toa[ids])]
    times=toa[ids]
    dt=np.diff(times)
    periods=candidate_periods(times,a.pri_min,a.pri_max,a.pri_bin,a.neighbours)
    records=[]
    for period in periods[:a.top_periods]:
        chain=grow_train(times,period,a.tolerance,a.max_skip)
        z=np.mean(np.exp(2j*np.pi*np.remainder(times,period)/period))
        record={"candidate_period":period,"greedy_chain_length":int(len(chain)),
                "all_pulse_phase_concentration":float(np.abs(z))}
        if len(chain):
            record["chain_span"]=float(times[chain[-1]]-times[chain[0]])
            record["chain_median_interval"]=float(np.median(np.diff(times[chain]))) if len(chain)>1 else None
            if truth is not None:
                _,counts=np.unique(truth[ids[chain]],return_counts=True)
                record["evaluation_only_chain_purity"]=float(counts.max()/counts.sum())
        records.append(record)
    output={"file":a.file,"group":a.group,"group_pdws":int(len(ids)),
            "group_time_span":float(times[-1]-times[0]),
            "adjacent_toa_difference_quantiles":quantiles(dt),
            "candidate_periods":records,
            "note":"Candidate periods come from nearby TOA differences, not an exhaustive PRI search."}
    if truth is not None:
        labels,counts=np.unique(truth[ids],return_counts=True)
        order=np.argsort(counts)[::-1]
        output["evaluation_only"]={"group_emitters":[{"label":str(labels[i]),"count":int(counts[i])}
                                                          for i in order],
                                   "dominant_emitter_intervals":{}}
        for i in order[:4]:
            subset=np.sort(toa[ids[truth[ids]==labels[i]]])
            output["evaluation_only"]["dominant_emitter_intervals"][str(labels[i])]={
                "pdws":int(len(subset)),"adjacent_toa_difference_quantiles":quantiles(np.diff(subset))}
    if os.path.isfile(a.train_file):
        with open(a.train_file) as f:
            saved=json.load(f)
        output["existing_extractor"]={"trains_in_group":[
            {"train_id":t["train_id"],"pulse_count":t["pulse_count"],
             "estimated_pri":t["estimated_pri"],
             "median_observed_interval":t["median_observed_interval"],
             "evaluation_only":t.get("evaluation_only")}
            for t in saved.get("trains",[]) if t.get("sector")==a.group]}
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
