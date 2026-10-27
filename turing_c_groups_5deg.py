"""Evaluate direct Approach C feature groups with 5-degree AOA scale.

No PRI requirement. Truth is used only for scoring, never clustering.
python turing_c_groups_5deg.py turing_data/config_100.h5
"""
import argparse
import json
from collections import Counter
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale


def evaluate(labels,truth,target):
    valid=labels>=0
    rows=[]
    for group in sorted(set(labels)-{-1}):
        ids=np.flatnonzero(labels==group)
        if truth is None:
            rows.append({"group":int(group),"pdws":len(ids)})
            continue
        counts=Counter(map(int,truth[ids]))
        dominant,n=counts.most_common(1)[0]
        rows.append({"group":int(group),"pdws":len(ids),
                     "dominant_emitter":dominant,"purity":n/len(ids),
                     "other_emitter_pdws":len(ids)-n,
                     "target_pdws":counts.get(target,0)})
    result={"candidate_groups":len(rows),"grouped_pdws":int(valid.sum()),
            "coverage":float(np.mean(valid))}
    if truth is not None:
        assigned=int(valid.sum())
        contamination=sum(r["other_emitter_pdws"] for r in rows)
        target_mask=truth==target
        result.update({"weighted_purity":(assigned-contamination)/assigned if assigned else None,
                       "cross_emitter_pdws":contamination,
                       "emitters_represented":int(len(np.unique(truth[valid]))),
                       "target_emitter":target,
                       "target_total":int(target_mask.sum()),
                       "target_grouped":int(np.count_nonzero(target_mask&valid)),
                       "target_group_count":sum(r["target_pdws"]>0 for r in rows),
                       "target_groups":[r for r in rows if r["target_pdws"]>0]})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--emitter",type=int,default=30)
    p.add_argument("--output",default="turing_c_groups_5deg.json")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    rf_scale=robust_scale(data[:,names.index("Frequency")])
    pw_scale=robust_scale(data[:,names.index("PulseWidth")])
    output={}
    for angle in (2.0,5.0):
        labels=groups(data,names,"C",angle,8,rf_scale,pw_scale)
        result=evaluate(labels,truth,a.emitter)
        output[str(angle)]=result
        print("AOA scale",angle,"degrees:",flush=True)
        print(json.dumps(result,indent=2),flush=True)
    output["notes"]=[
        "Direct feature grouping; no PRI filter or timing extractor.",
        "AOA scale is a combined DBSCAN distance scale, not a strict angular gate.",
        "Groups are candidate observations, not confirmed pulse trains.",
        "Ground-truth labels used for evaluation only.",
        "Emitter coverage alone does not imply each group is pure."
    ]
    with open(a.output,"w") as f:
        json.dump(output,f,indent=2)
    print("Saved",a.output)


if __name__=="__main__":
    main()
