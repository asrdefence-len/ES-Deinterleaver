"""Account for every truth emitter in direct AOA/RF/PW grouping.

Usage: python turing_emitter_accounting.py turing_data/config_100.h5
Truth is used for diagnostics only, never to form clusters.
"""
import argparse
import csv
from collections import Counter
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale


def audit(data, truth, names, angle, min_samples):
    rf=robust_scale(data[:,names.index("Frequency")])
    pw=robust_scale(data[:,names.index("PulseWidth")])
    cluster=groups(data,names,"C",angle,min_samples,rf,pw)
    dominant={}
    for g in np.unique(cluster):
        if g<0:
            continue
        counts=Counter(map(int,truth[cluster==g]))
        dominant[int(g)]=counts.most_common(1)[0][0]
    rows=[]
    for emitter in sorted(map(int,np.unique(truth))):
        mask=truth==emitter
        assigned=mask & (cluster>=0)
        own=np.zeros(len(truth),dtype=bool)
        for g,leader in dominant.items():
            if leader==emitter:
                own |= cluster==g
        own_count=int(np.count_nonzero(mask & own))
        mixed=int(np.count_nonzero(assigned & ~own))
        noise=int(np.count_nonzero(mask & (cluster<0)))
        groups_present=sorted(map(int,np.unique(cluster[assigned])))
        groups_dominant=sorted(g for g in groups_present if dominant[g]==emitter)
        rows.append({"emitter":emitter,"total":int(mask.sum()),
                     "in_dominant_groups":own_count,
                     "in_other_emitter_groups":mixed,"ungrouped":noise,
                     "groups_present":len(groups_present),
                     "groups_dominant":len(groups_dominant),
                     "group_ids":",".join(map(str,groups_present))})
    return rows,cluster


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.0)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--output",default="turing_emitter_accounting.csv")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Dataset needs truth labels for this diagnostic")
    rows,labels=audit(data,truth,names,a.aoa_deg,a.min_samples)
    with open(a.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    absent=[r for r in rows if r["groups_present"]==0]
    no_own=[r for r in rows if r["groups_dominant"]==0]
    print(f"AOA scale: {a.aoa_deg} deg; truth emitters: {len(rows)}; clusters: {len(set(labels)-{-1})}")
    print(f"Entirely ungrouped emitters ({len(absent)}): {[r['emitter'] for r in absent]}")
    print(f"No dominant group ({len(no_own)}): {[r['emitter'] for r in no_own]}")
    print("EMITTER  TOTAL  OWN-GROUP  OTHER-GROUP  NOISE  GROUPS  DOMINANT")
    for r in rows:
        if r["groups_dominant"]==0 or r["in_other_emitter_groups"]>0 or r["ungrouped"]>0:
            print(f"{r['emitter']:>7} {r['total']:>6} {r['in_dominant_groups']:>10} {r['in_other_emitter_groups']:>12} {r['ungrouped']:>6} {r['groups_present']:>6} {r['groups_dominant']:>9}")
    print("Totals:",{k:sum(r[k] for r in rows) for k in ("total","in_dominant_groups","in_other_emitter_groups","ungrouped")})
    print("Saved",a.output)


if __name__=="__main__":
    main()
