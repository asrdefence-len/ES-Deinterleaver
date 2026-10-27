"""Compare direct AOA+RF and AOA+RF+PW grouping without timing rejection.

Run: python turing_direct_grouping_ab.py turing_data/config_100.h5
Ground truth is used only for evaluation, never for clustering.
"""
import argparse
import csv
from collections import Counter
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale


def score(labels, truth):
    valid=labels>=0
    dominant={}
    for group in np.unique(labels[valid]):
        counts=Counter(map(int,truth[labels==group]))
        dominant[int(group)]=counts.most_common(1)[0][0]
    rows=[]
    for emitter in sorted(map(int,np.unique(truth))):
        mask=truth==emitter
        grouped=mask&valid
        own=sum(int(np.count_nonzero(mask&(labels==g))) for g,e in dominant.items() if e==emitter)
        rows.append({"emitter":emitter,"total":int(mask.sum()),
                     "grouped":int(grouped.sum()),"dominant_group_pdws":own,
                     "other_group_pdws":int(grouped.sum())-own,
                     "noise":int(np.count_nonzero(mask&~valid)),
                     "groups_present":len(set(map(int,labels[grouped]))),
                     "dominant_groups":sum(e==emitter for e in dominant.values())})
    correct=sum(r["dominant_group_pdws"] for r in rows)
    assigned=int(valid.sum())
    summary={"groups":len(dominant),"grouped":assigned,"coverage":assigned/len(labels),
             "purity":correct/assigned if assigned else 0,
             "cross_emitter":assigned-correct,
             "emitters_with_any_grouped_pdw":sum(r["grouped"]>0 for r in rows),
             "emitters_with_dominant_group":sum(r["dominant_groups"]>0 for r in rows)}
    return summary,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--output",default="turing_direct_grouping_ab.csv")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Dataset truth labels required for scoring")
    rf_scale=robust_scale(data[:,names.index("Frequency")])
    pw_scale=robust_scale(data[:,names.index("PulseWidth")])
    results={}
    for mode in ("B","C"):
        labels=groups(data,names,mode,a.aoa_deg,a.min_samples,rf_scale,pw_scale)
        results[mode]=score(labels,truth)
    print(f"AOA scale={a.aoa_deg} degrees; min_samples={a.min_samples}; RF scale={rf_scale:.6g}; PW scale={pw_scale:.6g}")
    print("B = AOA+RF; C = AOA+RF+PW. No PRI extraction.")
    for mode in ("B","C"):
        print(mode,results[mode][0])
    print("\nEmitter-by-emitter differences (B vs C):")
    print("ID  TOTAL  B_GROUPED  C_GROUPED  B_OWN  C_OWN  B_OTHER  C_OTHER  B_NOISE  C_NOISE")
    b={r["emitter"]:r for r in results["B"][1]}
    c={r["emitter"]:r for r in results["C"][1]}
    for e in sorted(b):
        x,y=b[e],c[e]
        if e in (18,22,30,33,54,79) or any(x[k]!=y[k] for k in ("grouped","dominant_group_pdws","other_group_pdws")):
            print(f"{e:2d} {x['total']:6d} {x['grouped']:10d} {y['grouped']:10d} {x['dominant_group_pdws']:6d} {y['dominant_group_pdws']:6d} {x['other_group_pdws']:8d} {y['other_group_pdws']:8d} {x['noise']:8d} {y['noise']:8d}")
    with open(a.output,"w",newline="") as f:
        fields=["mode"]+list(results["B"][1][0].keys())
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for mode in ("B","C"):
            for row in results[mode][1]:
                writer.writerow({"mode":mode,**row})
    print("\nSaved",a.output)
    print("NOTE: Purity is truth-label dominance within groups, not pulse-train coherence.")

if __name__=="__main__":
    main()
