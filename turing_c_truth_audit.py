"""Ground-truth audit of Approach C: where did unassigned PDWs go?

Re-runs the unchanged C grouping and extractor, then uses truth labels ONLY
to audit losses. Does not modify or improve the deinterleaver.

Usage:
 python turing_c_truth_audit.py turing_data/config_100.h5
"""
import argparse
import csv
import json
from collections import Counter
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import robust_scale, groups, extract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.0)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--rf-scale",type=float)
    p.add_argument("--pw-scale",type=float)
    p.add_argument("--min-train",type=int,default=8)
    p.add_argument("--max-trains-per-group",type=int,default=20)
    p.add_argument("--relative-tolerance",type=float,default=.05)
    p.add_argument("--output",default="turing_c_truth_audit.csv")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Ground truth labels are required for this audit")
    rf_scale=a.rf_scale if a.rf_scale is not None else robust_scale(data[:,names.index("Frequency")])
    pw_scale=a.pw_scale if a.pw_scale is not None else robust_scale(data[:,names.index("PulseWidth")])
    grouping=groups(data,names,"C",a.aoa_deg,a.min_samples,rf_scale,pw_scale)
    assigned,trains=extract(data[:,names.index("ToA")],grouping,
                            a.min_train,a.max_trains_per_group,a.relative_tolerance)
    grouped=grouping>=0
    recovered=assigned>=0
    # Grouping loss and timing loss are mutually exclusive.
    grouping_loss=~grouped
    timing_loss=grouped&~recovered
    rows=[]
    for label in np.unique(truth):
        m=truth==label
        total=int(m.sum())
        got=int(np.count_nonzero(m&recovered))
        no_group=int(np.count_nonzero(m&grouping_loss))
        no_train=int(np.count_nonzero(m&timing_loss))
        train_ids=np.unique(assigned[m&recovered])
        rows.append({"emitter":str(label),"total_pdws":total,
                     "assigned_pdws":got,"unassigned_pdws":total-got,
                     "grouping_rejected":no_group,"timing_unassigned":no_train,
                     "recovery_percent":round(100*got/total,2),
                     "recovered_train_count":int(len(train_ids))})
    rows.sort(key=lambda x:(-x["unassigned_pdws"],-x["total_pdws"]))
    with open(a.output,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary={"sample_pdws":len(data),
             "assigned":int(recovered.sum()),
             "unassigned":int((~recovered).sum()),
             "rejected_by_initial_grouping":int(grouping_loss.sum()),
             "grouped_but_not_in_recovered_train":int(timing_loss.sum()),
             "emitters_with_no_assigned_pdws":sum(r["assigned_pdws"]==0 for r in rows),
             "emitters_with_at_least_100_pdws_but_none_assigned":
                [r["emitter"] for r in rows if r["total_pdws"]>=100 and r["assigned_pdws"]==0],
             "top_15_emitters_by_unassigned_pdws":rows[:15],
             "note":"Truth emitter labels cannot establish whether unassigned pulses form valid pulse trains. This audit identifies the loss stage and source emitter only."}
    print(json.dumps(summary,indent=2))
    print("All per-emitter rows saved to",a.output)


if __name__=="__main__":
    main()
