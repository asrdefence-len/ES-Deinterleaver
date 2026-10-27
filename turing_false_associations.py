"""Diagnose false timing-lattice associations in mixed Turing cluster 13.

Inference is label blind. Ground truth is consulted only for post-selection
error attribution and per-emitter diagnostic summaries.

Usage:
 python turing_false_associations.py turing_data/config_100.h5
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def residual(t,period,phase):
    return np.abs(np.remainder(t-phase+period/2,period)-period/2)


def circular_phase(t,period):
    z=np.mean(np.exp(2j*np.pi*np.remainder(t,period)/period))
    return float(np.mod(np.angle(z)*period/(2*np.pi),period)),float(np.abs(z))


def describe(data,mask,names):
    result={"count":int(np.sum(mask))}
    for key in ("Frequency","PulseWidth","AoA"):
        if key in names and np.any(mask):
            v=data[mask,names.index(key)]
            result[key]={"median":float(np.median(v)),
                         "p10":float(np.percentile(v,10)),
                         "p90":float(np.percentile(v,90))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--reference",type=int,nargs="+",default=[12,32])
    p.add_argument("--target",type=int,default=13)
    p.add_argument("--period",type=float,default=25.)
    p.add_argument("--threshold",type=float,default=1.)
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    a=p.parse_args()
    if a.period<=0 or not 0<a.threshold<a.period/2:
        p.error("Invalid period or threshold")
    data,truth,names,total=load_sample(a.file,a.limit)
    pred=DBSCAN(eps=a.eps,min_samples=a.min_samples).fit_predict(build_features(data,names))
    ref=np.isin(pred,a.reference)
    target=pred==a.target
    if not ref.any() or not target.any() or np.any(ref&target):
        p.error("Reference and target must exist and be disjoint")
    toa=data[:,names.index("ToA")]
    phase,concentration=circular_phase(toa[ref],a.period)
    r=residual(toa,a.period,phase)
    selected=target&(r<=a.threshold)
    rejected=target&~selected
    report={"reference_clusters":a.reference,"target_cluster":a.target,
            "reference_count":int(ref.sum()),"target_count":int(target.sum()),
            "period":a.period,"threshold":a.threshold,
            "reference_phase":phase,"reference_concentration":concentration,
            "accepted_count":int(selected.sum()),"rejected_count":int(rejected.sum()),
            "reference_features":describe(data,ref,names),
            "accepted_features":describe(data,selected,names),
            "rejected_features":describe(data,rejected,names)}
    if truth is not None:
        labels,counts=np.unique(truth[ref],return_counts=True)
        dominant=labels[np.argmax(counts)]
        report["evaluation_only"]={"reference_dominant_label":str(dominant),
                                   "target_emitter_breakdown":[]}
        for label in np.unique(truth[target]):
            mask=target&(truth==label)
            sel=mask&selected
            rej=mask&rejected
            report["evaluation_only"]["target_emitter_breakdown"].append({
                "label":str(label),"total":int(mask.sum()),
                "accepted":int(sel.sum()),"rejected":int(rej.sum()),
                "accepted_fraction":float(sel.sum()/mask.sum()),
                "all_features":describe(data,mask,names),
                "accepted_features":describe(data,sel,names),
                "median_phase_residual":float(np.median(r[mask]))})
        report["evaluation_only"]["target_emitter_breakdown"].sort(
            key=lambda d:-d["accepted"])
        report["evaluation_only"]["correct_acceptances"]=int(np.sum(selected&(truth==dominant)))
        report["evaluation_only"]["false_acceptances"]=int(np.sum(selected&(truth!=dominant)))
    report["interpretation"]="Per-emitter labels are post-inference diagnostics only; feature differences are not yet validated decision rules."
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
