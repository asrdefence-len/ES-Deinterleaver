"""Recover pulses from a mixed DBSCAN group using a reference timing lattice.

Reference clusters define a phase at a fixed candidate period. Target group
pulses are accepted only by their TOA phase, without looking at truth labels.
Reports precision/recall AFTER inference, including reference contamination.
A control reference group is also evaluated to expose false matches.

Usage:
 python turing_lattice_recovery.py turing_data/config_100.h5 --reference 12 32 --target 13 --period 25
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def signed_residual(t,period,phase):
    return np.remainder(t-phase+period/2,period)-period/2


def circular_mean_phase(t,period):
    z=np.mean(np.exp(2j*np.pi*np.remainder(t,period)/period))
    return float(np.mod(np.angle(z)*period/(2*np.pi),period)),float(np.abs(z))


def evaluate(reference_t,target_t,period,threshold):
    phase,concentration=circular_mean_phase(reference_t,period)
    residual=np.abs(signed_residual(target_t,period,phase))
    return {"phase":phase,"reference_concentration":concentration,
            "accepted":residual<=threshold,
            "residual":residual}


def composition(labels,mask):
    vals,counts=np.unique(labels[mask],return_counts=True)
    return [{"label":str(v),"count":int(n)} for v,n in sorted(zip(vals,counts),key=lambda z:-z[1])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--reference",nargs="+",type=int,default=[12,32])
    p.add_argument("--target",type=int,default=13)
    p.add_argument("--control-reference",type=int,default=16)
    p.add_argument("--period",type=float,default=25.)
    p.add_argument("--threshold",type=float,default=1.)
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    args=p.parse_args()
    if args.period<=0 or args.threshold<=0 or args.threshold>=args.period/2:
        p.error("Require period>0 and 0<threshold<period/2")
    if args.target in args.reference or args.control_reference==args.target:
        p.error("Target must be disjoint from reference groups")
    data,truth,names,total=load_sample(args.file,args.limit)
    pred=DBSCAN(eps=args.eps,min_samples=args.min_samples).fit_predict(build_features(data,names))
    toa=data[:,names.index("ToA")]
    target_mask=pred==args.target
    if not np.any(target_mask):
        p.error("Target cluster not found")
    output={"file":args.file,"sample_pdws":len(data),"target_cluster":args.target,
            "period":args.period,"phase_acceptance_threshold":args.threshold,
            "random_uniform_phase_acceptance_fraction":2*args.threshold/args.period,
            "note":"Period and threshold are specified in advance; this is not a calibrated emitter probability.",
            "runs":[]}
    for name,ref_ids in [("candidate",args.reference),("control",[args.control_reference])]:
        ref_mask=np.isin(pred,ref_ids)
        if not np.any(ref_mask):
            p.error(f"Reference {ref_ids} not found")
        result=evaluate(toa[ref_mask],toa[target_mask],args.period,args.threshold)
        accepted=result.pop("accepted")
        residual=result.pop("residual")
        entry={"name":name,"reference_clusters":ref_ids,"reference_pdws":int(ref_mask.sum()),
               "target_pdws":int(target_mask.sum()),"accepted_pdws":int(accepted.sum()),
               "accepted_fraction":float(accepted.mean()),
               "median_absolute_phase_residual":float(np.median(residual)),**result}
        if truth is not None:
            # Oracle reference identity is used ONLY for reporting performance.
            ref_labels=truth[ref_mask]
            values,counts=np.unique(ref_labels,return_counts=True)
            dominant=values[np.argmax(counts)]
            target_labels=truth[target_mask]
            positive=target_labels==dominant
            tp=int(np.count_nonzero(accepted & positive))
            fp=int(np.count_nonzero(accepted & ~positive))
            fn=int(np.count_nonzero(~accepted & positive))
            entry["evaluation_only"]={
                "reference_dominant_truth_label":str(dominant),
                "reference_truth_composition":composition(truth,ref_mask),
                "target_truth_composition":composition(truth,target_mask),
                "true_positives":tp,"false_positives":fp,"false_negatives":fn,
                "precision":float(tp/(tp+fp)) if tp+fp else None,
                "recall":float(tp/(tp+fn)) if tp+fn else None}
        output["runs"].append(entry)
    output["limitations"]=[
        "Reference clusters and period are manually selected, not discovered automatically.",
        "A shared timing lattice may be shared by independent or synchronized emitters.",
        "Fixed-period model does not handle staggered/jittered PRI or drift.",
        "The control reference may have weak phase concentration; compare its results cautiously.",
        "Phase threshold was not tuned or calibrated on independent data."]
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
