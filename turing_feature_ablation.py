"""Label-blind ablation: timing, timing+AOA, timing+RF, and all three.

Reference clusters 12/32 supply phase, circular AOA and a multimodal RF
support model. Cluster 13 is classified without truth labels. Truth is used
only for final evaluation. Gates are exploratory, not Bayesian posteriors.

Usage:
 python turing_feature_ablation.py turing_data/config_100.h5
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def angle_distance(x,y):
    return np.abs((x-y+180)%360-180)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--reference",type=int,nargs="+",default=[12,32])
    p.add_argument("--target",type=int,default=13)
    p.add_argument("--period",type=float,default=25.)
    p.add_argument("--time-gate",type=float,default=1.)
    p.add_argument("--aoa-gate",type=float,default=3.)
    p.add_argument("--rf-gate",type=float,default=30.)
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    a=p.parse_args()
    if a.period<=0 or not 0<a.time_gate<a.period/2 or not 0<a.aoa_gate<=180 or a.rf_gate<=0:
        p.error("Invalid gate parameters")
    data,truth,names,total=load_sample(a.file,a.limit)
    pred=DBSCAN(eps=a.eps,min_samples=a.min_samples).fit_predict(build_features(data,names))
    ref=np.isin(pred,a.reference)
    tgt=pred==a.target
    if not ref.any() or not tgt.any() or np.any(ref&tgt):
        p.error("Reference/target clusters missing or overlapping")
    toa=data[:,names.index("ToA")]
    rf=data[:,names.index("Frequency")]
    aoa=data[:,names.index("AoA")]
    phase_vector=np.mean(np.exp(2j*np.pi*np.remainder(toa[ref],a.period)/a.period))
    phase=float(np.mod(np.angle(phase_vector)*a.period/(2*np.pi),a.period))
    residual=np.abs(np.remainder(toa[tgt]-phase+a.period/2,a.period)-a.period/2)
    time_ok=residual<=a.time_gate
    az=np.mean(np.exp(1j*np.deg2rad(aoa[ref])))
    az_center=float(np.rad2deg(np.angle(az)))
    aoa_ok=angle_distance(aoa[tgt],az_center)<=a.aoa_gate
    # Nearest observed reference RF accommodates multiple frequency bands.
    # This may overfit reference RF noise; threshold is user-configurable.
    rf_reference=np.sort(rf[ref])
    pos=np.searchsorted(rf_reference,rf[tgt])
    left=np.abs(rf[tgt]-rf_reference[np.maximum(pos-1,0)])
    right=np.abs(rf[tgt]-rf_reference[np.minimum(pos,len(rf_reference)-1)])
    rf_distance=np.minimum(left,right)
    rf_ok=rf_distance<=a.rf_gate
    masks={"timing":time_ok,"timing_aoa":time_ok&aoa_ok,
           "timing_rf":time_ok&rf_ok,"timing_aoa_rf":time_ok&aoa_ok&rf_ok}
    output={"file":a.file,"reference_clusters":a.reference,"target_cluster":a.target,
            "reference_pdws":int(ref.sum()),"target_pdws":int(tgt.sum()),
            "model":{"period":a.period,"phase":phase,
                     "phase_concentration":float(np.abs(phase_vector)),
                     "aoa_center_degrees":az_center,
                     "time_gate":a.time_gate,"aoa_gate_degrees":a.aoa_gate,
                     "rf_nearest_reference_gate":a.rf_gate},
            "results":[]}
    dominant=None
    if truth is not None:
        ids,counts=np.unique(truth[ref],return_counts=True)
        dominant=ids[np.argmax(counts)]
        positive=truth[tgt]==dominant
    for name,mask in masks.items():
        item={"method":name,"accepted":int(mask.sum()),
              "accepted_fraction":float(np.mean(mask))}
        if dominant is not None:
            tp=int(np.count_nonzero(mask&positive))
            fp=int(np.count_nonzero(mask&~positive))
            fn=int(np.count_nonzero(~mask&positive))
            item["evaluation_only"]={
                "reference_dominant_truth_label":str(dominant),
                "true_positives":tp,"false_positives":fp,"false_negatives":fn,
                "precision":tp/(tp+fp) if tp+fp else None,
                "recall":tp/(tp+fn) if tp+fn else None}
            if fp:
                values,counts=np.unique(truth[tgt][mask&~positive],return_counts=True)
                item["evaluation_only"]["false_by_emitter"]={
                    str(k):int(n) for k,n in zip(values,counts)}
        output["results"].append(item)
    output["limitations"]=[
        "Fixed phase lattice, reference clusters and thresholds are manually specified.",
        "AOA can drift or overlap for co-located emitters; fixed 3-degree gate is not calibrated.",
        "Nearest-reference RF gate may accept unrelated emitters at overlapping frequencies.",
        "Gates are hard decisions, not Bayesian likelihoods.",
        "Performance on this one target cluster does not establish generalisation."]
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
