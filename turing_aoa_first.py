"""Compare RF/PW/AOA DBSCAN against AOA-only candidate grouping on Turing PDWs.

AOA groups are *bearing sectors*, not emitter identities. Quantifies emitter
fragmentation, group mixing, and angular overlap. Truth labels are only used
after unsupervised grouping for evaluation.

Usage:
 python turing_aoa_first.py turing_data/config_100.h5 --limit 10000
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from turing_baseline import load_sample, build_features


def metrics(truth,pred):
    groups=np.unique(pred[pred>=0])
    result={"candidate_groups":int(len(groups)),
            "unassigned_pdws":int(np.sum(pred<0)),
            "unassigned_fraction":float(np.mean(pred<0))}
    if truth is None:
        return result
    h,c,v=homogeneity_completeness_v_measure(truth,pred)
    result["adjusted_rand_index"]=float(adjusted_rand_score(truth,pred))
    result["homogeneity"]=float(h)
    result["completeness"]=float(c)
    result["v_measure"]=float(v)
    split=0
    mixed=0
    for label in np.unique(truth):
        associated=np.unique(pred[(truth==label)&(pred>=0)])
        split+=int(len(associated)>1)
    for group in groups:
        mixed+=int(len(np.unique(truth[pred==group]))>1)
    result["emitters_in_multiple_groups"]=split
    result["groups_with_multiple_emitters"]=mixed
    return result


def angular_distance(a,b):
    return np.abs((a-b+180)%360-180)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--aoa-eps-degrees",type=float,default=2.)
    p.add_argument("--aoa-min-samples",type=int,default=8)
    p.add_argument("--overlap-degrees",type=float,default=3.)
    a=p.parse_args()
    if a.aoa_eps_degrees<=0 or a.overlap_degrees<=0:
        p.error("AOA thresholds must be positive")
    data,truth,names,total=load_sample(a.file,a.limit)
    aoa=data[:,names.index("AoA")]
    # Chord distance on the unit circle; no -180/+180 discontinuity.
    radians=np.deg2rad(aoa)
    angular_features=np.column_stack([np.sin(radians),np.cos(radians)])
    aoa_eps=2*np.sin(np.deg2rad(a.aoa_eps_degrees)/2)
    aoa_pred=DBSCAN(eps=aoa_eps,min_samples=a.aoa_min_samples).fit_predict(angular_features)
    baseline_pred=DBSCAN(eps=a.eps,min_samples=a.min_samples).fit_predict(build_features(data,names))
    report={"file":a.file,"sample_pdws":len(data),
            "aoa_eps_degrees":a.aoa_eps_degrees,
            "aoa_min_samples":a.aoa_min_samples,
            "baseline":metrics(truth,baseline_pred),
            "aoa_only":metrics(truth,aoa_pred),
            "aoa_candidate_groups":[]}
    for cl in sorted(set(aoa_pred)-{-1}):
        mask=aoa_pred==cl
        z=np.mean(np.exp(1j*radians[mask]))
        center=float(np.rad2deg(np.angle(z)))
        item={"group":int(cl),"pdws":int(mask.sum()),
              "mean_aoa_degrees":center,
              "aoa_p90_absolute_deviation_degrees":float(np.percentile(angular_distance(aoa[mask],center),90))}
        if truth is not None:
            ids,counts=np.unique(truth[mask],return_counts=True)
            item["evaluation_only"]={"unique_emitters":int(len(ids)),
                                     "largest_emitter_fraction":float(counts.max()/counts.sum()),
                                     "top_emitters":[{"label":str(k),"pdws":int(n)}
                                         for k,n in sorted(zip(ids,counts),key=lambda x:-x[1])[:8]]}
        report["aoa_candidate_groups"].append(item)
    if truth is not None:
        labels=np.unique(truth)
        centers=[]
        for label in labels:
            m=truth==label
            z=np.mean(np.exp(1j*radians[m]))
            centers.append(float(np.rad2deg(np.angle(z))))
        close=[]
        for i in range(len(labels)):
            for j in range(i+1,len(labels)):
                distance=float(angular_distance(centers[i],centers[j]))
                if distance<=a.overlap_degrees:
                    close.append({"emitter_a":str(labels[i]),"emitter_b":str(labels[j]),
                                  "center_separation_degrees":distance,
                                  "counts":[int(np.sum(truth==labels[i])),int(np.sum(truth==labels[j]))]})
        close.sort(key=lambda d:d["center_separation_degrees"])
        report["evaluation_only"]={
            "truth_emitters":int(len(labels)),
            "emitter_pairs_within_threshold":int(len(close)),
            "overlap_threshold_degrees":a.overlap_degrees,
            "closest_pairs":close[:25],
            "note":"Close truth AOA centres indicate difficult cases, not necessarily overlapping distributions."}
    report["limitations"]=[
        "AOA-only DBSCAN creates candidate bearing groups, not emitter tracks.",
        "Density connectivity can merge distinct nearby bearings and split sparse ones.",
        "This is not yet AOA-followed-by-timing separation.",
        "Ground-truth labels are used only for evaluation.",
        "Static AOA is not valid for all moving platforms or long observation windows."]
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
