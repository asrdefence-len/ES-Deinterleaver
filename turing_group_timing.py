"""Exploratory timing test for Turing DBSCAN pulse groups.

Uses ONLY predicted cluster IDs to select observations and construct timing
statistics. Ground truth is read solely to report composition after inference.
This does not establish common emitter identity or a calibrated Bayes factor.

Usage:
 python turing_group_timing.py turing_data/config_100.h5 --limit 10000 --clusters 12 13 32
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def nearest_distance(a,b):
    """Nearest cross-group TOA distance for each pulse in a."""
    if not len(a) or not len(b):
        return np.array([])
    b=np.sort(b)
    pos=np.searchsorted(b,a)
    left=np.abs(a-b[np.maximum(pos-1,0)])
    right=np.abs(a-b[np.minimum(pos,len(b)-1)])
    return np.minimum(left,right)


def analyse(path,limit,clusters,eps,min_samples):
    data,truth,names,total=load_sample(path,limit)
    if "ToA" not in names:
        raise ValueError("Missing ToA feature")
    pred=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(build_features(data,names))
    toa=data[:,names.index("ToA")]
    report={"file":path,"sample_pdws":len(data),"total_file_pdws":total,
            "selected_clusters":clusters,"groups":[],"pairs":[]}
    for cl in clusters:
        mask=pred==cl
        t=np.sort(toa[mask])
        if not len(t):
            raise ValueError(f"Cluster {cl} not present")
        d=np.diff(t)
        group={"cluster":cl,"pdws":int(len(t)),
               "toa_span":float(np.ptp(t)),
               "median_observed_interval":float(np.median(d)) if len(d) else None,
               "p10_observed_interval":float(np.percentile(d,10)) if len(d) else None,
               "p90_observed_interval":float(np.percentile(d,90)) if len(d) else None}
        if truth is not None:
            ids,counts=np.unique(truth[mask],return_counts=True)
            group["truth_composition_diagnostic_only"]=[
                {"emitter":str(k),"pdws":int(v)}
                for k,v in sorted(zip(ids,counts),key=lambda p:-p[1])]
        report["groups"].append(group)
    for i,a in enumerate(clusters):
        for b in clusters[i+1:]:
            ta=np.sort(toa[pred==a])
            tb=np.sort(toa[pred==b])
            da=nearest_distance(ta,tb)
            db=nearest_distance(tb,ta)
            report["pairs"].append({
                "clusters":[a,b],
                "median_nearest_cross_group_toa":float(np.median(da)),
                "p10_nearest_cross_group_toa":float(np.percentile(da,10)),
                "median_reverse_nearest_toa":float(np.median(db)),
                "note":"Proximity alone cannot distinguish shared emitter from independent interleaved trains."
            })
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--clusters",type=int,nargs="+",default=[12,13,32])
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    a=p.parse_args()
    if a.limit<2 or a.eps<=0 or a.min_samples<1 or len(set(a.clusters))!=len(a.clusters):
        p.error("Invalid limit, eps, min-samples or duplicate cluster IDs")
    print(json.dumps(analyse(a.file,a.limit,a.clusters,a.eps,a.min_samples),indent=2))


if __name__=="__main__":
    main()
