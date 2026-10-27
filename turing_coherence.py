"""Exploratory label-blind pulse-train coherence diagnostic.

Compare common periodic lattice and independent periodic lattices using held-out
TOA phase residuals. An independent model is more flexible: scores are not
Bayes factors and do NOT establish common physical emitter identity.
Ground-truth labels are only displayed after model fitting.

Usage:
 python turing_coherence.py turing_data/config_100.h5 --clusters 12 32
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def circular_distance(t, period, phase):
    r=np.mod(t-phase,period)
    return np.minimum(r,period-r)


def fit_lattice(t, periods, phase_bins=64):
    """Search periods and phase; use robust median absolute lattice residual."""
    best=None
    for period in periods:
        phases=np.linspace(0,period,phase_bins,endpoint=False)
        r=np.mod(t[:,None]-phases[None,:]+period/2,period)-period/2
        med=np.median(np.abs(r),axis=0)
        j=int(np.argmin(med))
        candidate=(float(med[j]),float(period),float(phases[j]))
        if best is None or candidate[0]<best[0]:
            best=candidate
    return best


def model_fit(groups, periods, phase_bins):
    pooled=np.concatenate(groups)
    common=fit_lattice(pooled,periods,phase_bins)
    independent=[fit_lattice(t,periods,phase_bins) for t in groups]
    return common,independent


def evaluate(groups,periods,phase_bins):
    # Interleaved chronological split avoids a pure extrapolation test;
    # closely correlated pulses may still make this optimistic.
    train=[t[::2] for t in groups]
    test=[t[1::2] for t in groups]
    if any(len(t)<10 for t in train+test):
        raise ValueError("Need at least 20 pulses per group")
    common,ind=model_fit(train,periods,phase_bins)
    common_errors=[circular_distance(t,common[1],common[2]) for t in test]
    ind_errors=[circular_distance(t,m[1],m[2]) for t,m in zip(test,ind)]
    return {
        "common":{"period":common[1],"phase":common[2],
                  "train_median_absolute_residual":common[0],
                  "heldout_median_absolute_residual":float(np.median(np.concatenate(common_errors))),
                  "heldout_median_residual_by_group":[float(np.median(x)) for x in common_errors]},
        "independent":[{"period":m[1],"phase":m[2],
                        "train_median_absolute_residual":m[0],
                        "heldout_median_absolute_residual":float(np.median(x))}
                       for m,x in zip(ind,ind_errors)],
        "heldout_common_minus_independent_median_residual":float(
            np.median(np.concatenate(common_errors))-np.median(np.concatenate(ind_errors))),
        "interpretation":"Lower residual is better; independent model has more parameters. No probability of shared emitter is inferred."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--clusters",type=int,nargs="+",default=[12,32])
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--period-min",type=float,default=25.)
    p.add_argument("--period-max",type=float,default=1000.)
    p.add_argument("--period-count",type=int,default=240)
    p.add_argument("--phase-bins",type=int,default=64)
    a=p.parse_args()
    if (a.period_min<=0 or a.period_max<=a.period_min or a.period_count<2
        or a.phase_bins<4 or len(a.clusters)<2 or len(set(a.clusters))!=len(a.clusters)):
        p.error("Invalid search parameters or clusters")
    data,truth,names,total=load_sample(a.file,a.limit)
    pred=DBSCAN(eps=a.eps,min_samples=a.min_samples).fit_predict(build_features(data,names))
    toa=data[:,names.index("ToA")]
    groups=[]
    composition=[]
    for cl in a.clusters:
        mask=pred==cl
        t=np.sort(toa[mask])
        if len(t)<20:
            p.error(f"Cluster {cl} has fewer than 20 pulses")
        groups.append(t)
        entry={"cluster":cl,"pdws":len(t)}
        if truth is not None:
            ids,counts=np.unique(truth[mask],return_counts=True)
            entry["truth_composition_diagnostic_only"]=[
                {"emitter":str(i),"pdws":int(n)}
                for i,n in sorted(zip(ids,counts),key=lambda z:-z[1])]
        composition.append(entry)
    periods=np.linspace(a.period_min,a.period_max,a.period_count)
    report={"file":a.file,"limit":a.limit,"clusters":composition,
            "period_search":{"min":a.period_min,"max":a.period_max,
                             "count":a.period_count,"phase_bins":a.phase_bins},
            "coherence":evaluate(groups,periods,a.phase_bins),
            "limitations":[
                "A shared lattice can occur for independent phase-locked or coincident emitters.",
                "Independent fits have more parameters; no calibrated model comparison is made.",
                "Fixed-PRI model may fail for staggered, jittered or agile PRI.",
                "No explicit missed-pulse or receiver collision likelihood is fitted.",
                "Only selected cluster IDs are tested; not an unsupervised emitter discovery result."]}
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
