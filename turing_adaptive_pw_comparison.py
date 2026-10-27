"""Controlled adaptive-PW DBSCAN experiment against C and D baselines.

Run: python turing_adaptive_pw_comparison.py turing_data/config_100.h5
Truth is used for evaluation only. Parameters are fixed in advance.
"""
import argparse
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale


def evaluate(labels, truth):
    valid=labels>=0
    leaders={}
    for g in np.unique(labels[valid]):
        counts=Counter(map(int,truth[labels==g]))
        leaders[int(g)]=counts.most_common(1)[0][0]
    n=int(valid.sum())
    correct=sum(int(np.count_nonzero((labels==g)&(truth==e))) for g,e in leaders.items())
    result={"groups":len(leaders),"grouped":n,"purity":correct/n if n else 0,
            "cross":n-correct,"dominant_emitters":len(set(leaders.values()))}
    details={}
    for e in np.unique(truth):
        e=int(e)
        m=truth==e
        own=sum(int(np.count_nonzero(m&(labels==g))) for g,leader in leaders.items() if leader==e)
        grouped=int(np.count_nonzero(m&valid))
        details[e]=(int(m.sum()),grouped,own,grouped-own)
    return result,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Truth labels needed for evaluation")
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    if np.any(pw<=0):
        p.error("PW must be positive")
    theta=np.deg2rad(data[:,names.index("AoA")])
    srf=robust_scale(rf)
    spw=robust_scale(pw)
    common=np.column_stack((np.sin(theta)/np.deg2rad(a.aoa_deg),
                            np.cos(theta)/np.deg2rad(a.aoa_deg),
                            (rf-np.median(rf))/srf))
    def cluster(pw_feature):
        return DBSCAN(eps=1.,min_samples=a.min_samples).fit_predict(
            np.column_stack((common,pw_feature)))
    results={}
    results["C absolute"]=evaluate(groups(data,names,"C",a.aoa_deg,a.min_samples,srf,spw),truth)
    log_pw=np.log(pw)
    results["D log"]=evaluate(cluster((log_pw-np.median(log_pw))/robust_scale(log_pw)),truth)
    # Smooth adaptive tolerance: derivative dz/dPW = 1/(S0+alpha*PW).
    # For PW << S0/alpha, matches C; for PW >> S0/alpha, approaches relative scaling.
    # Unlike a hard pairwise scale, this is a valid Euclidean feature transform for DBSCAN.
    for alpha in (0.01,0.02,0.05):
        transformed=np.log1p(alpha*pw/spw)/alpha
        results[f"Adaptive alpha={alpha:g}"]=evaluate(cluster(transformed),truth)
    print(f"AOA={a.aoa_deg}deg, RF scale={srf:.6g}, base PW scale={spw:.6g}, min_samples={a.min_samples}")
    print("Adaptive feature: log(1+alpha*PW/S0)/alpha, with S0=global PW scale.")
    print("PW local tolerance S0+alpha*PW. Fixed alphas; no truth-based parameter selection.")
    print("\nMETHOD                         GROUPS GROUPED PURITY% CROSS DOMINANT_EMITTERS")
    for label,(stats,_) in results.items():
        print(f"{label:30s} {stats['groups']:6d} {stats['grouped']:7d} {100*stats['purity']:7.3f} {stats['cross']:5d} {stats['dominant_emitters']:8d}")
    print("\nSelected emitters: total / grouped / in own-dominant groups / in other-dominant groups")
    for e in (18,22,30,32,33,35,44,51,54,79,81):
        print(f"Emitter {e}:")
        for label,(_,detail) in results.items():
            if e in detail:
                print(f"  {label:30s} {detail[e]}")
    print("\nNo changes to production grouping. Truth used only for scoring.")


if __name__=="__main__":
    main()
