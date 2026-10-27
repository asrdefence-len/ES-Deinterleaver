"""Compare direct B/C grouping with D: AOA + RF + log(PW).

Run: python turing_log_pw_comparison.py turing_data/config_100.h5
Truth labels are used for evaluation only.
"""
import argparse
import csv
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample
from turing_grouping_comparison import groups,robust_scale


def evaluate(labels,truth):
    valid=labels>=0
    leaders={}
    for g in np.unique(labels[valid]):
        counts=Counter(map(int,truth[labels==g]))
        leaders[int(g)]=counts.most_common(1)[0][0]
    rows={}
    for e in sorted(map(int,np.unique(truth))):
        mask=truth==e
        grouped=int(np.count_nonzero(mask&valid))
        own=sum(int(np.count_nonzero(mask&(labels==g))) for g,leader in leaders.items() if leader==e)
        rows[e]={"total":int(mask.sum()),"grouped":grouped,"own":own,
                 "other":grouped-own,"noise":int(mask.sum())-grouped,
                 "dominant_groups":sum(leader==e for leader in leaders.values())}
    correct=sum(r["own"] for r in rows.values())
    n=int(valid.sum())
    return {"groups":len(leaders),"grouped":n,"coverage":n/len(labels),
            "purity":correct/n if n else None,"cross_emitter":n-correct,
            "emitters_represented":sum(r["grouped"]>0 for r in rows.values()),
            "emitters_with_dominant_group":sum(r["dominant_groups"]>0 for r in rows.values())},rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--output",default="turing_log_pw_comparison.csv")
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Requires truth labels for scoring")
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    if np.any(pw<=0):
        p.error("PW must be strictly positive for log(PW) comparison")
    theta=np.deg2rad(data[:,names.index("AoA")])
    rf_scale=robust_scale(rf)
    pw_scale=robust_scale(pw)
    log_pw=np.log(pw)
    log_scale=robust_scale(log_pw)
    z=np.column_stack((np.sin(theta)/np.deg2rad(a.aoa_deg),
                       np.cos(theta)/np.deg2rad(a.aoa_deg),
                       (rf-np.median(rf))/rf_scale,
                       (log_pw-np.median(log_pw))/log_scale))
    predictions={
        "B":groups(data,names,"B",a.aoa_deg,a.min_samples,rf_scale,pw_scale),
        "C":groups(data,names,"C",a.aoa_deg,a.min_samples,rf_scale,pw_scale),
        "D":DBSCAN(eps=1.,min_samples=a.min_samples).fit_predict(z)}
    results={mode:evaluate(labels,truth) for mode,labels in predictions.items()}
    print(f"AOA scale={a.aoa_deg} deg; RF scale={rf_scale:.6g}; PW scale={pw_scale:.6g}; logPW scale={log_scale:.6g}; min_samples={a.min_samples}")
    print("B=AOA+RF; C=AOA+RF+PW; D=AOA+RF+log(PW); all direct grouping, no timing filter.")
    for mode in ("B","C","D"):
        print(mode,results[mode][0])
    print("\nEmitter  B(grouped/own/other) C(grouped/own/other) D(grouped/own/other)")
    for e in sorted(results["D"][1]):
        b,c,d=(results[m][1][e] for m in ("B","C","D"))
        if e in (18,22,30,33,54,79) or d!=c:
            fmt=lambda r:f"{r['grouped']}/{r['own']}/{r['other']}"
            print(f"{e:>7} {fmt(b):>19} {fmt(c):>19} {fmt(d):>19}")
    with open(a.output,"w",newline="") as f:
        fields=["mode","emitter"]+list(next(iter(results["D"][1].values())).keys())
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for mode in ("B","C","D"):
            for e,row in results[mode][1].items():
                writer.writerow({"mode":mode,"emitter":e,**row})
    print("\nSaved",a.output)
    print("Truth used only after clustering; purity measures dominant-emitter agreement, not timing coherence.")

if __name__=="__main__":
    main()
