"""PRI hypothesis diagnostics for Turing pulse groups (no labels in inference).

Plots inter-arrival histograms, PRI candidate scores, and shared-lattice phase
concentration. Controls use a different cluster pair; labels only reported
after inference. This is exploratory, not calibrated emitter association.

python turing_pri_hypotheses.py turing_data/config_100.h5 --clusters 12 32 --control 12 16
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def phase_concentration(t, pri):
    angles=(2*np.pi/pri)*np.remainder(t-t[0],pri)
    return float(np.abs(np.mean(np.exp(1j*angles))))


def interval_score(t, pri, tolerance):
    d=np.diff(t)
    if not len(d):
        return float("nan")
    # A valid candidate must explain intervals as positive integer PRI multiples.
    n=np.maximum(1,np.rint(d/pri))
    errors=np.abs(d-n*pri)
    return float(np.mean(errors<=tolerance))


def candidate_profile(t, pri_grid, tolerance):
    return np.array([interval_score(t,p,tolerance) for p in pri_grid])


def analyse_pair(groups, pri_grid, tolerance):
    a,b=groups
    combined=np.sort(np.concatenate((a,b)))
    sa=candidate_profile(a,pri_grid,tolerance)
    sb=candidate_profile(b,pri_grid,tolerance)
    sc=candidate_profile(combined,pri_grid,tolerance)
    # This descriptive score rewards consistency in each group and combined
    # intervals. It is NOT a likelihood ratio and may favour harmonics.
    score=(sa+sb+sc)/3
    top=np.argsort(score)[-8:][::-1]
    return {"candidate_pris":[{"pri":float(pri_grid[i]),
                               "score":float(score[i]),
                               "group_interval_fractions":[float(sa[i]),float(sb[i])],
                               "combined_interval_fraction":float(sc[i]),
                               "phase_concentration":[phase_concentration(a,pri_grid[i]),
                                                      phase_concentration(b,pri_grid[i]),
                                                      phase_concentration(combined,pri_grid[i])]}
                              for i in top]}, (sa,sb,sc,score)


def plot_pair(groups, ids, grid, profiles, out):
    a,b=groups
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for t,label in zip(groups,ids):
        intervals=np.diff(t)
        axes[0].hist(intervals,bins=90,alpha=.55,label=f"cluster {label}")
    axes[0].set(xlabel="Observed successive TOA interval (dataset units)",
                ylabel="Count",title="Observed intervals (not necessarily PRI)")
    axes[0].legend()
    for y,label in zip(profiles,["group A","group B","combined","mean score"]):
        axes[1].plot(grid,y,label=label)
    axes[1].set(xlabel="Candidate PRI (dataset units)",ylabel="Fraction of intervals explained",
                title="Integer-multiple PRI consistency (harmonics possible)",ylim=(0,1.02))
    axes[1].legend()
    fig.savefig(out,dpi=150)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--clusters",type=int,nargs=2,default=[12,32])
    p.add_argument("--control",type=int,nargs=2,default=[12,16])
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--pri-min",type=float,default=25)
    p.add_argument("--pri-max",type=float,default=1600)
    p.add_argument("--pri-step",type=float,default=1)
    p.add_argument("--tolerance",type=float,default=2)
    p.add_argument("--outdir",default="turing_plots")
    args=p.parse_args()
    if args.pri_min<=0 or args.pri_max<=args.pri_min or args.pri_step<=0 or args.tolerance<=0:
        p.error("Invalid PRI search parameters")
    data,truth,names,total=load_sample(args.file,args.limit)
    pred=DBSCAN(eps=args.eps,min_samples=args.min_samples).fit_predict(build_features(data,names))
    toa=data[:,names.index("ToA")]
    grid=np.arange(args.pri_min,args.pri_max+args.pri_step/2,args.pri_step)
    outdir=Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    result={"file":args.file,"sample_pdws":len(data),"pri_search":{"min":args.pri_min,
             "max":args.pri_max,"step":args.pri_step,"tolerance":args.tolerance},
            "experiments":[],"caveats":[
                "Truth is used only to report post-inference cluster composition.",
                "Interval multiple scoring favours PRI harmonics; high scores alone do not establish a true PRI.",
                "Shared PRI does not imply shared emitter. Compare target and negative control.",
                "Constant-PRI assumption may fail for staggered or jittered emitters.",
                "No detection probability, collision model, or calibrated hypothesis odds."]}
    for name,ids in [("candidate",args.clusters),("negative_control",args.control)]:
        groups=[]
        compositions=[]
        for cl in ids:
            mask=pred==cl
            t=np.sort(toa[mask])
            if len(t)<20:
                p.error(f"Cluster {cl} has fewer than 20 pulses")
            groups.append(t)
            item={"cluster":cl,"pdws":int(len(t))}
            if truth is not None:
                labels,counts=np.unique(truth[mask],return_counts=True)
                item["truth_composition_diagnostic_only"]=[
                    {"emitter":str(k),"pdws":int(v)}
                    for k,v in sorted(zip(labels,counts),key=lambda z:-z[1])]
            compositions.append(item)
        stats,profiles=analyse_pair(groups,grid,args.tolerance)
        filename=outdir/f"pri_{name}_{ids[0]}_{ids[1]}.png"
        plot_pair(groups,ids,grid,profiles,filename)
        result["experiments"].append({"name":name,"clusters":compositions,
                                       "plot":str(filename),**stats})
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
