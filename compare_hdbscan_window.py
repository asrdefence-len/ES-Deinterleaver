"""Compare baseline DBSCAN with experimental HDBSCAN on one Turing window.

Uses identical blind RF/PW/AOA features. Truth is used only for scoring and
optional diagnostic point outlines, never clustering.
Requires scikit-learn with sklearn.cluster.HDBSCAN (>=1.3).
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from candidate_grouping import (RF_FRACTION, AOA_SCALE_DEG, PW_ALPHA,
                                DBSCAN_EPS, DBSCAN_MIN_SAMPLES, robust_scale)


def features(x,names):
    rf=x[:,names.index("Frequency")]
    pw=x[:,names.index("PulseWidth")]
    a=np.deg2rad(x[:,names.index("AoA")])
    if np.any(rf<=0):raise ValueError("RF must be positive")
    ps=robust_scale(pw)
    return np.column_stack((np.sin(a)/np.deg2rad(AOA_SCALE_DEG),
                            np.cos(a)/np.deg2rad(AOA_SCALE_DEG),
                            np.log(rf)/RF_FRACTION,
                            np.log1p(PW_ALPHA*pw/ps)/PW_ALPHA))


def evaluate(labels,truth):
    valid=labels>=0
    grouped=int(valid.sum())
    wrong=0
    for gid in np.unique(labels[valid]):
        counts=Counter(truth[labels==gid].tolist())
        wrong+=sum(counts.values())-counts.most_common(1)[0][1]
    return dict(groups=len(np.unique(labels[valid])),grouped=grouped,
                noise=len(labels)-grouped,cross=wrong,
                purity=100*(grouped-wrong)/grouped if grouped else 0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path",nargs="?",default="turing_data/config_101.h5")
    ap.add_argument("--window",type=int,default=26)
    ap.add_argument("--window-size",type=int,default=10000)
    ap.add_argument("--min-cluster-size",type=int,default=8)
    ap.add_argument("--min-samples",type=int,default=8)
    ap.add_argument("--output",default=None)
    args=ap.parse_args()
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise SystemExit("HDBSCAN unavailable in scikit-learn; upgrade: python -m pip install -U scikit-learn") from exc
    start=args.window*args.window_size
    with h5py.File(args.path) as f:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
        x=np.asarray(f["data"][start:start+args.window_size],dtype=float)
        truth=np.asarray(f["labels"][start:start+len(x)]).reshape(-1)
    if len(x)<8:raise SystemExit("Window too small or missing")
    z=features(x,names)
    t=time.perf_counter()
    baseline=DBSCAN(eps=DBSCAN_EPS,min_samples=DBSCAN_MIN_SAMPLES).fit_predict(z)
    db_time=time.perf_counter()-t
    # sklearn HDBSCAN counts the point itself in min_samples, unlike contrib hdbscan.
    t=time.perf_counter()
    experimental=HDBSCAN(min_cluster_size=args.min_cluster_size,
                          min_samples=args.min_samples).fit_predict(z)
    hd_time=time.perf_counter()-t
    print(f"Window {args.window}, rows {start}:{start+len(x)}; {len(x)} PDWs")
    for name,labels,elapsed in (("DBSCAN",baseline,db_time),("HDBSCAN",experimental,hd_time)):
        result=evaluate(labels,truth)
        print(f"{name}: {result}, elapsed={elapsed:.3f}s")
        print("  Largest groups (size, truth makeup):")
        sizes=Counter(labels[labels>=0].tolist())
        for gid,size in sizes.most_common(8):
            print(f"    group {gid}: n={size} emitters={Counter(truth[labels==gid].tolist()).most_common(5)}")
    rf=x[:,names.index("Frequency")]
    pw=x[:,names.index("PulseWidth")]
    aoa=x[:,names.index("AoA")]
    fig,axes=plt.subplots(2,2,figsize=(15,11),layout="constrained")
    for col,(method,labels) in enumerate((("DBSCAN",baseline),("HDBSCAN",experimental))):
        for row,(xx,yy,xlabel,ylabel) in enumerate(((aoa,pw,"AOA (degrees)","Pulse width (dataset units)"),
                                                    (rf,pw,"Frequency (dataset units)","Pulse width (dataset units)"))):
            ax=axes[row,col]
            gids=[int(g) for g,_ in Counter(labels[labels>=0].tolist()).most_common(15)]
            # Plot all groups faintly; highlight 15 largest. Colour is predicted cluster, NOT truth.
            ax.scatter(xx,yy,s=3,c="lightgray",alpha=.15,rasterized=True)
            palette=plt.get_cmap("tab20")
            for i,g in enumerate(gids):
                mask=labels==g
                ax.scatter(xx[mask],yy[mask],s=7,alpha=.65,
                           color=palette(i%20),label=f"G{g} ({mask.sum()})",rasterized=True)
            ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
            ax.set_title(f"{method}: {xlabel} vs {ylabel}")
            ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=7,ncol=3,loc="best")
    fig.suptitle(f"{Path(args.path).name} window {args.window}: predicted groups (truth never used for clustering)")
    output=args.output or f"turing_config101_window{args.window}_dbscan_vs_hdbscan.png"
    fig.savefig(output,dpi=170)
    plt.close(fig)
    print(f"Saved {output}")


if __name__=="__main__":
    main()
