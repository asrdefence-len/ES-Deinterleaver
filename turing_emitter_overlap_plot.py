"""Plot a known difficult mixed DBSCAN group, with truth used for colouring only.

Usage: python turing_emitter_overlap_plot.py turing_data/config_100.h5 --window 26
"""
import argparse
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from turing_grouping_comparison import robust_scale


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window",type=int,default=26)
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--emitters",type=int,nargs="+",default=[46,72,80])
    p.add_argument("--output",default=None)
    a=p.parse_args()
    start=a.window*a.window_size
    with h5py.File(a.file,"r") as f:
        stop=min(start+a.window_size,len(f["data"]))
        data=np.asarray(f["data"][start:stop],dtype=float)
        truth=np.asarray(f["labels"][start:stop]).reshape(-1)
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    aoa=data[:,names.index("AoA")]
    alpha=.05
    theta=np.deg2rad(aoa)
    sr=robust_scale(rf)
    sp=robust_scale(pw)
    z=np.column_stack((np.sin(theta)/np.deg2rad(2),
                       np.cos(theta)/np.deg2rad(2),
                       (rf-np.median(rf))/sr,
                       np.log1p(alpha*pw/sp)/alpha))
    labels=DBSCAN(eps=1.,min_samples=8).fit_predict(z)
    targets=np.isin(truth,a.emitters)
    candidates=Counter(labels[targets & (labels>=0)].tolist())
    if not candidates:
        raise RuntimeError("No selected emitter pulses grouped in this window")
    # Select group containing the most selected-emitter observations.
    # Prefer the group containing both emitters 46 and 80.
    # Otherwise favour groups representing more requested emitters.
    def rank(g):
        members=truth[labels==g]
        represented=sum(np.any(members==e) for e in a.emitters)
        has_pair=int(np.any(members==46) and np.any(members==80))
        return (has_pair,represented,candidates[g])
    gid=max(candidates,key=rank)
    in_group=labels==gid
    print(f"Window {a.window}: rows {start}:{stop}; selected DBSCAN group {gid}")
    print("Group composition:",dict(Counter(truth[in_group].tolist()).most_common()))
    print("Selected emitter counts (total / in group):")
    for e in a.emitters:
        print(f"  {e}: {(truth==e).sum()} / {((truth==e)&in_group).sum()}")
    out=a.output or f"turing_overlap_window{a.window}_group{gid}.png"
    fig,axs=plt.subplots(1,3,figsize=(17,5.2),constrained_layout=True)
    pairs=[(rf,aoa,"Frequency (dataset units)","AOA (degrees)"),
           (pw,aoa,"Pulse width (dataset units)","AOA (degrees)"),
           (rf,pw,"Frequency (dataset units)","Pulse width (dataset units)")]
    colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Plot other members first, so selected emitter populations remain visible.
    other=in_group & ~np.isin(truth,a.emitters)
    for ax,(xx,yy,xlabel,ylabel) in zip(axs,pairs):
        if other.any():
            ax.scatter(xx[other],yy[other],s=12,c="0.65",alpha=.4,label="Other group members")
        for j,e in enumerate(a.emitters):
            sel=in_group & (truth==e)
            if sel.any():
                ax.scatter(xx[sel],yy[sel],s=9,alpha=.5,color=colors[j%len(colors)],label=f"Emitter {e} (n={sel.sum()})")
        ax.set(xlabel=xlabel,ylabel=ylabel)
        ax.grid(alpha=.2)
    axs[0].legend(loc="best",fontsize=8)
    fig.suptitle(f"Adaptive 5% DBSCAN group {gid} | window {a.window} | truth colours for diagnosis only")
    fig.savefig(out,dpi=170)
    plt.close(fig)
    print(f"Saved {Path(out).resolve()}")
    print("Truth labels are used only for diagnostics, not grouping.")


if __name__=="__main__":
    main()
