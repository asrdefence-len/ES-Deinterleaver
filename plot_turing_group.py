"""Plot Turing candidate grouping failures with truth used only for colouring."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from candidate_grouping import group_pdws


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("path",nargs="?",default="turing_data/config_101.h5")
    p.add_argument("--window",type=int,default=26)
    p.add_argument("--group",type=int,default=2)
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    start=a.window*a.window_size
    with h5py.File(a.path) as f:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
        x=np.asarray(f["data"][start:start+a.window_size],dtype=float)
        truth=np.asarray(f["labels"][start:start+len(x)]).reshape(-1)
    if not len(x):raise ValueError("Empty window")
    groups=group_pdws(x,names)
    selected=groups==a.group
    if not selected.any():raise ValueError(f"Group {a.group} absent in window {a.window}")
    rf=x[:,names.index("Frequency")]
    pw=x[:,names.index("PulseWidth")]
    aoa=x[:,names.index("AoA")]
    toa=x[:,names.index("ToA")]
    ids=np.unique(truth[selected])
    print(f"Window {a.window}, group {a.group}: {selected.sum()} PDWs; emitter labels {ids.tolist()}")
    for emitter in ids:
        print(f"  emitter {emitter}: {np.sum(selected & (truth==emitter))} PDWs")
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout="constrained")
    fields=[(rf,pw,"Frequency (dataset units)","Pulse width (dataset units)"),
            (aoa,pw,"AOA (degrees)","Pulse width (dataset units)"),
            (rf,aoa,"Frequency (dataset units)","AOA (degrees)"),
            (toa,rf,"Time of arrival (dataset units)","Frequency (dataset units)")]
    for ax,(xx,yy,xlab,ylab) in zip(axs.flat,fields):
        for emitter in ids:
            mask=selected & (truth==emitter)
            ax.scatter(xx[mask],yy[mask],s=7,alpha=.45,label=f"Emitter {emitter} (n={mask.sum()})",rasterized=True)
        ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.grid(alpha=.2)
    axs[0,0].legend(markerscale=2,fontsize=9)
    fig.suptitle(f"{Path(a.path).name} — window {a.window}, DBSCAN group {a.group} (truth-coloured diagnostic)",fontsize=14)
    output=a.output or f"turing_config101_window{a.window}_group{a.group}.png"
    fig.savefig(output,dpi=180)
    plt.close(fig)
    print(f"Saved {output}")
    print("Truth labels are for visual diagnosis only; grouping is blind.")


if __name__=="__main__":
    main()
