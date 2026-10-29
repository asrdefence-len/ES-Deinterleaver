"""Scatter diagnostics for RF blobs in one batch AOA population.

Usage: python -u plot_rf_scatter.py turing_data/config_101.h5
Truth labels are used for plotting only, never for grouping.
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from compare_aoa_batch import batch_aoa_populations
from compare_density_cores import group_window


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window",type=int,default=0)
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--aoa-pop",type=int,default=15)
    p.add_argument("--output",default="rf_scatter_w0_aoa15.png")
    a=p.parse_args()
    with h5py.File(a.file) as h:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in h["metadata/feature_names"][:]]
        start=a.window*a.window_size
        x=np.asarray(h["data"][start:start+a.window_size],float)
        truth=np.asarray(h["labels"][start:start+a.window_size]).reshape(-1)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    if not 0<=a.aoa_pop<npop:
        raise ValueError(f"AOA population {a.aoa_pop} outside 0..{npop-1}")
    pred=group_window(x,names,pops,npop,8,8,0.20,0.02)
    ix=np.flatnonzero(pops==a.aoa_pop)
    print(f"AOA population {a.aoa_pop}: {len(ix)} PDWs")
    print("Truth contributors:",[(int(t),int(np.sum(truth[ix]==t))) for t in np.unique(truth[ix])])
    fields=[("ToA","Time of arrival"),("PulseWidth","Pulse width"),("AoA","Angle of arrival")]
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout="constrained")
    true_ids=np.unique(truth[ix])
    groups=np.unique(pred[ix])
    for row,(field,ylabel) in enumerate(fields):
        xx=x[ix,names.index("Frequency")]
        yy=x[ix,names.index(field)]
        ax=axes[row,0]
        for t in true_ids:
            mask=truth[ix]==t
            ax.scatter(xx[mask],yy[mask],s=9,alpha=.65,label=f"Emitter {t}")
        ax.set(title=f"Truth (evaluation only): RF vs {field}",xlabel="RF (dataset units)",ylabel=ylabel)
        if row==0: ax.legend(loc="best",fontsize=8)
        ax=axes[row,1]
        for g in groups:
            mask=pred[ix]==g
            ax.scatter(xx[mask],yy[mask],s=9,alpha=.65,
                       label="Unassigned" if g==-1 else f"Blob {g}",
                       color="0.65" if g==-1 else None)
        ax.set(title=f"Density-core assignments: RF vs {field}",xlabel="RF (dataset units)",ylabel=ylabel)
        if row==0 and len(groups)<=12: ax.legend(loc="best",fontsize=7)
    fig.suptitle(f"Window {a.window}, AOA population {a.aoa_pop} ({len(ix)} PDWs)")
    fig.savefig(a.output,dpi=170)
    print("Saved",a.output)
    plt.show()


if __name__=="__main__":
    main()
