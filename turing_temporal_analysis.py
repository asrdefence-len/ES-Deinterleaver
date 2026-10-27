"""Plot Turing emitter fragmentation and cluster mixing using truth for diagnosis only.

Recreates the label-blind DBSCAN baseline, then uses labels to select diagnostic
views. The plots are NOT evidence available to an operational deinterleaver.

Run:
  python turing_temporal_analysis.py turing_data/config_100.h5 --limit 10000
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample, build_features


def analyse(path, limit=10000, eps=0.25, min_samples=8, outdir="turing_plots"):
    data, labels, names, total = load_sample(path, limit)
    if labels is None:
        raise ValueError("Ground truth required for diagnostic plots")
    ix={name:i for i,name in enumerate(names)}
    required={"ToA","Frequency","PulseWidth"}
    if not required.issubset(ix):
        raise ValueError(f"Missing features: {required-set(ix)}")
    pred=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(build_features(data,names))
    out=Path(outdir)
    out.mkdir(parents=True,exist_ok=True)
    toa=data[:,ix["ToA"]]
    order=np.argsort(toa,kind="stable")

    def emitter_plot(emitter):
        mask=labels==emitter
        rows=np.flatnonzero(mask)
        if not len(rows):
            print(f"Emitter {emitter} absent from sample")
            return
        rows=rows[np.argsort(toa[rows])]
        t=toa[rows]-toa[rows[0]]
        fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=False)
        for cl in np.unique(pred[rows]):
            selected=pred[rows]==cl
            ax[0].scatter(t[selected],data[rows[selected],ix["Frequency"]],
                          s=9,label=f"cluster {cl}",alpha=.75)
            ax[1].scatter(t[selected],data[rows[selected],ix["PulseWidth"]],
                          s=9,alpha=.75)
        ax[0].set_ylabel("Frequency (dataset units)")
        ax[1].set_ylabel("Pulse width (dataset units)")
        ax[0].legend(fontsize=8,ncol=4)
        dt=np.diff(toa[rows])
        ax[2].scatter(t[1:],dt,s=9,alpha=.7)
        ax[2].set_ylabel("Successive observed-pulse interval (dataset units)")
        ax[2].set_xlabel("Relative TOA (dataset units)")
        ax[1].set_xlabel("Relative TOA (dataset units)")
        ax[0].set_xlabel("Relative TOA (dataset units)")
        ax[2].set_yscale("symlog",linthresh=max(1e-9,float(np.median(dt[dt>0])) if np.any(dt>0) else 1))
        fig.suptitle(f"Truth emitter {emitter}: {len(rows)} PDWs, {len(np.unique(pred[rows]))} predicted groups")
        fig.tight_layout()
        file=out/f"emitter_{emitter}.png"
        fig.savefig(file,dpi=150)
        plt.close(fig)
        print(f"Saved {file}")

    for emitter in (80,35):
        emitter_plot(emitter)

    cluster=13
    rows=np.flatnonzero(pred==cluster)
    if len(rows):
        fig,ax=plt.subplots(3,1,figsize=(12,9))
        for emitter in np.unique(labels[rows]):
            erows=rows[labels[rows]==emitter]
            erows=erows[np.argsort(toa[erows])]
            t=toa[erows]-toa[rows].min()
            ax[0].scatter(t,data[erows,ix["Frequency"]],s=9,alpha=.7,label=f"truth {emitter}")
            ax[1].scatter(t,data[erows,ix["PulseWidth"]],s=9,alpha=.7)
            if len(erows)>1:
                ax[2].scatter(t[1:],np.diff(toa[erows]),s=9,alpha=.7)
        ax[0].legend(fontsize=8,ncol=4)
        for a in ax:
            a.set_xlabel("Relative TOA (dataset units)")
        ax[0].set_ylabel("Frequency (dataset units)")
        ax[1].set_ylabel("Pulse width (dataset units)")
        ax[2].set_ylabel("Same-truth successive interval (dataset units)")
        ax[2].set_yscale("symlog")
        fig.suptitle(f"Predicted cluster {cluster}: {len(rows)} PDWs, truth-coloured (diagnostic only)")
        fig.tight_layout()
        file=out/f"cluster_{cluster}_mixed.png"
        fig.savefig(file,dpi=150)
        plt.close(fig)
        print(f"Saved {file}")
    else:
        print("Cluster 13 absent; cluster numbering depends on settings and sample")
    print(f"Analysed {len(data)} of {total} PDWs; labels used only for plotting")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--eps",type=float,default=.25)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--outdir",default="turing_plots")
    a=p.parse_args()
    analyse(a.file,a.limit,a.eps,a.min_samples,a.outdir)


if __name__=="__main__":
    main()
