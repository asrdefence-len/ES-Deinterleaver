"""Diagnostic only: examine RF concentrations before changing grouping.

Usage: python -u diagnose_rf_density.py turing_data/config_101.h5
Truth labels select diagnostic AOA populations and colour evaluation overlays ONLY.
No clustering or membership assignments are made.
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from compare_aoa_batch import batch_aoa_populations

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--window-size", type=int, default=10000)
    ap.add_argument("--emitters", type=int, nargs="+", default=[9,46,20])
    ap.add_argument("--bins", type=int, default=2048)
    args = ap.parse_args()
    with h5py.File(args.file) as h:
        names = [v.decode() if isinstance(v,bytes) else str(v) for v in h["metadata/feature_names"][:]]
        st = args.window*args.window_size
        en = min(st+args.window_size, len(h["data"]))
        x = np.asarray(h["data"][st:en],float)
        truth = np.asarray(h["labels"][st:en]).reshape(-1)
    rf = x[:,names.index("Frequency")]
    aoa = x[:,names.index("AoA")]
    pops,npop = batch_aoa_populations(aoa)
    print(f"Window {args.window}: {len(x)} PDWs, {npop} AOA populations")
    done=set()
    for eid in args.emitters:
        indices=np.flatnonzero(truth==eid)
        if not len(indices):
            print(f"Emitter {eid}: absent")
            continue
        for pop in np.unique(pops[indices]):
            if int(pop) in done:
                print(f"Emitter {eid}: shares already plotted AOA population {pop}")
                continue
            done.add(int(pop))
            ii=np.flatnonzero(pops==pop)
            v=rf[ii]
            bins=min(args.bins,max(128, int(len(ii)*0.75)))
            edges=np.linspace(v.min(),v.max()+1e-9,bins+1)
            counts,_=np.histogram(v,bins=edges)
            centers=(edges[:-1]+edges[1:])/2
            width=edges[1]-edges[0]
            fig,ax=plt.subplots(3,1,figsize=(13,10),layout="constrained")
            ax[0].hist(v,bins=edges,color="0.65",label="All PDWs in AOA population")
            ax[0].set_title(f"Window {args.window}, AOA population {pop}: {len(ii)} PDWs; selected emitter {eid}")
            ax[0].set_ylabel("PDWs / bin")
            ax[0].legend()
            for sigma in (1,3,8,20):
                sm=gaussian_filter1d(counts.astype(float),sigma=sigma)
                peaks,_=find_peaks(sm,prominence=max(1,sm.max()*0.02))
                ax[1].plot(centers,sm,label=f"smoothing {sigma} bins ({sigma*width:.3g} RF units), {len(peaks)} peaks")
            ax[1].set_ylabel("Smoothed counts")
            ax[1].legend(fontsize=8)
            sorted_v=np.sort(v)
            gaps=np.diff(sorted_v)
            if len(gaps):
                ax[2].plot((sorted_v[:-1]+sorted_v[1:])/2,gaps,".",markersize=2,label="Adjacent sorted RF gap")
                ax[2].set_yscale("symlog",linthresh=max(1e-6,np.percentile(gaps,50)))
            ax[2].set_ylabel("Sorted RF gap")
            ax[2].set_xlabel("RF (dataset units)")
            ax[2].legend()
            out=f"rf_density_w{args.window}_aoa{pop}.png"
            fig.savefig(out,dpi=150)
            plt.close(fig)
            print(f"  AOA pop {pop}: {len(ii)} PDWs, RF {v.min():.3f}..{v.max():.3f}; saved {out}")
            ids,ns=np.unique(truth[ii],return_counts=True)
            print("  Top truth contributors (evaluation only):",list(zip(ids[np.argsort(ns)[-8:][::-1]].tolist(),np.sort(ns)[-8:][::-1].tolist())))
            print("  Sorted RF gap quantiles:",np.percentile(gaps,[50,90,99,99.9]) if len(gaps) else [])
    print("No detector changed. Inspect these plots before choosing density scale or membership threshold.")

if __name__=="__main__":
    main()
