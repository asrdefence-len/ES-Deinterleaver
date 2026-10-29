"""Diagnose cross-emitter mixing in multiscale RF blobs (truth for evaluation only).

Run:
 python -u diagnose_rf_contamination.py turing_data/config_101.h5 --top 15

Outputs CSV of all mixed blobs, a summary of minority-emitter contributions,
and RF/AOA scatter plots of the worst mixed blobs. Each window is independent.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


def pct_overlap(a,b,bins=80):
    """Empirical RF histogram overlap on a common range (diagnostic only)."""
    lo=min(float(np.min(a)),float(np.min(b)))
    hi=max(float(np.max(a)),float(np.max(b)))
    if hi<=lo:
        return 1.0
    edges=np.linspace(lo,hi,bins+1)
    ha=np.histogram(a,bins=edges)[0].astype(float)
    hb=np.histogram(b,bins=edges)[0].astype(float)
    return float(np.minimum(ha/max(ha.sum(),1),hb/max(hb.sum(),1)).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--top",type=int,default=15)
    p.add_argument("--plots",type=int,default=5)
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--windows",type=int,default=0,help="0 means all")
    p.add_argument("--outdir",default="rf_contamination_diagnostics")
    a=p.parse_args()
    out=Path(a.outdir)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    contributions=Counter()
    n_total=0
    n_mixed_pdws=0
    n_minority=0
    saved=[]
    with h5py.File(a.file) as h:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in h["metadata/feature_names"][:]]
        rf_col=names.index("Frequency")
        aoa_col=names.index("AoA")
        for wi,start in enumerate(range(0,len(h["data"]),a.window_size)):
            if a.windows and wi>=a.windows:
                break
            x=np.asarray(h["data"][start:start+a.window_size],float)
            truth=np.asarray(h["labels"][start:start+a.window_size]).reshape(-1)
            if len(x)<8: continue
            n_total+=len(x)
            pops,npop=batch_aoa_populations(x[:,aoa_col])
            pred=group_window(x,names,pops,npop,8,.20,6.0,8)
            for g in np.unique(pred[pred>=0]):
                ii=np.flatnonzero(pred==g)
                ct=Counter(truth[ii].tolist())
                if len(ct)<2:
                    continue
                majority,majority_n=ct.most_common(1)[0]
                minority=len(ii)-majority_n
                n_minority+=minority
                n_mixed_pdws+=len(ii)
                for eid,num in ct.items():
                    if eid!=majority:
                        contributions[(int(majority),int(eid))]+=int(num)
                rf=x[ii,rf_col]
                aoa=x[ii,aoa_col]
                minority_rf=rf[truth[ii]!=majority]
                majority_rf=rf[truth[ii]==majority]
                minority_aoa=aoa[truth[ii]!=majority]
                majority_aoa=aoa[truth[ii]==majority]
                row=dict(window=wi,group=int(g),aoa_pop=int(Counter(pops[ii].tolist()).most_common(1)[0][0]),
                         size=len(ii),majority=int(majority),majority_count=int(majority_n),
                         minority_count=int(minority),minority_pct=round(100*minority/len(ii),3),
                         emitters=";".join(f"{int(e)}:{num}" for e,num in ct.most_common()),
                         rf_min=float(np.min(rf)),rf_max=float(np.max(rf)),
                         aoa_min=float(np.min(aoa)),aoa_max=float(np.max(aoa)),
                         rf_hist_overlap=round(pct_overlap(majority_rf,minority_rf),4),
                         rf_range_overlap=round(max(0,min(np.max(majority_rf),np.max(minority_rf))-max(np.min(majority_rf),np.min(minority_rf)))/max(np.ptp(rf),1e-12),4),
                         aoa_range_overlap=round(max(0,min(np.max(majority_aoa),np.max(minority_aoa))-max(np.min(majority_aoa),np.min(minority_aoa)))/max(np.ptp(aoa),1e-12),4))
                rows.append(row)
            if (wi+1)%10==0:
                print(f"Processed {min(start+a.window_size,len(h['data'])):,}/{len(h['data']):,}",flush=True)
        rows.sort(key=lambda r:(-r["minority_count"],-r["size"]))
        if rows:
            with (out/"mixed_blobs.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with (out/"emitter_pair_contributions.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["majority_emitter","minority_emitter","minority_pdws"])
            for (major,minor),num in contributions.most_common():
                writer.writerow([major,minor,num])
        print(f"Evaluated {n_total:,} PDWs; {len(rows)} mixed blobs")
        print(f"Minority PDWs: {n_minority:,}; PDWs in mixed blobs: {n_mixed_pdws:,}")
        print("Note: minority PDWs are not the same as all PDWs in mixed blobs.")
        print("Top mixed blobs (window, group, size, minority, emitters, RF histogram overlap):")
        for r in rows[:a.top]:
            print(f"  W{r['window']:02d} G{r['group']:03d} size={r['size']:4d} minority={r['minority_count']:4d} "
                  f"emitters={r['emitters']} RFoverlap={r['rf_hist_overlap']:.3f}")
        # Recompute only selected windows for plots; no need to store all PDWs.
        for rank,r in enumerate(rows[:a.plots],1):
            wi=r["window"]
            start=wi*a.window_size
            x=np.asarray(h["data"][start:start+a.window_size],float)
            truth=np.asarray(h["labels"][start:start+a.window_size]).reshape(-1)
            pops,npop=batch_aoa_populations(x[:,aoa_col])
            pred=group_window(x,names,pops,npop,8,.20,6.0,8)
            ii=np.flatnonzero(pred==r["group"])
            fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
            for eid in np.unique(truth[ii]):
                jj=ii[truth[ii]==eid]
                axes[0].scatter(x[jj,rf_col],x[jj,aoa_col],s=11,alpha=.65,label=f"Emitter {int(eid)} ({len(jj)})")
                if "ToA" in names:
                    axes[1].scatter(x[jj,names.index("ToA")],x[jj,rf_col],s=11,alpha=.65,label=f"Emitter {int(eid)}")
            axes[0].set(xlabel="RF (dataset units)",ylabel="AOA (degrees)",title="RF versus AOA")
            axes[1].set(xlabel="TOA (dataset units)",ylabel="RF (dataset units)",title="RF versus TOA")
            axes[0].legend(fontsize=8)
            fig.suptitle(f"Rank {rank}: window {wi}, blob {r['group']}, minority {r['minority_count']} / {r['size']}")
            fig.savefig(out/f"mixed_rank{rank:02d}_w{wi}_g{r['group']}.png",dpi=170)
            plt.close(fig)
    print(f"Saved diagnostics to {out}/")
    print("Overlap measures are descriptive; RF hist overlap depends on bin count.")
    print("Truth labels are used only to score and colour plots, not for grouping.")


if __name__=="__main__":
    main()
