"""RF-scale sensitivity for conservative candidate grouping (truth used for scoring only).

Usage: python turing_rf_scale_sensitivity.py turing_data/config_100.h5
"""
import argparse
from collections import Counter
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from turing_grouping_comparison import robust_scale


def evaluate(labels, truth):
    active=labels>=0
    correct=0
    cross=0
    mixed=0
    for gid in np.unique(labels[active]):
        counts=Counter(truth[labels==gid].tolist())
        n=sum(counts.values())
        correct+=max(counts.values())
        cross+=n-max(counts.values())
        mixed+=int(len(counts)>1)
    grouped=int(active.sum())
    return dict(groups=len(np.unique(labels[active])),grouped=grouped,
                noise=len(labels)-grouped,correct=correct,cross=cross,
                purity=100*correct/grouped if grouped else 0.,mixed=mixed)


def cluster(x,names,rf_scale,pw_scale):
    rf=x[:,names.index("Frequency")]
    pw=x[:,names.index("PulseWidth")]
    aoa=np.deg2rad(x[:,names.index("AoA")])
    alpha=.05
    z=np.column_stack((np.sin(aoa)/np.deg2rad(2.),
                       np.cos(aoa)/np.deg2rad(2.),
                       (rf-np.median(rf))/rf_scale,
                       np.log1p(alpha*pw/pw_scale)/alpha))
    return DBSCAN(eps=1.,min_samples=8).fit_predict(z)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--focus-window",type=int,default=26)
    p.add_argument("--rf-scales",type=float,nargs="+",default=[1000.,500.,250.])
    p.add_argument("--all-windows",action="store_true",
                   help="Also score all windows; uses local RF scale as baseline")
    a=p.parse_args()
    with h5py.File(a.file,"r") as f:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
        if "labels" not in f:raise ValueError("Truth labels required for scoring")
        total=len(f["data"])
        starts=(range(0,total,a.window_size) if a.all_windows
                else [a.focus_window*a.window_size])
        summary={}
        for start in starts:
            stop=min(start+a.window_size,total)
            if stop-start<8:continue
            x=np.asarray(f["data"][start:stop],dtype=float)
            truth=np.asarray(f["labels"][start:stop]).reshape(-1)
            sr=robust_scale(x[:,names.index("Frequency")])
            sp=robust_scale(x[:,names.index("PulseWidth")])
            methods=[("baseline",sr)]+[(f"RF={s:g}",s) for s in a.rf_scales]
            if start==a.focus_window*a.window_size:
                print(f"FOCUS window {start}:{stop}; baseline RF scale={sr:.3f}; PW scale={sp:.6f}")
            for name,scale in methods:
                labels=cluster(x,names,scale,sp)
                m=evaluate(labels,truth)
                agg=summary.setdefault(name,dict(grouped=0,correct=0,cross=0,noise=0,groups=0,windows=0))
                for key in ("grouped","correct","cross","noise","groups"):agg[key]+=m[key]
                agg["windows"]+=1
                if start==a.focus_window*a.window_size:
                    print(f"  {name:12s} groups={m['groups']:3d} grouped={m['grouped']:5d} cross={m['cross']:4d} purity={m['purity']:7.3f}% noise={m['noise']:4d}")
                    for e in (46,72,80):
                        idx=truth==e
                        c=Counter(labels[idx].tolist())
                        print(f"    emitter {e}: n={idx.sum()} groups={dict(c.most_common(12))}")
                    mixed=[]
                    for gid in np.unique(labels[labels>=0]):
                        c=Counter(truth[labels==gid].tolist())
                        if c[46] and c[80]:
                            mixed.append((int(gid),c[46],c[80],c[72]))
                    print(f"    groups containing BOTH 46 and 80: {mixed}")
        if a.all_windows:
            print("\nALL-WINDOW SUMMARY")
            for name,s in summary.items():
                print(f"{name:12s} windows={s['windows']:2d} grouped={s['grouped']:7d} correct={s['correct']:7d} cross={s['cross']:6d} noise={s['noise']:6d} total_groups={s['groups']:5d} purity={100*s['correct']/s['grouped']:.3f}%")
    print("\nRF scales are in dataset frequency units. Labels used for evaluation only.")
    print("Caution: reducing RF scale can fragment frequency-agile emitters; total groups are window-local.")


if __name__=="__main__":
    main()
