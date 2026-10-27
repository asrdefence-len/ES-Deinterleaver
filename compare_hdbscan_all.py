"""Full-data regression: canonical DBSCAN versus experimental HDBSCAN.

Usage:
 python -u compare_hdbscan_all.py turing_data/config_*.h5
Ground truth is used ONLY for evaluation. Never for feature construction or fitting.
Each window is independently grouped; group IDs are not persistent emitter IDs.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
try:
    from sklearn.cluster import HDBSCAN
except ImportError as exc:
    raise SystemExit("Requires scikit-learn >=1.3: python -m pip install -U scikit-learn") from exc
from candidate_grouping import (group_pdws, RF_FRACTION, AOA_SCALE_DEG,
                                PW_ALPHA, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
                                robust_scale)


def feature_matrix(x,names):
    rf=np.asarray(x[:,names.index("Frequency")],dtype=float)
    pw=np.asarray(x[:,names.index("PulseWidth")],dtype=float)
    aoa=np.deg2rad(np.asarray(x[:,names.index("AoA")],dtype=float))
    if not (np.all(np.isfinite(rf)) and np.all(np.isfinite(pw)) and np.all(np.isfinite(aoa))):
        raise ValueError("Non-finite RF/PW/AOA")
    if np.any(rf<=0):raise ValueError("Non-positive RF")
    scale=robust_scale(pw)
    return np.column_stack((np.sin(aoa)/np.deg2rad(AOA_SCALE_DEG),
                            np.cos(aoa)/np.deg2rad(AOA_SCALE_DEG),
                            np.log(rf)/RF_FRACTION,
                            np.log1p(PW_ALPHA*pw/scale)/PW_ALPHA))


def score(pred,truth):
    grouped=int(np.sum(pred>=0))
    cross=0
    group_count=0
    for gid in np.unique(pred[pred>=0]):
        c=Counter(truth[pred==gid].tolist())
        cross+=sum(c.values())-c.most_common(1)[0][1]
        group_count+=1
    return dict(groups=group_count,grouped=grouped,noise=len(pred)-grouped,
                cross=cross,correct=grouped-cross,
                purity_pct=100*(grouped-cross)/grouped if grouped else 0.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("files",nargs="+")
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--min-cluster-size",type=int,default=8)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--output",default="turing_hdbscan_full_results.csv")
    p.add_argument("--focus-window",type=int,default=26)
    a=p.parse_args()
    rows=[]
    for path in a.files:
        with h5py.File(path) as f:
            names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
            n=len(f["data"])
            if "labels" not in f:raise ValueError(f"Missing truth labels: {path}")
            print(f"\n{path}: {n:,} PDWs",flush=True)
            for wi,start in enumerate(range(0,n,a.window_size)):
                stop=min(n,start+a.window_size)
                if stop-start<8:continue
                x=np.asarray(f["data"][start:stop],dtype=float)
                truth=np.asarray(f["labels"][start:stop]).reshape(-1)
                t=time.perf_counter()
                db=group_pdws(x,names)
                db_seconds=time.perf_counter()-t
                t=time.perf_counter()
                hd=HDBSCAN(min_cluster_size=a.min_cluster_size,min_samples=a.min_samples).fit_predict(feature_matrix(x,names))
                hd_seconds=time.perf_counter()-t
                for method,pred,elapsed in (("DBSCAN",db,db_seconds),("HDBSCAN",hd,hd_seconds)):
                    m=score(pred,truth)
                    rows.append(dict(file=path,window=wi,start=start,stop=stop,method=method,seconds=elapsed,**m))
                    if wi==a.focus_window:
                        print(f"  Window {wi} {method}: {m}",flush=True)
                if (wi+1)%10==0 or stop==n:
                    print(f"  processed {stop:,}/{n:,} PDWs",flush=True)
    if not rows:raise SystemExit("No windows with >=8 PDWs")
    with open(a.output,"w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print("\nFULL-DATA RESULTS (weighted by grouped PDWs)",flush=True)
    for path in a.files:
        for method in ("DBSCAN","HDBSCAN"):
            rr=[r for r in rows if r["file"]==path and r["method"]==method]
            grouped=sum(r["grouped"] for r in rr)
            correct=sum(r["correct"] for r in rr)
            cross=sum(r["cross"] for r in rr)
            noise=sum(r["noise"] for r in rr)
            groups=sum(r["groups"] for r in rr)
            elapsed=sum(r["seconds"] for r in rr)
            purity=100*correct/grouped if grouped else 0
            print(f"{Path(path).name:25s} {method:7s} purity={purity:7.3f}% cross={cross:7d} noise={noise:7d} groups={groups:7d} seconds={elapsed:8.2f}",flush=True)
    print("\nWORST HDBSCAN WINDOWS (by cross-emitter PDWs)")
    for r in sorted((r for r in rows if r["method"]=="HDBSCAN"),key=lambda r:r["cross"],reverse=True)[:10]:
        print(f"{Path(r['file']).name} window={r['window']} cross={r['cross']} purity={r['purity_pct']:.3f}% noise={r['noise']}")
    print(f"Saved {a.output}. Ground truth used only for scoring; DBSCAN baseline unchanged.")


if __name__=="__main__":
    main()
