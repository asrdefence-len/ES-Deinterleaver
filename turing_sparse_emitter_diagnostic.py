"""Diagnose missing clusters 33 and 54: feature distances versus TOA gaps.

Run: python turing_sparse_emitter_diagnostic.py turing_data/config_100.h5
Truth is used for post-hoc diagnostics only.
"""
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Ground truth required for diagnostic")
    aoa=np.deg2rad(data[:,names.index("AoA")])
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    toa=data[:,names.index("ToA")]
    rf_scale=robust_scale(rf)
    pw_scale=robust_scale(pw)
    z=np.column_stack((np.sin(aoa)/np.deg2rad(a.aoa_deg),
                       np.cos(aoa)/np.deg2rad(a.aoa_deg),
                       (rf-np.median(rf))/rf_scale,
                       (pw-np.median(pw))/pw_scale))
    labels=groups(data,names,"C",a.aoa_deg,a.min_samples,rf_scale,pw_scale)
    nn=NearestNeighbors(radius=1.0,algorithm="ball_tree").fit(z)
    print(f"Global RF scale={rf_scale:.6g}; PW scale={pw_scale:.6g}; AOA scale={a.aoa_deg} deg; DBSCAN eps=1, min_samples={a.min_samples}")
    print("IMPORTANT: TOA is NOT a DBSCAN feature, so temporal separation cannot directly prevent clustering.")
    for emitter in (54,33):
        ids=np.flatnonzero(truth==emitter)
        if not len(ids):
            print(f"Emitter {emitter} not in sample")
            continue
        ts=np.sort(toa[ids])
        dt=np.diff(ts)
        neighborhoods=nn.radius_neighbors(z[ids],return_distance=False)
        counts=np.array([len(x) for x in neighborhoods])
        same_counts=np.array([np.count_nonzero(truth[x]==emitter) for x in neighborhoods])
        print(f"\n=== EMITTER {emitter} ({len(ids)} pulses) ===")
        print(f"TOA span {ts[-1]-ts[0]:.6g} native units; gaps min/median/max {np.min(dt):.6g}/{np.median(dt):.6g}/{np.max(dt):.6g}" if len(dt) else "Single pulse")
        print("DBSCAN assignments:",dict(zip(*np.unique(labels[ids],return_counts=True))))
        print(f"Neighbour counts within eps=1 (including self): min={counts.min()}, median={np.median(counts):.1f}, max={counts.max()}, core candidates={np.count_nonzero(counts>=a.min_samples)}/{len(ids)}")
        print(f"Same-emitter neighbours: min={same_counts.min()}, median={np.median(same_counts):.1f}, max={same_counts.max()}")
        print("Pulse  TOA  gap_from_previous  RF  PW  AOA  neighbours  same_emitter")
        order=ids[np.argsort(toa[ids])]
        previous=None
        lookup={int(i):j for j,i in enumerate(ids)}
        for i in order:
            j=lookup[int(i)]
            gap=float(toa[i]-previous) if previous is not None else float("nan")
            print(f"{i:5d} {toa[i]:.6f} {gap:.6f} {rf[i]:.4f} {pw[i]:.4f} {np.rad2deg(aoa[i]):.3f} {counts[j]:3d} {same_counts[j]:3d}")
            previous=toa[i]
    print("\nNo clustering parameters changed.")

if __name__=="__main__":
    main()
