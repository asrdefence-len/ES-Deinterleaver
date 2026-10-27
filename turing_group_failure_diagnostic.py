"""Inspect why emitters 54/33 are ungrouped and 18/22 are mixed.

python turing_group_failure_diagnostic.py turing_data/config_100.h5
Truth labels are used ONLY to report results, never for clustering.
"""
import argparse
from collections import Counter
import numpy as np
from turing_baseline import load_sample
from turing_grouping_comparison import groups,robust_scale

def describe(x):
    if not len(x):
        return "none"
    return f"{np.min(x):.4g}..{np.max(x):.4g} (median {np.median(x):.4g})"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    a=p.parse_args()
    data,truth,names,_=load_sample(a.file,a.limit)
    if truth is None:
        p.error("Requires truth labels for diagnostic")
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    aoa=data[:,names.index("AoA")]
    toa=data[:,names.index("ToA")]
    labels=groups(data,names,"C",a.aoa_deg,a.min_samples,robust_scale(rf),robust_scale(pw))
    print(f"AOA scale {a.aoa_deg} deg; DBSCAN min_samples {a.min_samples}")
    for emitter in (54,33,18,22):
        ids=np.flatnonzero(truth==emitter)
        print(f"\n=== Emitter {emitter}: {len(ids)} PDWs ===")
        print("RF:",describe(rf[ids]),"PW:",describe(pw[ids]))
        print("AOA:",describe(aoa[ids]),"TOA:",describe(toa[ids]))
        by_group=Counter(map(int,labels[ids]))
        print("Assignments (-1 is ungrouped):",dict(sorted(by_group.items())))
        for group,n in sorted(by_group.items()):
            if group<0:
                continue
            members=np.flatnonzero(labels==group)
            composition=Counter(map(int,truth[members]))
            print(f"  Group {group}: {n} target PDWs / {len(members)} total; truth composition {dict(composition.most_common(8))}")
            print("  Group RF:",describe(rf[members]),"PW:",describe(pw[members]),"AOA:",describe(aoa[members]))
        if np.any(labels[ids]<0):
            missing=ids[labels[ids]<0]
            print("  Ungrouped RF/PW/AOA/TOA (first 30):")
            for i in missing[:30]:
                print(f"    {rf[i]:.6g}  {pw[i]:.6g}  {aoa[i]:.3f}  {toa[i]:.6g}")
    print("\nDiagnostic only: no clustering parameters were changed.")

if __name__=="__main__":
    main()
