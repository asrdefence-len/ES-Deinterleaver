"""Compare C and adaptive-5%-PW grouping across multiple Turing HDF5 configurations.

Run: python turing_multiconfig_validation.py 'turing_data/config_*.h5'
Optional: --limit 10000 --output turing_multiconfig_results.csv
Truth is used for evaluation only, never for grouping.
"""
import argparse
import csv
import glob
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale


def metrics(labels, truth):
    mask=labels>=0
    n=int(mask.sum())
    groups_seen=np.unique(labels[mask])
    leaders={}
    for group in groups_seen:
        counts=Counter(map(int,truth[labels==group]))
        leaders[int(group)]=counts.most_common(1)[0][0]
    correct=sum(int(np.count_nonzero((labels==group)&(truth==leader))) for group,leader in leaders.items())
    return dict(groups=len(leaders), grouped=n, coverage=n/len(labels),
                purity=correct/n if n else 0, cross=n-correct,
                correct=correct, dominant_emitters=len(set(leaders.values())),
                truth_emitters=len(np.unique(truth)))


def compare(path, limit, aoa_deg, min_samples):
    data,truth,names,_=load_sample(str(path),limit)
    if truth is None:
        raise ValueError("Ground truth unavailable")
    rf=data[:,names.index("Frequency")]
    pw=data[:,names.index("PulseWidth")]
    if np.any(pw<=0):
        raise ValueError("Non-positive pulse width")
    theta=np.deg2rad(data[:,names.index("AoA")])
    srf=robust_scale(rf)
    spw=robust_scale(pw)
    c=groups(data,names,"C",aoa_deg,min_samples,srf,spw)
    alpha=.05
    z=np.column_stack((np.sin(theta)/np.deg2rad(aoa_deg),
                       np.cos(theta)/np.deg2rad(aoa_deg),
                       (rf-np.median(rf))/srf,
                       np.log1p(alpha*pw/spw)/alpha))
    adaptive=DBSCAN(eps=1.,min_samples=min_samples).fit_predict(z)
    return srf,spw,metrics(c,truth),metrics(adaptive,truth)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("patterns",nargs="*",default=["turing_data/config_*.h5"],
                   help="HDF5 filenames or glob patterns (quote wildcard)")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--aoa-deg",type=float,default=2.)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--output",default="turing_multiconfig_results.csv")
    args=p.parse_args()
    paths=sorted({str(Path(path)) for pattern in args.patterns for path in glob.glob(pattern)})
    if not paths:
        p.error("No matching HDF5 files. Try: ls turing_data/config_*.h5")
    print(f"Comparing C vs adaptive alpha=0.05 on {len(paths)} files; limit={args.limit}, AOA={args.aoa_deg}, min_samples={args.min_samples}")
    rows=[]
    for path in paths:
        try:
            srf,spw,c,a=compare(path,args.limit,args.aoa_deg,args.min_samples)
        except Exception as exc:
            print(f"SKIP {path}: {type(exc).__name__}: {exc}")
            continue
        print(f"\n{path} (RF scale={srf:.4g}, PW scale={spw:.4g}, truth emitters={c['truth_emitters']})")
        for name,m in (("C",c),("Adaptive 5%",a)):
            print(f"  {name:12s}: groups={m['groups']:3d}, grouped={m['grouped']:5d}, purity={100*m['purity']:7.3f}%, cross={m['cross']:4d}, dominant emitters={m['dominant_emitters']:3d}")
            rows.append(dict(file=path,method=name,rf_scale=srf,pw_scale=spw,**m))
        print(f"  Adaptive minus C: grouped={a['grouped']-c['grouped']:+d}, cross={a['cross']-c['cross']:+d}, dominant emitters={a['dominant_emitters']-c['dominant_emitters']:+d}")
    if not rows:
        p.error("No valid datasets were processed")
    with open(args.output,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs={r["file"]:{} for r in rows}
    for r in rows:
        pairs[r["file"]][r["method"]]=r
    complete=[pair for pair in pairs.values() if "C" in pair and "Adaptive 5%" in pair]
    c_total=sum(p["C"]["correct"] for p in complete)
    a_total=sum(p["Adaptive 5%"]["correct"] for p in complete)
    c_grouped=sum(p["C"]["grouped"] for p in complete)
    a_grouped=sum(p["Adaptive 5%"]["grouped"] for p in complete)
    print(f"\nSUMMARY: {len(complete)} datasets")
    print(f"C: grouped={c_grouped}, correct={c_total}, micro-purity={100*c_total/c_grouped:.3f}%")
    print(f"Adaptive: grouped={a_grouped}, correct={a_total}, micro-purity={100*a_total/a_grouped:.3f}%")
    print(f"Adaptive improves coverage in {sum(p['Adaptive 5%']['grouped']>p['C']['grouped'] for p in complete)}/{len(complete)} datasets")
    print(f"Adaptive improves purity in {sum(p['Adaptive 5%']['purity']>p['C']['purity'] for p in complete)}/{len(complete)} datasets")
    print(f"Saved {args.output}")
    print("No parameters fitted using ground truth. Results across configurations may not be statistically independent.")


if __name__=="__main__":
    main()
