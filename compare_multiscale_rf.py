"""Experimental multiscale RF segmentation, independent of PW/TOA/truth.

Run: python -u compare_multiscale_rf.py turing_data/config_101.h5 --windows 1
This is a candidate, not a validated detector. Uses the same AOA grouping
and score function as the existing experiments.
"""
import argparse
import csv
import time
from collections import Counter
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_hdbscan_all import score


def multiscale_rf_labels(values, min_mass=8, valley_ratio=0.20,
                         min_gap_factor=6.0, k=8):
    """Segment sorted RF by stable, supported valleys across multiple scales.

    First detect robust *macro* separations with a k-neighbour bridge test,
    rather than a gap threshold dominated by a dense emitter. At each macro
    region estimate a local adaptive KDE; split only supported deep valleys.
    """
    f=np.asarray(values,float)
    labels=np.full(len(f),-1,dtype=int)
    if len(f)<min_mass:
        return labels
    order=np.argsort(f,kind="stable")
    s=f[order]
    n=len(s)
    if s[-1]==s[0]:
        labels[:]=0
        return labels
    gaps=np.diff(s)
    # Candidate macro boundaries: use the distance to k neighbours on
    # either side of the gap, avoiding a dense population setting the scale
    # for a remote, sparse one. Require support on both sides.
    macro=[]
    for i,g in enumerate(gaps):
        if i+1<min_mass or n-i-1<min_mass:
            continue
        left=s[i]-s[max(0,i-k)]
        right=s[min(n-1,i+1+k)]-s[i+1]
        # A bridge is anomalous only if it dwarfs both adjacent k-neighbour
        # extents. This does not use a global RF histogram resolution.
        if g>min_gap_factor*max(left,right,1e-10):
            macro.append(i+1)
    # Avoid repeated cuts near one discontinuity; each segment gets its own
    # KDE, and segments with insufficient mass remain unassigned.
    bounds=[0]+macro+[n]
    gid=0
    for a,b in zip(bounds[:-1],bounds[1:]):
        z=s[a:b]
        m=len(z)
        if m<min_mass:
            continue
        if z[-1]==z[0]:
            labels[order[a:b]]=gid
            gid+=1
            continue
        # Local kNN kernel widths, with a small robust floor to prevent
        # single repeated values producing unbounded density.
        kk=min(k,max(2,m//4))
        h=np.empty(m)
        for j in range(m):
            l=max(0,j-kk)
            r=min(m-1,j+kk)
            h[j]=max(z[j]-z[l],z[r]-z[j])
        pos=np.diff(z)
        pos=pos[pos>0]
        floor=max(float(np.median(pos)) if len(pos) else 1e-6,1e-6)
        h=np.maximum(h,2*floor)
        ng=int(np.clip(m*2,256,2048))
        grid=np.linspace(z[0]-2*np.median(h),z[-1]+2*np.median(h),ng)
        density=np.zeros(ng)
        for start in range(0,m,128):
            zz=z[start:start+128]
            hh=h[start:start+128]
            u=(grid[:,None]-zz[None,:])/hh[None,:]
            density+=np.sum(np.exp(-0.5*u*u)/(hh[None,:]*np.sqrt(2*np.pi)),axis=1)
        density=gaussian_filter1d(density,1)
        peaks,_=find_peaks(density,prominence=max(density.max()*.03,1e-12))
        if len(peaks)==0:
            peaks=np.array([np.argmax(density)])
        cuts=[]
        for p,q in zip(peaks[:-1],peaks[1:]):
            v=p+np.argmin(density[p:q+1])
            if density[v]<valley_ratio*min(density[p],density[q]):
                cut=grid[v]
                count_left=np.searchsorted(z,cut)
                if count_left>=min_mass and m-count_left>=min_mass:
                    cuts.append(cut)
        segments=np.searchsorted(cuts,z,side="right")
        for local in np.unique(segments):
            idx=np.flatnonzero(segments==local)
            if len(idx)<min_mass:
                continue
            # All observations in a supported segment are retained. There is
            # no global absolute density threshold that suppresses sparse RF.
            labels[order[a+idx]]=gid
            gid+=1
    return labels


def group_window(x,names,pops,npop,min_mass,valley_ratio,min_gap_factor,k):
    rf=x[:,names.index("Frequency")]
    pred=np.full(len(x),-1,dtype=int)
    gid=0
    for p in range(npop):
        ii=np.flatnonzero(pops==p)
        local=multiscale_rf_labels(rf[ii],min_mass,valley_ratio,min_gap_factor,k)
        for lab in np.unique(local[local>=0]):
            pred[ii[local==lab]]=gid
            gid+=1
    return pred


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--windows",type=int,default=1,help="0 for all windows")
    ap.add_argument("--window-size",type=int,default=10000)
    ap.add_argument("--min-mass",type=int,default=8)
    ap.add_argument("--valley-ratio",type=float,default=.20)
    ap.add_argument("--min-gap-factor",type=float,default=6.0)
    ap.add_argument("--k",type=int,default=8)
    ap.add_argument("--output",default="multiscale_rf_comparison.csv")
    a=ap.parse_args()
    if a.min_mass<2 or a.k<2 or a.min_gap_factor<=1 or not 0<a.valley_ratio<1:
        ap.error("Invalid parameters")
    rows=[]
    with h5py.File(a.file) as h:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in h["metadata/feature_names"][:]]
        n=len(h["data"])
        print(f"{a.file}: {n:,} PDWs",flush=True)
        for wi,start in enumerate(range(0,n,a.window_size)):
            if a.windows and wi>=a.windows: break
            end=min(start+a.window_size,n)
            x=np.asarray(h["data"][start:end],float)
            truth=np.asarray(h["labels"][start:end]).reshape(-1)
            if len(x)<8: continue
            pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
            for method in ("joint_RF_PW","multiscale_RF"):
                t0=time.perf_counter()
                pred=(rf_pw_groups(x,names,pops,npop) if method=="joint_RF_PW" else
                      group_window(x,names,pops,npop,a.min_mass,a.valley_ratio,a.min_gap_factor,a.k))
                elapsed=time.perf_counter()-t0
                m=score(pred,truth)
                rows.append(dict(window=wi,method=method,pdws=len(x),seconds=elapsed,**m))
                if wi==0:
                    print(f"WINDOW 0 {method}: {m['groups']} groups, {m['cross']} cross-emitter PDWs, {m['noise']} unassigned",flush=True)
                    for eid in (9,46,20,58):
                        ids=pred[truth==eid]
                        cnt=Counter(ids[ids>=0].tolist())
                        print(f"  emitter {eid}: {len(ids)} PDWs, {len(cnt)} groups, {sum(cnt.values())} grouped, largest={max(cnt.values(),default=0)}",flush=True)
            if (wi+1)%10==0 or end==n:
                print(f"Processed {end:,}/{n:,}",flush=True)
    if not rows: return
    with open(a.output,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmethod purity% coverage% cross noise groups seconds")
    for method in ("joint_RF_PW","multiscale_RF"):
        rr=[r for r in rows if r["method"]==method]
        n=sum(r["pdws"] for r in rr)
        grouped=sum(r["grouped"] for r in rr)
        print(f"{method:12s} {100*sum(r['correct'] for r in rr)/grouped if grouped else 0:7.3f} "
              f"{100*grouped/n:8.3f} {sum(r['cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} {sum(r['groups'] for r in rr):7d} "
              f"{sum(r['seconds'] for r in rr):7.2f}")
    print("Note: multiscale RF still uses a conservative k-neighbour macro-gap heuristic.")
    print("Saved",a.output)

if __name__=="__main__":
    main()
