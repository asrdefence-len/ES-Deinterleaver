"""Experimental multiscale RF density cores and conservative membership.

Usage: python -u compare_density_cores.py turing_data/config_101.h5 --windows 1
Truth is used only for evaluation. Never for clustering.
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


def density_core_labels(values, k=8, min_mass=8, valley_ratio=0.20,
                        support_fraction=0.02):
    """Find density-supported RF populations using local kNN bandwidth.

    Step 1: split only at genuinely empty, large-scale gaps using robust
    global within-population gap scale (avoids local tail overfragmentation).
    Step 2: within each region, use a local-bandwidth density field to find
    peaks separated by deep valleys. Keep distinct supported peaks.
    Step 3: retain observations supported by a density core; no nearest-core
    assignment across unsupported regions.
    """
    f=np.asarray(values,float)
    n=len(f)
    labels=np.full(n,-1,dtype=int)
    if n<min_mass:
        return labels
    order=np.argsort(f,kind="stable")
    s=f[order]
    gaps=np.diff(s)
    positive=gaps[gaps>0]
    if not len(positive):
        labels[:]=0
        return labels
    # An absolute empty-space separator, using a robust distribution-wide
    # reference, NOT a single local gap. Conservative to avoid tail splits.
    scale=np.percentile(positive,75)
    big=np.flatnonzero(gaps>max(25*scale,1e-9))+1
    bounds=np.r_[0,big,n]
    gid=0
    for a,b in zip(bounds[:-1],bounds[1:]):
        m=b-a
        if m<min_mass:
            continue
        z=s[a:b]
        if np.ptp(z)<1e-10:
            labels[order[a:b]]=gid
            gid+=1
            continue
        # k-nearest-neighbour radius in frequency units; bounded below by
        # local instrument-scale floor estimated from this region.
        kk=min(k,max(2,m//3))
        radii=np.empty(m)
        for i in range(m):
            l=max(0,i-kk)
            r=min(m-1,i+kk)
            radii[i]=max(z[i]-z[l],z[r]-z[i])
        floor=max(np.percentile(np.diff(z)[np.diff(z)>0],50) if np.any(np.diff(z)>0) else 1e-6,1e-6)
        h=np.maximum(radii,2*floor)
        # Evaluate normalized adaptive KDE on a grid. Each kernel integrates
        # to unit mass; broad blobs cannot dominate by raw peak height alone.
        ng=int(np.clip(m*2,256,2048))
        grid=np.linspace(z[0]-3*np.median(h),z[-1]+3*np.median(h),ng)
        density=np.zeros(ng)
        for start in range(0,m,128):
            zz=z[start:start+128]
            hh=h[start:start+128]
            u=(grid[:,None]-zz[None,:])/hh[None,:]
            density+=np.sum(np.exp(-0.5*u*u)/(hh[None,:]*np.sqrt(2*np.pi)),axis=1)
        density=gaussian_filter1d(density,1.0)
        peaks,_=find_peaks(density,prominence=max(density.max()*0.03,1e-12))
        if not len(peaks):
            peaks=np.array([np.argmax(density)])
        # Split only at deep valleys; evaluate mass after proposed splits.
        cuts=[]
        for p,q in zip(peaks[:-1],peaks[1:]):
            valley=p+np.argmin(density[p:q+1])
            if density[valley]<valley_ratio*min(density[p],density[q]):
                cuts.append(grid[valley])
        local=np.searchsorted(cuts,z,side="right")
        for lab in np.unique(local):
            idx=np.flatnonzero(local==lab)
            if len(idx)<min_mass:
                continue
            # Reject unsupported tails relative to each core's own peak.
            # Interpolate density for each observed frequency.
            dens=np.interp(z[idx],grid,density)
            lo,hi=z[idx[0]],z[idx[-1]]
            mask=dens>=support_fraction*np.max(dens)
            if mask.sum()>=min_mass:
                labels[order[a+idx[mask]]]=gid
                gid+=1
    return labels


def group_window(x,names,pops,npop,k,min_mass,valley_ratio,support_fraction):
    rf=x[:,names.index("Frequency")]
    pred=np.full(len(x),-1,dtype=int)
    gid=0
    for p in range(npop):
        ii=np.flatnonzero(pops==p)
        local=density_core_labels(rf[ii],k,min_mass,valley_ratio,support_fraction)
        for lab in np.unique(local[local>=0]):
            pred[ii[local==lab]]=gid
            gid+=1
    return pred


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--windows",type=int,default=1,help="0 for all windows")
    ap.add_argument("--window-size",type=int,default=10000)
    ap.add_argument("--k",type=int,default=8)
    ap.add_argument("--min-mass",type=int,default=8)
    ap.add_argument("--valley-ratio",type=float,default=0.20)
    ap.add_argument("--support-fraction",type=float,default=0.02)
    ap.add_argument("--output",default="density_cores_comparison.csv")
    a=ap.parse_args()
    if a.k<2 or a.min_mass<2 or not 0<a.valley_ratio<1 or not 0<a.support_fraction<1:
        ap.error("Invalid density parameters")
    rows=[]
    with h5py.File(a.file) as h:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in h["metadata/feature_names"][:]]
        n=len(h["data"])
        print(f"{a.file}: {n:,} PDWs",flush=True)
        for wi,start in enumerate(range(0,n,a.window_size)):
            if a.windows and wi>=a.windows:
                break
            end=min(start+a.window_size,n)
            x=np.asarray(h["data"][start:end],float)
            truth=np.asarray(h["labels"][start:end]).reshape(-1)
            if len(x)<8:
                continue
            pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
            for method in ("joint_RF_PW","density_cores"):
                t0=time.perf_counter()
                pred=(rf_pw_groups(x,names,pops,npop) if method=="joint_RF_PW" else
                      group_window(x,names,pops,npop,a.k,a.min_mass,a.valley_ratio,a.support_fraction))
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
    with open(a.output,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmethod purity% coverage% cross noise groups seconds")
    for method in ("joint_RF_PW","density_cores"):
        rr=[r for r in rows if r["method"]==method]
        n=sum(r["pdws"] for r in rr)
        grouped=sum(r["grouped"] for r in rr)
        print(f"{method:12s} {100*sum(r['correct'] for r in rr)/grouped if grouped else 0:7.3f} "
              f"{100*grouped/n:8.3f} {sum(r['cross'] for r in rr):7d} "
              f"{sum(r['noise'] for r in rr):7d} {sum(r['groups'] for r in rr):7d} "
              f"{sum(r['seconds'] for r in rr):7.2f}")
    print("Exploratory density cores: gap presegmentation is a heuristic, not calibrated attraction.")
    print("Saved",a.output)

if __name__=="__main__":
    main()
