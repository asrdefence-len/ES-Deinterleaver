"""Robust folded-phase core width for every RF blob of a labelled emitter.

Run: python -u compare_folded_core.py turing_data/config_101.h5 --window 0 --emitter 46

Truth labels select which blobs to REPORT, never used in PRI estimation or
phase statistics. Outliers have reduced influence through circular MAD.
"""
import argparse
import csv
import h5py
import numpy as np
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from deinterleaver_v0 import discover


def fold_stats(toa, period):
    t=np.asarray(toa,float)
    theta=2*np.pi*np.mod(t-t.min(),period)/period
    c=np.mean(np.cos(theta))
    s=np.mean(np.sin(theta))
    R=float(np.hypot(c,s))
    centre=np.arctan2(s,c)
    # Signed circular distances from the circular mean, in raw TOA units.
    residual=np.angle(np.exp(1j*(theta-centre)))*period/(2*np.pi)
    # Robust centre refinement: wrapped median, then circular absolute distances.
    shift=float(np.median(residual))
    core_residual=np.mod(residual-shift+period/2,period)-period/2
    mad=float(np.median(np.abs(core_residual)))
    core_sd=1.4826*mad
    return R,core_sd,100*core_sd/period,mad


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window",type=int,default=0)
    ap.add_argument("--emitter",type=int,default=46)
    ap.add_argument("--out",default="folded_core_comparison.csv")
    a=ap.parse_args()
    with h5py.File(a.file) as h:
        start=a.window*10000
        end=min(start+10000,len(h["data"]))
        if start<0 or end<=start: ap.error("Window outside dataset")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:end],float)
        labels=np.asarray(h["labels"][start:end]).reshape(-1)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    blobs=group_window(x,names,pops,npop,8,.20,6.,8)
    toa=x[:,names.index("ToA")]
    rf=x[:,names.index("Frequency")]
    selected=np.flatnonzero(labels==a.emitter)
    if not len(selected): ap.error("Emitter not found in this window")
    blob_ids=sorted(set(blobs[selected].tolist())-{-1})
    print(f"Emitter {a.emitter}, window {a.window}, RF blobs={len(blob_ids)}")
    print("blob  RF_median  PDWs  train   PRI       R       core_SD   core_pct  status")
    rows=[]
    for blob in blob_ids:
        ix=np.flatnonzero(blobs==blob)
        if len(ix)<3: continue
        tracks,_=discover(toa[ix],20.,200.,2.,20,4,.1)
        if not tracks:
            print(f"{blob:4d} {np.median(rf[ix]):10.2f} {len(ix):5d} "
                  "   -       -        -          -          -     NO_PRI")
            continue
        for tr in tracks:
            R,sd,pct,mad=fold_stats(toa[ix],tr.period)
            rows.append(dict(blob=blob,rf_median=float(np.median(rf[ix])),
                pdws=len(ix),track=tr.id,period=tr.period,
                coherence_R=R,core_sd=sd,core_pct=pct,mad=mad))
            print(f"{blob:4d} {np.median(rf[ix]):10.2f} {len(ix):5d} "
                  f"{tr.id:>5s} {tr.period:8.3f} {R:7.4f} "
                  f"{sd:10.4f} {pct:9.3f}  OK")
    if rows:
        with open(a.out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {a.out}")
    print("Core percentage is circular MAD-based PHASE dispersion, not "
          "transmitter PRI jitter. Every row folds ALL PDWs in its blob at "
          "the V0 candidate period, not just V0-assigned pulses. Multiple "
          "tracks within one blob therefore share PDWs. A harmonic PRI "
          "(e.g. 170.2 versus 85.1) changes the folded distribution and "
          "should not be compared without resolving that ambiguity. "
          "Truth is used only to select report blobs.")


if __name__=="__main__":
    main()
