"""Compare direct observed PRI variability against V0.1 for every RF blob of a truth-labelled emitter.

Usage: python -u compare_blob_jitter.py turing_data/config_101.h5 --window 0 --emitter 46
Truth is used ONLY for evaluation and selecting the reference emitter, never for
the V0.1 fit. Direct measurement uses consecutive observed pulses only (k=1).
If too few exist, reports N/A rather than inferring missing transmitted pulses.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from deinterleaver_v0 import discover
from deinterleaver_v01 import assess


def direct_measure(t, period, tolerance_fraction, min_gaps):
    """Truth-labelled observed adjacent intervals close to one candidate period.

    A conditional, gated descriptive statistic, NOT intrinsic transmitter jitter.
    """
    d=np.diff(np.sort(np.asarray(t,float)))
    gate=(d>0)&(np.abs(d-period)<=tolerance_fraction*period)
    chosen=d[gate]
    if len(chosen)<min_gaps:
        return len(chosen),None,None,None
    mu=float(np.mean(chosen))
    sd=float(np.std(chosen,ddof=1))
    return len(chosen),mu,sd,100*sd/mu


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window",type=int,default=0)
    ap.add_argument("--emitter",type=int,default=46)
    ap.add_argument("--toa-sigma",type=float,default=.25)
    ap.add_argument("--single-gap-tolerance",type=float,default=.12)
    ap.add_argument("--min-direct-gaps",type=int,default=8)
    ap.add_argument("--out",default="blob_jitter_comparison.csv")
    a=ap.parse_args()
    if not (a.toa_sigma>0 and 0<a.single_gap_tolerance<.5 and a.min_direct_gaps>=2):
        ap.error("Invalid parameters")
    with h5py.File(a.file) as h:
        start=a.window*10000
        stop=min(start+10000,len(h["data"]))
        if start<0 or start>=stop: ap.error("Window outside dataset")
        names=[v.decode() if isinstance(v,bytes) else str(v)
               for v in h["metadata/feature_names"][:]]
        x=np.asarray(h["data"][start:stop],float)
        truth=np.asarray(h["labels"][start:stop]).reshape(-1)
    pops,npop=batch_aoa_populations(x[:,names.index("AoA")])
    blobs=group_window(x,names,pops,npop,8,.20,6.,8)
    toa=x[:,names.index("ToA")]
    rf=x[:,names.index("Frequency")]
    target=np.flatnonzero(truth==a.emitter)
    blob_ids=sorted(set(blobs[target].tolist())-{-1})
    rows=[]
    print(f"Emitter {a.emitter}, window {a.window}: truth-labelled PDWs={len(target)}, "
          f"RF blobs={len(blob_ids)}")
    print("blob  RF_median   truth_PDW  V0_track  V0_assigned  PRI       direct_n  "
          "direct_CV%   V01_jitter%  status")
    for b in blob_ids:
        all_idx=np.flatnonzero(blobs==b)
        labelled=all_idx[truth[all_idx]==a.emitter]
        if len(all_idx)<3: continue
        # V0 and V0.1 operate on ALL PDWs in the blob, with no truth filtering.
        tracks,_=discover(toa[all_idx],20.,200.,2.,20,4,.1)
        if not tracks:
            rows.append(dict(blob=b,rf_median=float(np.median(rf[labelled])),
                truth_pdws=len(labelled),v0_track="NONE",v0_assigned=0,
                v0_period="",direct_n=0,direct_mean="",direct_sd="",
                direct_cv_pct="",v01_jitter_pct="",v01_status="NO_TRAIN"))
            print(f"{b:>4} {np.median(rf[labelled]):>10.2f} {len(labelled):>10} "
                  f"{'NONE':>9} {'-':>11} {'-':>9} {'-':>9} {'N/A':>11} "
                  f"{'N/A':>12} NO_TRAIN")
            continue
        for tr in tracks:
            # Direct truth-labelled measurement uses ALL observed target pulses in
            # this RF blob, independently of V0 association (but candidate PRI
            # from V0 is used as the single-gap selection reference).
            count,mean,sd,cv=direct_measure(toa[labelled],tr.period,
                a.single_gap_tolerance,a.min_direct_gaps)
            est=assess(toa[all_idx[tr.pdw_indices]],tr.period,a.toa_sigma)
            v=est.get("jitter_pct")
            status=est["status"]
            rows.append(dict(blob=b,rf_median=float(np.median(rf[labelled])),
                truth_pdws=len(labelled),v0_track=tr.id,
                v0_assigned=len(tr.pdw_indices),v0_period=tr.period,
                direct_n=count,direct_mean="" if mean is None else mean,
                direct_sd="" if sd is None else sd,
                direct_cv_pct="" if cv is None else cv,
                v01_jitter_pct="" if v is None else v,v01_status=status))
            cv_text=f"{cv:.2f}" if cv is not None else "N/A"
            v_text=f"{v:.2f}" if v is not None else "N/A"
            print(f"{b:>4} {np.median(rf[labelled]):>10.2f} {len(labelled):>10} "
                  f"{tr.id:>9} {len(tr.pdw_indices):>11} {tr.period:>9.3f} "
                  f"{count:>9} {cv_text:>11} {v_text:>12} {status}")
    path=Path(a.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["blob","rf_median","truth_pdws",
            "v0_track","v0_assigned","v0_period","direct_n","direct_mean",
            "direct_sd","direct_cv_pct","v01_jitter_pct","v01_status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")
    print("IMPORTANT: Direct CV is calculated ONLY from adjacent truth-labelled "
          "observed TOA gaps within +/-{:.0f}% of the V0 period. "
          "It is a selection-dependent observed variability measure, NOT "
          "independent transmitter-jitter ground truth. If direct_n is small, "
          "comparison is inconclusive. V0.1 also has fixed-phase selection bias. "
          "Truth never enters V0/V0.1.".format(100*a.single_gap_tolerance))


if __name__=="__main__":
    main()
