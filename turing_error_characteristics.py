"""Diagnose mixed-emitter DBSCAN groups and their PDW characteristics.

Usage: python turing_error_characteristics.py turing_data/config_100.h5
Amplitude is a received measurement, NOT SNR unless a calibrated noise floor is supplied.
"""
import argparse
from collections import Counter, defaultdict
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from turing_grouping_comparison import groups, robust_scale


def pct(x):
    if len(x)==0:return "n/a"
    return "/".join(f"{v:.4g}" for v in np.percentile(x,[10,50,90]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--windows",type=int,nargs="*",default=[6,22,26,31],
                   help="Zero-based 10000-PDW window indices")
    p.add_argument("--window-size",type=int,default=10000)
    p.add_argument("--method",choices=["C","adaptive"],default="adaptive")
    p.add_argument("--noise-floor",type=float,default=None,
                   help="Only if Amplitude is calibrated dB power: noise floor in same dB units")
    a=p.parse_args()
    with h5py.File(a.file,"r") as f:
        names=[v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
        print("Features:",names)
        if "labels" not in f:raise ValueError("Ground-truth labels needed for diagnostics")
        for wi in a.windows:
            start=wi*a.window_size
            stop=min(start+a.window_size,f["data"].shape[0])
            if stop-start<8:continue
            x=np.asarray(f["data"][start:stop],dtype=float)
            truth=np.asarray(f["labels"][start:stop]).reshape(-1)
            rf=x[:,names.index("Frequency")]
            pw=x[:,names.index("PulseWidth")]
            aoa=x[:,names.index("AoA")]
            amp=x[:,names.index("Amplitude")] if "Amplitude" in names else None
            sr=robust_scale(rf);sp=robust_scale(pw)
            if a.method=="C":
                labels=groups(x,names,"C",2.,8,sr,sp)
            else:
                theta=np.deg2rad(aoa);alpha=.05
                z=np.column_stack((np.sin(theta)/np.deg2rad(2.),
                                   np.cos(theta)/np.deg2rad(2.),
                                   (rf-np.median(rf))/sr,
                                   np.log1p(alpha*pw/sp)/alpha))
                labels=DBSCAN(eps=1.,min_samples=8).fit_predict(z)
            wrong=np.zeros(len(x),dtype=bool)
            right=np.zeros(len(x),dtype=bool)
            contributors=[]
            for g in np.unique(labels[labels>=0]):
                ix=np.where(labels==g)[0]
                c=Counter(truth[ix].tolist())
                lead,n=c.most_common(1)[0]
                right[ix[truth[ix]==lead]]=True
                wrong[ix[truth[ix]!=lead]]=True
                if len(c)>1:
                    contributors.append((len(ix)-n,int(g),lead,c,ix))
            contributors.sort(reverse=True)
            print(f"\nWINDOW {wi} ({start}:{stop}) method={a.method}; correct={right.sum()}, cross={wrong.sum()}, noise={(labels<0).sum()}")
            for title,mask in (("Correctly grouped",right),("Cross-emitter minority",wrong),("Ungrouped",labels<0)):
                print(f"  {title} n={mask.sum()}")
                for label,values in (("RF",rf),("PW",pw),("AOA",aoa),("Amplitude",amp)):
                    if values is not None:print(f"    {label} p10/median/p90: {pct(values[mask])}")
                if amp is not None and a.noise_floor is not None:
                    print(f"    Approx SNR (dB) p10/median/p90: {pct(amp[mask]-a.noise_floor)}")
            print("  Largest contaminated groups (minority pulses, group ID, majority emitter, membership):")
            for n,g,lead,c,ix in contributors[:8]:
                print(f"    {n:4d} cross; group {g}; majority {lead}; emitters {dict(c.most_common(6))}")
                for emitter,count in c.most_common(3):
                    sub=ix[truth[ix]==emitter]
                    details=f"RF={pct(rf[sub])} PW={pct(pw[sub])} AOA={pct(aoa[sub])}"
                    if amp is not None:details+=f" Amplitude={pct(amp[sub])}"
                    print(f"      emitter {emitter} n={count}: {details}")
    if a.noise_floor is None:
        print("\nSNR NOT AVAILABLE: Amplitude is not automatically SNR. To estimate SNR, verify amplitude units, detector calibration, receiver noise power and processing gain.")
    print("Majority-label contamination is a diagnostic; it does not prove which pulses a real tracker would misidentify.")


if __name__=="__main__":
    main()
