"""Compare population, fixed and fractional RF scaling for candidate PDW groups.

Run:
  python turing_fractional_rf_comparison.py turing_data/config_100.h5 turing_data/testconfig_100.h5

Ground truth is used for scoring only. Groups are window-local, not emitter tracks.
"""
import argparse
import csv
from collections import Counter
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from turing_grouping_comparison import robust_scale


from candidate_grouping import group_pdws, RF_FRACTION

METHODS=("Population adaptive","Fixed 1000","Fractional 10%","Fractional 8%","Fractional 5%")


def grouping(x,names,method):
    """Historical comparators; 8% uses the canonical production-candidate path."""
    if method=="Fractional 8%":
        return group_pdws(x,names)
    rf=x[:,names.index("Frequency")]
    pw=x[:,names.index("PulseWidth")]
    angle=np.deg2rad(x[:,names.index("AoA")])
    pw_scale=robust_scale(pw)
    alpha=.05
    angular=np.column_stack((np.sin(angle)/np.deg2rad(2.),
                             np.cos(angle)/np.deg2rad(2.)))
    if method=="Population adaptive":
        rf_z=(rf-np.median(rf))/robust_scale(rf)
    elif method=="Fixed 1000":
        rf_z=(rf-np.median(rf))/1000.
    else:
        frac={"Fractional 10%":.10,"Fractional 5%":.05}[method]
        if np.any(rf<=0):
            raise ValueError("Fractional RF scaling requires positive frequency")
        rf_z=np.log(rf)/frac
    pw_z=np.log1p(alpha*pw/pw_scale)/alpha
    z=np.column_stack((angular,rf_z,pw_z))
    return DBSCAN(eps=1.,min_samples=8).fit_predict(z)


