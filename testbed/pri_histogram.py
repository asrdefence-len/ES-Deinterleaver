"""Local observed-gap mean and one/two humps; no fixed-train inference.

The original diagnostic algorithm and defaults are preserved. The mean is
conditional on selected half-height bins. It is not guaranteed to recover a
transmitter fundamental or an unbiased jitter distribution.
"""
from dataclasses import dataclass
import math
import numpy as np


def measure(toa, bin_width=2., min_gap=10., max_gap=400., neighbourhood=0.15):
    t=np.sort(np.asarray(toa,float))
    gaps=np.diff(t)
    gaps=gaps[(gaps>=min_gap)&(gaps<max_gap)]
    if len(gaps)<4: return None,gaps,None,None,None
    edges=np.arange(min_gap,max_gap+bin_width*1.01,bin_width)
    counts,_=np.histogram(gaps,bins=edges)
    # Identify local maxima; adjacent flat bins are treated as a single peak.
    peaks=[]
    for i in range(len(counts)):
        if counts[i]==0: continue
        left=counts[i-1] if i else -1
        right=counts[i+1] if i+1<len(counts) else -1
        if counts[i]>=left and counts[i]>=right:
            peaks.append(i)
    if not peaks: return None,gaps,counts,edges,None
    peaks.sort(key=lambda i:(-counts[i],edges[i]))
    first=peaks[0]
    centre=lambda i:(edges[i]+edges[i+1])/2
    primary=centre(first)
    # Restrict the optional second hump to the same local PRI neighbourhood.
    # This excludes distant missed-pulse multiples (2T, 3T), but a very broad
    # jitter distribution or genuine stagger beyond 15% may be excluded.
    nearby=[i for i in peaks[1:] if abs(centre(i)-primary)<=neighbourhood*primary
            and abs(i-first)>=2]
    second=nearby[0] if nearby else None
    chosen=[first]+([second] if second is not None else [])
    # Assign histogram bins around the selected hump(s), stopping at a local
    # valley or a low-count boundary. A broad unimodal hump is one population.
    # Limit each hump to its contiguous half-height support (min 1 count),
    # preventing distant tails/multiples from contaminating the estimate.
    selected=np.zeros(len(counts),dtype=bool)
    for p in chosen:
        threshold=max(1, int(np.ceil(counts[p]*0.5)))
        left=right=p
        while left>0 and counts[left-1]>=threshold:
            left-=1
        while right+1<len(counts) and counts[right+1]>=threshold:
            right+=1
        selected[left:right+1]=True
    ids=np.searchsorted(edges,gaps,side="right")-1
    keep=(ids>=0)&(ids<len(counts))&selected[np.clip(ids,0,len(counts)-1)]
    used=gaps[keep]
    if len(used)<2: return None,gaps,counts,edges,selected
    mean=float(np.mean(used))
    sd=float(np.std(used,ddof=1))
    return dict(pri=mean,sd=sd,pct=100*sd/mean,n_used=len(used),
                n_gaps=len(gaps),peak1=primary,
                peak2=centre(second) if second is not None else float("nan"),
                n_humps=len(chosen)),gaps,counts,edges,selected


@dataclass(frozen=True)
class GapPRI:
    mean_s: float
    spread_s: float
    peaks_s: tuple[float, ...]
    used_gaps: int
    in_range_gaps: int
    bin_width_s: float
    min_gap_s: float
    max_gap_s: float
    neighbourhood: float
    estimator: str = "local_gap_humps_v1"

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in
                    (self.mean_s, self.spread_s, self.bin_width_s,
                     self.min_gap_s, self.max_gap_s, self.neighbourhood, *self.peaks_s))
                or self.mean_s <= 0 or self.spread_s < 0
                or not 1 <= len(self.peaks_s) <= 2
                or not 2 <= self.used_gaps <= self.in_range_gaps
                or not 0 <= self.min_gap_s < self.max_gap_s
                or self.bin_width_s <= 0 or not 0 < self.neighbourhood < 1):
            raise ValueError("Invalid gap-PRI summary")

    @property
    def measurement_sigma_s(self):
        # Conservative initial dispersion/binning proxy, NOT a calibrated
        # standard error of the mean. Do not shrink it by pulse count.
        return max(self.spread_s, self.bin_width_s / math.sqrt(12.), .05e-6)


def estimate_gap_pri(toa_s, *, bin_width_s=2e-6, min_gap_s=10e-6,
                     max_gap_s=400e-6, neighbourhood=.15):
    values = np.asarray(toa_s, float)
    if (values.ndim != 1 or not np.all(np.isfinite(values))
            or not all(math.isfinite(v) for v in
                       (bin_width_s, min_gap_s, max_gap_s, neighbourhood))
            or not 0 <= min_gap_s < max_gap_s or bin_width_s <= 0
            or not 0 < neighbourhood < 1):
        raise ValueError("Invalid PRI data or estimator settings")
    result, *_ = measure(values * 1e6, bin_width_s * 1e6,
                          min_gap_s * 1e6, max_gap_s * 1e6, neighbourhood)
    if result is None:
        return None
    peaks = (result["peak1"],) + (() if result["n_humps"] == 1 else (result["peak2"],))
    return GapPRI(result["pri"] * 1e-6, result["sd"] * 1e-6,
                  tuple(v * 1e-6 for v in peaks), result["n_used"],
                  result["n_gaps"], bin_width_s, min_gap_s, max_gap_s, neighbourhood)
