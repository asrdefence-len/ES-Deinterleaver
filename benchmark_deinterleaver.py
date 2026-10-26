"""Measure *current Python* deinterleaver throughput and history growth.

No tracker changes. Measures process() including snapshot() with perf_counter;
tracemalloc measures Python allocations (and slows execution, so memory is
measured in a separate run). Synthetic observations are independently resolved.
These are offline single-thread throughput figures, NOT realtime guarantees.
"""
import argparse
import math
import statistics
import time
import tracemalloc
from testbed.joint_bayesian_timing import JointTimingTracker
from testbed.bayesian_timing_state import BayesianTimingEstimator


def samples(kind, n):
    if kind == "joint":
        # Alternating well-separated emitters, each at 1 kHz.
        return [((i//2)*.001+(i%2)*.0003,
                 20.) for i in range(n)]
    return [(i*.001,20.) for i in range(n)]


def make_tracker(kind,beam,retained_mass=None,min_beam=1):
    if kind == "joint":
        return JointTimingTracker(
            ((0.,.001,1e-6,.2e-6),(.0003,.001,1e-6,.2e-6)),
            beam_width=beam,max_skip=12,\n            retained_mass=retained_mass,min_beam=min_beam)
    return BayesianTimingEstimator(.001,.2e-6,
                                    beam_width=beam,max_missing=12)


def measure(kind,beam,n,repeats=3,memory=False,\n            retained_mass=None,min_beam=1):
    if n < 2 or repeats < 1 or beam < 1:
        raise ValueError("Positive beam/repeats and n>=2 required")
    observations=samples(kind,n)
    rates=[]
    latency=[]
    peak_bytes=None\n    mean_retained=[]
    for _ in range(repeats):
        tracker=make_tracker(kind,beam,retained_mass,min_beam)
        if memory:
            tracemalloc.start()
        t0=time.perf_counter()
        for toa,snr in observations:
            before=time.perf_counter()
            tracker.process(toa,snr)
            if not memory:
                latency.append((time.perf_counter()-before)*1e6)
        elapsed=time.perf_counter()-t0
        rates.append(n/elapsed)\n        if kind=="joint":\n            mean_retained.append(len(tracker.hypotheses))
        if memory:
            _,peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_bytes=max(peak_bytes or 0,peak)
    latency.sort()
    def percentile(p):
        if not latency:
            return None
        return latency[math.ceil(p*len(latency))-1]
    return dict(kind=kind,beam=beam,pdws=n,repeats=repeats,
                throughput_pdws_s=round(statistics.median(rates),1),
                throughput_min_pdws_s=round(min(rates),1),
                throughput_max_pdws_s=round(max(rates),1),
                mean_latency_us=(round(statistics.mean(latency),2)
                                 if latency else None),
                p95_latency_us=(round(percentile(.95),2)
                                if latency else None),
                p99_latency_us=(round(percentile(.99),2)
                                if latency else None),
                final_retained_hypotheses=(mean_retained[-1] if mean_retained else None),\n                peak_python_allocated_mb=(round(peak_bytes/1048576,3)
                                          if peak_bytes is not None else None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beam",type=int,nargs="+",default=[1,8,32,64,128])
    p.add_argument("--length",type=int,nargs="+",default=[45,180])
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--retained-mass",type=float,default=None)\n    p.add_argument("--min-beam",type=int,default=1)\n    p.add_argument("--memory",action="store_true",
                   help="Separate tracemalloc run; timings slower")
    args=p.parse_args()
    print("kind,beam,pdws,repeats,throughput_pdws_s,p95_latency_us,p99_latency_us,peak_python_allocated_mb,final_retained_hypotheses")
    for kind in ("single","joint"):
        for n in args.length:
            for beam in args.beam:
                r=measure(kind,beam,n,args.repeats,args.memory,\n                          args.retained_mass,args.min_beam)
                print(",".join(str(r[k]) for k in
                    ("kind","beam","pdws","repeats","throughput_pdws_s",
                     "p95_latency_us","p99_latency_us",
                     "peak_python_allocated_mb","final_retained_hypotheses")))


if __name__=="__main__":
    main()
