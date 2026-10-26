"""Inspect Stage A branch accounting without changing the production tracker.

Run: python inspect_joint_detection.py --guard-us 5
     python inspect_joint_detection.py --guard-us 0
"""
import argparse
import math
from testbed.joint_bayesian_timing import JointTimingTracker, update
from two_emitter_snr_baseline import toa_sigma


def inspect_branches(tracker, toa, snr, top=8):
    """Reconstruct per-branch terms from tracker state before processing a PDW."""
    pd = tracker.joint_detection_probability
    if pd is None:
        raise ValueError("Requires joint_detection_probability")
    rows = []
    r = toa_sigma(snr)**2
    for parent_id, h in enumerate(tracker.hypotheses):
        for emitter, track in enumerate(h.tracks):
            start = h.accounted_indices[emitter] + 1
            for n in range(start, start+tracker.max_skip):
                result = update(track, toa, n, r)
                if result is None:
                    continue
                _, timing_ll = result
                other = 1-emitter
                t = h.tracks[other]
                due = math.ceil(((toa-tracker.joint_guard_s)-t.epoch)/t.pri)-1
                other_end = max(h.accounted_indices[other], due)
                missed_selected = n-h.accounted_indices[emitter]-1
                missed_other = other_end-h.accounted_indices[other]
                if missed_selected < 0 or missed_other < 0:
                    continue
                if pd == 1 and (missed_selected+missed_other):
                    continue
                detect_ll = math.log(pd)
                if missed_selected+missed_other:
                    detect_ll += (missed_selected+missed_other)*math.log1p(-pd)
                total = h.log_weight+timing_ll+detect_ll
                rows.append(dict(parent=parent_id, emitter=emitter, index=n,
                                 timing_ll=timing_ll, missed_selected=missed_selected,
                                 missed_other=missed_other, detection_ll=detect_ll,
                                 total=total, next_accounted=(n, other_end)
                                 if emitter==0 else (other_end,n)))
    rows.sort(key=lambda x:x["total"], reverse=True)
    return rows[:top]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard-us",type=float,default=5.)
    ap.add_argument("--pd",type=float,default=.7)
    ap.add_argument("--top",type=int,default=8)
    args=ap.parse_args()
    # Deliberately offset by 2 us, close enough for the 5 us guard to matter.
    initial=[(100e-6,.001,.2e-6,.01e-6),
             (102e-6,.001,.2e-6,.01e-6)]
    t=JointTimingTracker(initial,beam_width=64,max_skip=3,
                         joint_detection_probability=args.pd,
                         joint_guard_s=args.guard_us*1e-6)
    observations=[(100e-6,30.),(1.100e-3,30.),(2.102e-3,20.)]
    print("guard_us:",args.guard_us,"Pd:",args.pd)
    for toa,snr in observations:
        print("\nTOA us:",round(toa*1e6,3))
        for row in inspect_branches(t,toa,snr,args.top):
            print("parent={parent} emitter={emitter} index={index} "
                  "timing={timing_ll:.3f} miss_selected={missed_selected} "
                  "miss_other={missed_other} detection={detection_ll:.3f} "
                  "total={total:.3f} accounted={next_accounted}".format(**row))
        snapshot=t.process(toa,snr)
        print("posterior P(A),P(B):",snapshot["association_probabilities"])


if __name__=="__main__":
    main()
