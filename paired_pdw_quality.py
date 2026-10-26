"""Paired PDW-quality comparison: identical detections, aware vs fixed-sigma tracker.

Receiver-independent: no I/Q, SNR or matched filtering. Same initial
priors and PDW stream in both arms. Abstention is reporting-only.
"""
import argparse
import math
import random
from pdw_quality_stress import simulate
from testbed.joint_bayesian_timing import JointTimingTracker


def run_arm(pdws, initial, beam, threshold, quality_aware, fixed_sigma_s):
    tracker=JointTimingTracker(initial,beam_width=beam,max_skip=12)
    n=correct=assigned=assigned_correct=confident_wrong=0
    brier=0.
    bins=[[0,0.,0.] for _ in range(10)]
    confidence_bins=[[0,0.,0.] for _ in range(10)]
    for toa,sigma,emitter,_ in pdws:
        used_sigma=sigma if quality_aware else fixed_sigma_s
        snap=tracker.process(toa,toa_sigma_s=used_sigma)
        p=snap["association_probabilities"][1]
        prediction=int(p>.5)
        confidence=max(p,1-p)
        is_correct=int(prediction==emitter)
        n+=1
        correct+=is_correct
        brier+=(p-emitter)**2
        idx=min(9,int(p*10))
        bins[idx][0]+=1
        bins[idx][1]+=p
        bins[idx][2]+=emitter
        cidx=min(9,int(confidence*10))
        confidence_bins[cidx][0]+=1
        confidence_bins[cidx][1]+=confidence
        confidence_bins[cidx][2]+=is_correct
        if confidence>=threshold:
            assigned+=1
            assigned_correct+=is_correct
            confident_wrong+=1-is_correct
    return dict(n=n,correct=correct,assigned=assigned,
                assigned_correct=assigned_correct,confident_wrong=confident_wrong,
                brier_sum=brier,bins=bins,confidence_bins=confidence_bins)


def compare_trial(seed,scenario="very_close",count=45,pd=.7,beam=128,
                  sigmas_ns=(50.,100.,250.),fixed_sigma_ns=None,
                  threshold=.8):
    if not .5<=threshold<=1.:
        raise ValueError("threshold must be between 0.5 and 1")
    sigmas_ns=tuple(float(s) for s in sigmas_ns)
    if not sigmas_ns or any(not math.isfinite(s) or s<=0 for s in sigmas_ns):
        raise ValueError("sigmas must be positive and finite")
    if fixed_sigma_ns is None:
        fixed_sigma_ns=math.sqrt(sum(s*s for s in sigmas_ns)/len(sigmas_ns))
    if not math.isfinite(fixed_sigma_ns) or fixed_sigma_ns<=0:
        raise ValueError("fixed sigma must be positive and finite")
    models,pdws=simulate(seed,scenario,count,pd,variable_sigma_ns=sigmas_ns)
    if not pdws:
        raise ValueError("No PDWs in trial")
    rng=random.Random(seed+700001)
    initial=[(epoch+rng.gauss(0,1e-6),pri+rng.gauss(0,.2e-6),1e-6,.2e-6)
             for epoch,pri in models]
    aware=run_arm(pdws,initial,beam,threshold,True,fixed_sigma_ns*1e-9)
    blind=run_arm(pdws,initial,beam,threshold,False,fixed_sigma_ns*1e-9)
    return dict(aware=aware,blind=blind,fixed_sigma_ns=fixed_sigma_ns)


def aggregate(rows,arm):
    total=sum(r[arm]["n"] for r in rows)
    assigned=sum(r[arm]["assigned"] for r in rows)
    wrong=sum(r[arm]["confident_wrong"] for r in rows)
    bins=[[sum(r[arm]["bins"][i][k] for r in rows) for k in range(3)]
          for i in range(10)]
    cbins=[[sum(r[arm]["confidence_bins"][i][k] for r in rows) for k in range(3)]
           for i in range(10)]
    return dict(
        detected_pdws=total,
        accuracy=sum(r[arm]["correct"] for r in rows)/total,
        brier=sum(r[arm]["brier_sum"] for r in rows)/total,
        assignment_coverage=assigned/total,
        assigned_accuracy=sum(r[arm]["assigned_correct"] for r in rows)/assigned
                          if assigned else None,
        false_confident_rate_all_pdws=wrong/total,
        false_confident_rate_among_assigned=wrong/assigned if assigned else None,
        calibration=[dict(probability_bin=f"{i/10:.1f}-{(i+1)/10:.1f}",
                          n=v[0],mean_predicted=v[1]/v[0],observed=v[2]/v[0])
                     for i,v in enumerate(bins) if v[0]],
        confidence_calibration=[
            dict(confidence_bin=f"{i/10:.1f}-{(i+1)/10:.1f}",
                 n=v[0],mean_confidence=v[1]/v[0],observed_accuracy=v[2]/v[0])
            for i,v in enumerate(cbins) if v[0]]
    )


def experiment(trials=100,**kwargs):
    if trials<1:
        raise ValueError("trials must be positive")
    rows=[compare_trial(19017+7919*i,**kwargs) for i in range(trials)]
    aware=aggregate(rows,"aware")
    blind=aggregate(rows,"blind")
    return dict(trials=trials,fixed_sigma_ns=rows[0]["fixed_sigma_ns"],
                quality_aware=aware,quality_blind=blind,
                paired_deltas_aware_minus_blind={
                    k:aware[k]-blind[k]
                    for k in ("accuracy","brier","assignment_coverage",
                              "false_confident_rate_all_pdws")})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario",choices=("baseline","close","very_close"),
                   default="very_close")
    p.add_argument("--trials",type=int,default=100)
    p.add_argument("--count",type=int,default=45)
    p.add_argument("--pd",type=float,default=.7)
    p.add_argument("--beam",type=int,default=128)
    p.add_argument("--sigmas-ns",type=float,nargs="+",default=(50.,100.,250.))
    p.add_argument("--fixed-sigma-ns",type=float,default=None)
    p.add_argument("--threshold",type=float,default=.8)
    p.add_argument("--show-calibration",action="store_true")
    a=p.parse_args()
    result=experiment(trials=a.trials,scenario=a.scenario,count=a.count,pd=a.pd,
                      beam=a.beam,sigmas_ns=a.sigmas_ns,
                      fixed_sigma_ns=a.fixed_sigma_ns,threshold=a.threshold)
    print("trials:",result["trials"])
    print("fixed_sigma_ns:",result["fixed_sigma_ns"])
    for arm in ("quality_aware","quality_blind"):
        print("\\n"+arm)
        for key,value in result[arm].items():
            if key.endswith("calibration") and not a.show_calibration:
                continue
            print(f"{key}: {value}")
    print("\\npaired_deltas_aware_minus_blind:")
    for k,v in result["paired_deltas_aware_minus_blind"].items():
        print(f"{k}: {v}")


if __name__=="__main__":
    main()
