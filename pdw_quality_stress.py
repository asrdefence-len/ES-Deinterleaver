"""Receiver-independent PDW timing-quality experiment.

Two established emitters with crossing PRIs. Generates TOA measurements
directly with specified sigma; no SNR, I/Q, detection waveform or receiver
processing. Detection availability is independent Bernoulli. Truth is used
only for evaluation, never supplied to the tracker except optional diagnostic.
"""
import argparse
import math
import random
from testbed.joint_bayesian_timing import JointTimingTracker

SCENARIOS = {"baseline": 10e-6, "close": 1e-6, "very_close": .1e-6}


def simulate(seed, scenario="very_close", count=45, pd=.7,
             toa_sigma_ns=100., variable_sigma_ns=None):
    if scenario not in SCENARIOS or count < 3 or not 0 < pd <= 1:
        raise ValueError("Invalid scenario, count or availability")
    sigmas = (float(toa_sigma_ns),) if variable_sigma_ns is None else tuple(variable_sigma_ns)
    if not sigmas or any(not math.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("TOA sigma must be positive and finite")
    delta = SCENARIOS[scenario]
    models = ((100e-6, .001), (100e-6+22*delta, .001-delta))
    rng = random.Random(seed)
    pdws = []
    for emitter, (epoch, pri) in enumerate(models):
        for n in range(count):
            if rng.random() < pd:
                sigma_ns = sigmas[n % len(sigmas)]
                toa = epoch+n*pri+rng.gauss(0.,sigma_ns*1e-9)
                pdws.append((toa,sigma_ns*1e-9,emitter,n))
    pdws.sort(key=lambda row:row[0])
    return models,pdws


def trial(seed, scenario="very_close", count=45, pd=.7,
          toa_sigma_ns=100., variable_sigma_ns=None, beam=128,
          abstain_threshold=.8):
    if not .5 <= abstain_threshold <= 1.:
        raise ValueError("abstain_threshold must be between .5 and 1")
    models,pdws=simulate(seed,scenario,count,pd,toa_sigma_ns,variable_sigma_ns)
    prior_rng=random.Random(seed+700001)
    initial=[(epoch+prior_rng.gauss(0,1e-6),
              pri+prior_rng.gauss(0,.2e-6),1e-6,.2e-6)
             for epoch,pri in models]
    tracker=JointTimingTracker(initial,beam_width=beam,max_skip=12)
    correct=assigned=assigned_correct=0
    brier=0.
    for toa,sigma,emitter,n in pdws:
        # No receiver SNR is passed. sigma is the PDW measurement quality.
        snap=tracker.process(toa,toa_sigma_s=sigma)
        p=snap["association_probabilities"][1]
        prediction=int(p>.5)
        correct+=int(prediction==emitter)
        brier+=(p-emitter)**2
        if max(p,1-p)>=abstain_threshold:
            assigned+=1
            assigned_correct+=int(prediction==emitter)
    total=len(pdws)
    return dict(detected_pdws=total,
                accuracy=correct/total if total else None,
                brier=brier/total if total else None,
                assignment_coverage=assigned/total if total else None,
                assigned_accuracy=assigned_correct/assigned if assigned else None)


def experiment(trials=20,**kwargs):
    if trials<1:
        raise ValueError("trials must be positive")
    rows=[trial(19017+7919*i,**kwargs) for i in range(trials)]
    total=sum(r["detected_pdws"] for r in rows)
    # For aggregate assigned accuracy, reconstruct correct assignments
    assigned=sum(r["assignment_coverage"]*r["detected_pdws"] for r in rows)
    correct_assigned=sum((r["assigned_accuracy"] or 0)*
                         r["assignment_coverage"]*r["detected_pdws"] for r in rows)
    return dict(trials=trials,detected_pdws=total,
                accuracy=sum(r["accuracy"]*r["detected_pdws"] for r in rows)/total,
                brier=sum(r["brier"]*r["detected_pdws"] for r in rows)/total,
                assignment_coverage=assigned/total,
                assigned_accuracy=correct_assigned/assigned if assigned else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario",choices=SCENARIOS,default="very_close")
    p.add_argument("--trials",type=int,default=20)
    p.add_argument("--pd",type=float,default=.7,
                   help="PDW availability independent of measurement quality")
    p.add_argument("--beam",type=int,default=128)
    p.add_argument("--toa-sigma-ns",type=float,default=100.)
    p.add_argument("--variable-sigma-ns",type=float,nargs="+",default=None)
    p.add_argument("--abstain-threshold",type=float,default=.8)
    a=p.parse_args()
    for k,v in experiment(trials=a.trials,scenario=a.scenario,pd=a.pd,
                          beam=a.beam,toa_sigma_ns=a.toa_sigma_ns,
                          variable_sigma_ns=a.variable_sigma_ns,
                          abstain_threshold=a.abstain_threshold).items():
        print(f"{k}: {v}")


if __name__=="__main__":
    main()
