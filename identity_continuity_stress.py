"""Crossing-PRI emitter identity continuity, using PDWs only.

Online decisions are compared with the final MAP reconstructed history.
A sustained swap requires a run of >= --min-run confident, wrong,
same-identity decisions for each true emitter (separately). This is a
proxy for track identity errors, NOT an inferred physical emitter swap.
"""
import argparse
import math
import random
from pdw_quality_stress import simulate
from testbed.joint_bayesian_timing import JointTimingTracker


def streaks(sequence, minimum=3):
    """Runs of True of length >= minimum, returned as (start, stop) inclusive."""
    if minimum < 1:
        raise ValueError("minimum must be positive")
    out=[]
    start=None
    for i,yes in enumerate(list(sequence)+[False]):
        if yes and start is None:
            start=i
        if not yes and start is not None:
            if i-start>=minimum:
                out.append((start,i-1))
            start=None
    return out


def evaluate(seed,scenario="very_close",count=45,pd=.7,beam=128,
             sigmas_ns=(50.,100.,250.),threshold=.8,min_run=3,
             quality_aware=True):
    if not .5<=threshold<=1.:
        raise ValueError("threshold must be between .5 and 1")
    if min_run<1:
        raise ValueError("min_run must be >=1")
    sigmas_ns=tuple(float(s) for s in sigmas_ns)
    if not sigmas_ns or any(not math.isfinite(s) or s<=0 for s in sigmas_ns):
        raise ValueError("Invalid TOA uncertainties")
    models,pdws=simulate(seed,scenario,count,pd,variable_sigma_ns=sigmas_ns)
    if not pdws:
        raise ValueError("No PDWs")
    rng=random.Random(seed+700001)
    initial=[(epoch+rng.gauss(0,1e-6),pri+rng.gauss(0,.2e-6),1e-6,.2e-6)
             for epoch,pri in models]
    tracker=JointTimingTracker(initial,beam_width=beam,max_skip=12)
    fixed_sigma=math.sqrt(sum(s*s for s in sigmas_ns)/len(sigmas_ns))*1e-9
    records=[]
    crossing=sum(epoch+22*pri for epoch,pri in models)/2
    for toa,sigma,truth,index in pdws:
        p=tracker.process(toa,toa_sigma_s=sigma if quality_aware else fixed_sigma)[
            "association_probabilities"][1]
        decision=int(p>.5)
        confidence=max(p,1-p)
        # Define pre/cross/post by the true pulse index; overlap spans
        # pulse indices 19..25 inclusive, not an arbitrary time gate.
        region="pre" if index<19 else "cross" if index<=25 else "post"
        records.append(dict(toa=toa,true_emitter=truth,pulse_index=index,
                            predicted_emitter=decision,confidence=confidence,
                            confident=confidence>=threshold,region=region))
    final_labels=tracker.snapshot()["best_assignments"]
    assert len(final_labels)==len(records)
    for rec,label in zip(records,final_labels):
        rec["final_map_emitter"]=label

    regions={}
    for region in ("pre","cross","post"):
        subset=[r for r in records if r["region"]==region]
        assigned=[r for r in subset if r["confident"]]
        regions[region]=dict(
            n=len(subset),
            online_accuracy=sum(r["predicted_emitter"]==r["true_emitter"]
                                for r in subset)/len(subset) if subset else None,
            confident_coverage=len(assigned)/len(subset) if subset else None,
            confident_accuracy=sum(r["predicted_emitter"]==r["true_emitter"]
                                   for r in assigned)/len(assigned) if assigned else None,
            final_map_accuracy=sum(r["final_map_emitter"]==r["true_emitter"]
                                   for r in subset)/len(subset) if subset else None)

    per_emitter={}
    for emitter in (0,1):
        rows=[r for r in records if r["true_emitter"]==emitter]
        # Only confident wrong online decisions count toward a sustained
        # identity-error episode; ambiguous observations break the run.
        wrong=[r["confident"] and r["predicted_emitter"]!=emitter for r in rows]
        episodes=streaks(wrong,min_run)
        recovery=sum(any(r["confident"] and r["predicted_emitter"]==emitter
                         for r in rows[stop+1:]) for _,stop in episodes)
        per_emitter[emitter]=dict(
            n=len(rows),sustained_wrong_runs=len(episodes),
            recovered_runs=recovery,
            longest_confident_wrong_run=max((b-a+1 for a,b in episodes),default=0),
            online_identity_accuracy=sum(r["predicted_emitter"]==emitter
                                         for r in rows)/len(rows) if rows else None,
            final_map_identity_accuracy=sum(r["final_map_emitter"]==emitter
                                           for r in rows)/len(rows) if rows else None)
    return dict(regions=regions,per_emitter=per_emitter,
                records=records,quality_aware=quality_aware)


def experiment(trials=100,**kwargs):
    if trials<1:
        raise ValueError("trials must be positive")
    rows=[evaluate(19017+7919*i,**kwargs) for i in range(trials)]
    regions={}
    for region in ("pre","cross","post"):
        entries=[r["records"][i] for r in rows
                 for i in range(len(r["records"]))
                 if r["records"][i]["region"]==region]
        confident=[r for r in entries if r["confident"]]
        regions[region]=dict(
            n=len(entries),
            online_accuracy=sum(r["predicted_emitter"]==r["true_emitter"]
                                for r in entries)/len(entries) if entries else None,
            confident_coverage=len(confident)/len(entries) if entries else None,
            confident_accuracy=sum(r["predicted_emitter"]==r["true_emitter"]
                                   for r in confident)/len(confident) if confident else None,
            final_map_accuracy=sum(r["final_map_emitter"]==r["true_emitter"]
                                   for r in entries)/len(entries) if entries else None)
    emitters={}
    for e in (0,1):
        stats=[r["per_emitter"][e] for r in rows]
        emitters[e]=dict(
            sustained_wrong_runs=sum(x["sustained_wrong_runs"] for x in stats),
            recovered_runs=sum(x["recovered_runs"] for x in stats),
            trials_with_sustained_wrong_run=sum(x["sustained_wrong_runs"]>0 for x in stats),
            max_confident_wrong_run=max(x["longest_confident_wrong_run"] for x in stats),
            recovery_fraction=(sum(x["recovered_runs"] for x in stats)/
                               sum(x["sustained_wrong_runs"] for x in stats)
                               if sum(x["sustained_wrong_runs"] for x in stats) else None))
    return dict(trials=trials,regions=regions,per_emitter=emitters)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario",choices=("baseline","close","very_close"),default="very_close")
    p.add_argument("--trials",type=int,default=100)
    p.add_argument("--pd",type=float,default=.7)
    p.add_argument("--count",type=int,default=45)
    p.add_argument("--beam",type=int,default=128)
    p.add_argument("--sigmas-ns",nargs="+",type=float,default=(50.,100.,250.))
    p.add_argument("--threshold",type=float,default=.8)
    p.add_argument("--min-run",type=int,default=3)
    p.add_argument("--quality-blind",action="store_true")
    a=p.parse_args()
    out=experiment(trials=a.trials,scenario=a.scenario,pd=a.pd,count=a.count,
                   beam=a.beam,sigmas_ns=a.sigmas_ns,threshold=a.threshold,
                   min_run=a.min_run,quality_aware=not a.quality_blind)
    for k,v in out.items():
        print(f"{k}: {v}")


if __name__=="__main__":
    main()
