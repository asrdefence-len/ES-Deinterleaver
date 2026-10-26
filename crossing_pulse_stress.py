"""Paired crossing-pulse stress test; tracker implementation unchanged.

Three PRI separations, crossings near pulse 22. Uses the SAME random
detections, SNR noise, and initial-prior standard-normal draws per trial
across scenarios. Exact coincident/overlapping pulses are independently
resolved by the simulated receiver: this is an idealization.

Caution: The current tracker uses uniform skip priors, not Pd, and a
truncated beam. Its posterior is not guaranteed calibrated.
"""
import argparse
import math
import random
from testbed.joint_bayesian_timing import JointTimingTracker
from two_emitter_snr_baseline import toa_sigma, association

SCENARIOS = {"baseline":10e-6, "close":1e-6, "very_close":.1e-6}
COUNT = 45
CROSS_INDEX = 22


def trial(seed, scenario="very_close", count=COUNT, pd=.95,
          beam=32, epoch_sigma=1e-6, pri_sigma=.2e-6,
          record_trace=False, retained_mass=None, min_beam=1,
          detection_probability=None):
    if scenario not in SCENARIOS or count < 3 or not 0 < pd <= 1:
        raise ValueError("Invalid scenario/count/Pd")
    delta = SCENARIOS[scenario]
    # A minus B arrival time = delta * (CROSS_INDEX - n).
    models = ((100e-6,.001),
              (100e-6+delta*CROSS_INDEX,.001-delta))
    rng = random.Random(seed)
    obs = []
    for emitter,(epoch,pri) in enumerate(models):
        for n in range(count):
            snr = (0. if count//3 <= n < 2*count//3 else 30.) if emitter==0 else 20.
            if rng.random() < pd:
                obs.append((epoch+n*pri+rng.gauss(0,toa_sigma(snr)),
                            snr,emitter,n))
    obs.sort(key=lambda x:x[0])
    prior_rng = random.Random(seed+700001)
    initial = [(epoch+prior_rng.gauss(0,epoch_sigma),
                pri+prior_rng.gauss(0,pri_sigma),
                epoch_sigma,pri_sigma)
               for epoch,pri in models]
    tracker = JointTimingTracker(initial,beam_width=beam,max_skip=12,
                                 retained_mass=retained_mass,min_beam=min_beam,
                                 detection_probability=detection_probability)
    oracle_models = [(epoch,pri,count) for epoch,pri in models]
    correct=oracle_correct=0
    brier=oracle_brier=0.
    ambiguous=0
    truth_alive=True
    first_pruned=None
    bins=[[0,0.,0.] for _ in range(10)]
    trace=[]
    retained_counts=[]
    discarded_masses=[]
    assignment_truth=[]
    index_truth=[]
    for step,(toa,snr,emitter,n) in enumerate(obs):
        snapshot=tracker.process(toa,snr)
        retained_counts.append(snapshot["retained_hypotheses"])
        discarded_masses.append(snapshot["pruning_diagnostics"]["discarded_mass_before_renormalization"])
        p=snapshot["association_probabilities"][1]
        op=association(toa,snr,oracle_models)[1]
        correct+=int((p>.5)==bool(emitter))
        oracle_correct+=int((op>.5)==bool(emitter))
        brier+=(p-emitter)**2
        oracle_brier+=(op-emitter)**2
        ambiguous+=int(.1<p<.9)
        bin_index=min(9,int(p*10))
        bins[bin_index][0]+=1
        bins[bin_index][1]+=p
        bins[bin_index][2]+=emitter
        assignment_truth.append(emitter)
        index_truth.append(n)
        if truth_alive and not any(
            h.assignments==tuple(assignment_truth) and
            h.indices==tuple(index_truth)
            for h in tracker.hypotheses):
            truth_alive=False
            first_pruned=step
        if record_trace:
            trace.append((step,toa,snr,emitter,n,p,op,
                          snapshot["retained_hypotheses"],truth_alive))
    best=tracker.snapshot()["best_tracks"]
    return dict(n=len(obs),accuracy=correct/len(obs),
                oracle_accuracy=oracle_correct/len(obs),
                brier=brier/len(obs),oracle_brier=oracle_brier/len(obs),
                ambiguous=ambiguous/len(obs),
                truth_history_retained=truth_alive,
                first_truth_pruned=first_pruned,
                pri_error_ns=[(best[j].pri-models[j][1])*1e9 for j in range(2)],
                mean_retained_hypotheses=sum(retained_counts)/len(retained_counts),
                mean_discarded_mass=sum(discarded_masses)/len(discarded_masses),
                bins=bins,trace=trace)


def experiment(scenario="very_close",trials=20,count=COUNT,pd=.95,beam=32,
               retained_mass=None,min_beam=1,detection_probability=None):
    if trials<1:
        raise ValueError("Positive trials required")
    rows=[trial(19017+7919*i,scenario,count,pd,beam,
                retained_mass=retained_mass,min_beam=min_beam)
          for i in range(trials)]
    total=sum(r["n"] for r in rows)
    bins=[[sum(r["bins"][j][k] for r in rows) for k in range(3)]
          for j in range(10)]
    return dict(scenario=scenario,pri_difference_ns=SCENARIOS[scenario]*1e9,
                trials=trials,pd=pd,beam=beam,detected_pdws=total,
                mean_retained_hypotheses=sum(r["mean_retained_hypotheses"]*r["n"] for r in rows)/total,
                mean_discarded_mass=sum(r["mean_discarded_mass"]*r["n"] for r in rows)/total,
                joint_accuracy=sum(r["accuracy"]*r["n"] for r in rows)/total,
                oracle_accuracy=sum(r["oracle_accuracy"]*r["n"] for r in rows)/total,
                joint_brier=sum(r["brier"]*r["n"] for r in rows)/total,
                oracle_brier=sum(r["oracle_brier"]*r["n"] for r in rows)/total,
                ambiguous_fraction=sum(r["ambiguous"]*r["n"] for r in rows)/total,
                truth_history_survival_fraction=sum(r["truth_history_retained"]
                                                     for r in rows)/trials,
                first_truth_pruned=[r["first_truth_pruned"] for r in rows],
                terminal_pri_rmse_ns=[math.sqrt(sum(r["pri_error_ns"][j]**2
                                                       for r in rows)/trials)
                                      for j in range(2)],
                calibration=[dict(bin=f"{j/10:.1f}-{(j+1)/10:.1f}",
                                  n=v[0],predicted=v[1]/v[0],
                                  observed=v[2]/v[0])
                             for j,v in enumerate(bins) if v[0]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario",choices=(*SCENARIOS,"all"),default="all")
    parser.add_argument("--trials",type=int,default=20)
    parser.add_argument("--count",type=int,default=COUNT)
    parser.add_argument("--pd",type=float,default=.95)
    parser.add_argument("--beam",type=int,default=32)
    parser.add_argument("--retained-mass",type=float,default=None)
    parser.add_argument("--min-beam",type=int,default=1)
    parser.add_argument("--detection-probability",type=float,default=None)
    parser.add_argument("--trace",action="store_true",
                        help="Print per-PDW trace for first trial")
    args=parser.parse_args()
    scenarios=SCENARIOS if args.scenario=="all" else (args.scenario,)
    for scenario in scenarios:
        print("\nSCENARIO",scenario)
        for key,value in experiment(scenario,args.trials,args.count,
                                    args.pd,args.beam,args.retained_mass,args.min_beam,
                                    args.detection_probability).items():
            print(f"{key}: {value}")
        if args.trace:
            print("step,toa_s,snr_db,true_emitter,pulse_index,p_emitter_B,oracle_p_B,beam_count,truth_alive")
            for row in trial(19017,scenario,args.count,args.pd,args.beam,
                             record_trace=True,retained_mass=args.retained_mass,
                             min_beam=args.min_beam,
                             detection_probability=args.detection_probability)["trace"]:
                print(",".join(map(str,row)))


if __name__=="__main__":
    main()
