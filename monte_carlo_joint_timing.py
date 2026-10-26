"""Monte Carlo diagnostics for the experimental two-emitter joint tracker.

Reports causal MAP association, last-PDW posterior calibration, full-history
truth survival in the truncated beam, and terminal PRI errors. The oracle
fixed-model comparison knows true PRI/epoch and is an UPPER-BOUND reference,
not a fair deployable competitor. SNR-dependent TOA noise is illustrative.
Receiver overlaps are independently resolved; Pd is used in simulation but
NOT yet in the joint tracker's hypothesis prior.
"""
import argparse
import math
import random
from testbed.joint_bayesian_timing import JointTimingTracker
from two_emitter_snr_baseline import association, toa_sigma


def snr_a_at(n, count):
    return 0. if count//3 <= n < 2*count//3 else 30.


def simulate(seed, count=45, pd=.95):
    if count < 3 or not 0 < pd <= 1:
        raise ValueError("Invalid count or Pd")
    rng = random.Random(seed)
    truth = ((100e-6,.001),(105e-6,.00099))
    observations = []
    for emitter,(epoch,pri) in enumerate(truth):
        for n in range(count):
            snr = snr_a_at(n,count) if emitter == 0 else 20.
            if rng.random() < pd:
                toa = epoch+n*pri+rng.gauss(0,toa_sigma(snr))
                observations.append((toa,snr,emitter,n))
    observations.sort(key=lambda row:row[0])
    return observations,truth


def one_trial(seed, count=45, pd=.95, beam=32,
              epoch_sigma=1e-6, pri_sigma=.2e-6):
    observations, truth = simulate(seed,count,pd)
    rng = random.Random(seed+700001)
    initial = [(epoch+rng.gauss(0,epoch_sigma),
                pri+rng.gauss(0,pri_sigma),
                epoch_sigma,pri_sigma)
               for epoch,pri in truth]
    tracker = JointTimingTracker(initial,beam_width=beam,
                                 max_skip=12)
    oracle_models = [(epoch,pri,count) for epoch,pri in truth]
    correct = oracle_correct = 0
    brier = oracle_brier = 0.
    ambiguous = 0
    truth_alive = True
    first_pruned = None
    bins = [[0,0.,0.] for _ in range(10)]
    for step,(toa,snr,emitter,index) in enumerate(observations):
        result = tracker.process(toa,snr)
        p = result["association_probabilities"][1]
        op = association(toa,snr,oracle_models)[1]
        correct += int((p > .5) == bool(emitter))
        oracle_correct += int((op > .5) == bool(emitter))
        brier += (p-emitter)**2
        oracle_brier += (op-emitter)**2
        ambiguous += int(.1 < p < .9)
        b = min(9,int(p*10))
        bins[b][0] += 1
        bins[b][1] += p
        bins[b][2] += emitter
        # Exact full history survival is a stringent diagnostic:
        # one pruned branch cannot return. This is not track continuity.
        if truth_alive and not any(
                h.assignments == tuple(o[2] for o in observations[:step+1])
                and h.indices == tuple(o[3] for o in observations[:step+1])
                for h in tracker.hypotheses):
            truth_alive = False
            first_pruned = step
    best = tracker.snapshot()["best_tracks"]
    return {
        "n":len(observations),
        "accuracy":correct/len(observations),
        "oracle_accuracy":oracle_correct/len(observations),
        "brier":brier/len(observations),
        "oracle_brier":oracle_brier/len(observations),
        "ambiguous":ambiguous/len(observations),
        "truth_history_retained":truth_alive,
        "first_truth_pruned":first_pruned,
        "terminal_pri_error_ns":[(best[j].pri-truth[j][1])*1e9
                                 for j in range(2)],
        "bins":bins,
    }


def experiment(trials=10,count=45,pd=.95,beam=32):
    if trials < 1:
        raise ValueError("Positive trial count required")
    rows = [one_trial(19017+7919*i,count,pd,beam) for i in range(trials)]
    total = sum(r["n"] for r in rows)
    bins = [[sum(r["bins"][b][k] for r in rows) for k in range(3)]
            for b in range(10)]
    return {
        "trials":trials,"pd":pd,"beam":beam,"count_per_emitter":count,
        "detected_pdws":total,
        "joint_accuracy":sum(r["accuracy"]*r["n"] for r in rows)/total,
        "oracle_accuracy":sum(r["oracle_accuracy"]*r["n"] for r in rows)/total,
        "joint_brier":sum(r["brier"]*r["n"] for r in rows)/total,
        "oracle_brier":sum(r["oracle_brier"]*r["n"] for r in rows)/total,
        "ambiguous_fraction":sum(r["ambiguous"]*r["n"] for r in rows)/total,
        "truth_history_survival_fraction":sum(r["truth_history_retained"]
                                               for r in rows)/trials,
        "first_truth_pruned":[r["first_truth_pruned"] for r in rows],
        "terminal_pri_rmse_ns":[math.sqrt(sum(r["terminal_pri_error_ns"][j]**2
                                               for r in rows)/trials)
                                for j in range(2)],
        "calibration":[{"range":f"{i/10:.1f}-{(i+1)/10:.1f}",
                        "n":v[0],"predicted":v[1]/v[0],
                        "observed":v[2]/v[0]}
                       for i,v in enumerate(bins) if v[0]],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials",type=int,default=10)
    parser.add_argument("--count",type=int,default=45)
    parser.add_argument("--beam",type=int,default=32)
    parser.add_argument("--pd",type=float,default=.95)
    args=parser.parse_args()
    result=experiment(args.trials,args.count,args.pd,args.beam)
    for key,value in result.items():
        print(f"{key}: {value}")


if __name__=="__main__":
    main()
