"""Compare joint PDW tracker on three interleaved synthetic scenarios.

Run: python compare_joint_deinterleaver.py
Optional: python compare_joint_deinterleaver.py --pulses 80 --beam 20
This is a diagnostic benchmark, not evidence of operational accuracy.
"""
import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from testbed.joint_deinterleaving_tracker import JointDeinterleavingTracker


@dataclass(frozen=True)
class PDW:
    pdw_id: int
    toa_s: float
    frequency_hz: float = 9.4e9
    pulse_width_s: float = 10e-6
    aoa_deg: float = 45.0


def scenario(name, pulses=80, pd=.95, seed=13):
    """Truth is returned separately and never passed to the tracker."""
    if name not in ("two_constant", "change", "change_return"):
        raise ValueError(name)
    rng = random.Random(seed)
    events = []
    for emitter, phase in (("A", 0.00010), ("B", 0.00035)):
        toa = phase
        for index in range(pulses):
            if emitter == "B":
                pri_us = 750 if name == "two_constant" else 1020
            elif name == "two_constant":
                pri_us = 1000
            elif name == "change":
                pri_us = 1000 if index < pulses//2 else 750
            else:
                pri_us = (1000 if index < pulses//3 or index >= 2*pulses//3
                          else 750)
            if rng.random() < pd:
                events.append((toa+rng.gauss(0, 50e-9), emitter))
            toa += pri_us*1e-6
    events.sort(key=lambda e: e[0])
    pdws = [PDW(i, t) for i, (t, _) in enumerate(events)]
    truth = {i: label for i, (_, label) in enumerate(events)}
    return pdws, truth


def evaluate(tracker, truth):
    best = tracker.best()
    groups = defaultdict(list)
    for emitter in best.emitters:
        for pid in emitter.pdw_ids:
            groups[emitter.emitter_id].append(pid)
    assigned = {pid: eid for eid, ids in groups.items() for pid in ids}
    # Best one-to-one emitter-to-truth matching (two physical sources).
    # This is computed only for evaluation, never used during estimation.
    ids = sorted(groups)
    labels = sorted(set(truth.values()))
    from itertools import permutations
    best_match = 0
    for picked in permutations(ids, min(len(ids), len(labels))):
        score = sum(sum(truth[pid] == label for pid in groups[eid])
                    for eid, label in zip(picked, labels))
        best_match = max(best_match, score)
    return {
        "observed": len(truth),
        "assigned": len(assigned),
        "unassigned": len(truth)-len(assigned),
        "emitter_hypotheses": len(groups),
        "one_to_one_correct": best_match,
        "one_to_one_accuracy": best_match/len(truth),
        "purity_by_track": {
            eid: dict(Counter(truth[pid] for pid in ids))
            for eid, ids in groups.items()},
        "models": {e.emitter_id: (e.model, round(e.pri_s*1e6, 2))
                   for e in best.emitters},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulses", type=int, default=80)
    parser.add_argument("--pd", type=float, default=.95)
    parser.add_argument("--beam", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.pulses < 12 or not 0 < args.pd <= 1:
        parser.error("Require pulses>=12 and 0<Pd<=1")
    for name in ("two_constant", "change", "change_return"):
        pdws, truth = scenario(name, args.pulses, args.pd, args.seed)
        engine = JointDeinterleavingTracker(
            beam_width=args.beam, detection_probability=args.pd)
        for p in pdws:
            engine.process(p)
        result = evaluate(engine, truth)
        print(f"\n{name}: {result['observed']} PDWs, "
              f"{result['emitter_hypotheses']} emitter hypotheses, "
              f"{result['unassigned']} unassigned")
        print(f"  one-to-one correct: {result['one_to_one_correct']}/"
              f"{result['observed']} ({result['one_to_one_accuracy']:.1%})")
        print(f"  composition: {result['purity_by_track']}")
        print(f"  last models (model, PRI us): {result['models']}")
    print("\nCaution: this engine uses heuristic relative scores, not "
          "calibrated Bayesian posteriors. Failure is informative.")


if __name__ == "__main__":
    main()
