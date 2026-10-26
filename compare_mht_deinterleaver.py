"""Diagnostic comparison on identical PDWs; truth is used ONLY for scoring.

Train IDs are not physical emitter IDs. Early mapping is frozen before the
PRI switch, so later identity drift cannot be hidden by retrospective mapping.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver
from testbed.mht_deinterleaver import MHTDeinterleaver

PERIODS = ((0.0, 0.4, "before"), (0.4, 0.6, "high_prf"),
           (0.6, 1.0, "after"))


def score(assignments, truth, pdws, early_cutoff=0.35):
    """Diagnostic scoring, never passed to either deinterleaver."""
    by_id = {p.pdw_id: p for p in pdws}
    early_votes = defaultdict(Counter)
    full_votes = defaultdict(Counter)
    for pid, train in assignments.items():
        full_votes[train][truth[pid]] += 1
        if by_id[pid].toa_s < early_cutoff:
            early_votes[train][truth[pid]] += 1
    early_map = {train: votes.most_common(1)[0][0]
                 for train, votes in early_votes.items()}
    full_map = {train: votes.most_common(1)[0][0]
                for train, votes in full_votes.items()}
    rows = []
    for start, end, label in PERIODS:
        for emitter in sorted(set(truth.values())):
            ids = [p.pdw_id for p in pdws
                   if start <= p.toa_s < end and truth[p.pdw_id] == emitter]
            assigned = [pid for pid in ids if pid in assignments]
            correct_early = sum(early_map.get(assignments[pid]) == emitter
                                for pid in assigned)
            incorrect_early = sum(assignments[pid] in early_map and
                                  early_map[assignments[pid]] != emitter
                                  for pid in assigned)
            unknown = len(assigned)-correct_early-incorrect_early
            rows.append(dict(period=label, emitter=emitter, observed=len(ids),
                             assigned=len(assigned), correct=correct_early,
                             incorrect=incorrect_early, unmapped=unknown,
                             withheld=len(ids)-len(assigned),
                             train_ids=dict(Counter(assignments[pid]
                                                    for pid in assigned))))
    return dict(early_map=early_map, full_map=full_map, rows=rows)


def run(path="scenarios/pri_mode_change.json"):
    pdws, truth = generate(json.loads(Path(path).read_text()))
    print("PDWs:", len(pdws))
    results = {}
    for name, processor in [
        ("Fixed-PRI streaming", StreamingDeinterleaver(bootstrap_pdws=600)),
        ("Bounded AOA-MHT", MHTDeinterleaver(bootstrap_pdws=600)),
    ]:
        start = perf_counter()
        for i in range(0, len(pdws), 100):
            processor.process_chunk(pdws[i:i+100])
        elapsed = perf_counter()-start
        diagnostic = score(processor.assignments, truth, pdws)
        correct_full = sum(diagnostic["full_map"].get(train) == truth[pid]
                           for pid, train in processor.assignments.items())
        print(f"\n{name}: assigned={len(processor.assignments)}/{len(pdws)} "
              f"full-run majority correct={correct_full}/{len(pdws)} "
              f"({correct_full/len(pdws):.1%}) time={elapsed:.3f}s")
        print("Frozen early train-to-truth map:", diagnostic["early_map"])
        print("period     radar   seen assigned correct wrong unmapped withheld  assigned train counts")
        for row in diagnostic["rows"]:
            print(f'{row["period"]:10} {row["emitter"]:5} '
                  f'{row["observed"]:5} {row["assigned"]:8} '
                  f'{row["correct"]:7} {row["incorrect"]:5} '
                  f'{row["unmapped"]:8} {row["withheld"]:8}  {row["train_ids"]}')
        print("Snapshot:", processor.snapshot())
        results[name] = diagnostic
    print("\nNOTES:")
    print("- Early mapping uses truth only for scoring and is frozen at 0.35 s.")
    print("- A pulse assigned to a train absent from the early map is 'unmapped',")
    print("  not silently counted as correct or wrong.")
    print("- Withheld includes unexplained and ambiguous PDWs; snapshot splits them.")
    print("- Train fragmentation and physical emitter continuity are NOT directly")
    print("  measured by this script; neither processor tracks physical emitter IDs.")
    print("- This MHT is bounded per-train beam search, not global MHT or N-scan.")
    return results


if __name__ == "__main__":
    run()
