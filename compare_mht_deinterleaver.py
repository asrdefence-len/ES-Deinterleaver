"""Compare fixed-PRI streaming and bounded AOA-assisted MHT on identical PDWs.

This evaluates *pulse-train association*, not persistent emitter identity.
MHT uses the existing phase-folding bootstrap and is not full global MHT.
"""
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver
from testbed.mht_deinterleaver import MHTDeinterleaver


def run(path="scenarios/pri_mode_change.json"):
    pdws, truth = generate(json.loads(Path(path).read_text()))
    print("PDWs:", len(pdws))
    for name, processor in [
        ("Fixed-PRI streaming", StreamingDeinterleaver(bootstrap_pdws=600)),
        ("Bounded AOA-MHT", MHTDeinterleaver(bootstrap_pdws=600)),
    ]:
        start = perf_counter()
        for i in range(0, len(pdws), 100):
            processor.process_chunk(pdws[i:i+100])
        elapsed = perf_counter()-start
        # Map train IDs to truth only for post-hoc evaluation.
        mapping = {}
        for train in set(processor.assignments.values()):
            votes = Counter(truth[pid] for pid, tid in processor.assignments.items()
                            if tid == train)
            mapping[train] = votes.most_common(1)[0][0]
        correct = sum(mapping[train] == truth[pid]
                      for pid, train in processor.assignments.items())
        print(f"{name}: assigned={len(processor.assignments)}/{len(pdws)} "
              f"correct={correct}/{len(pdws)} ({correct/len(pdws):.1%}) "
              f"time={elapsed:.3f}s snapshot={processor.snapshot()}")
    print("NOTE: train-label accuracy is optimistically mapped using full-run truth.")
    print("NOTE: this MHT is per-train bounded beam search, not full global MHT.")


if __name__ == "__main__":
    run()
