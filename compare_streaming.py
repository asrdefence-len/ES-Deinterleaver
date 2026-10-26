"""First incremental streaming benchmark: fixed-PRI, same-frequency emitters."""
import json
import time
from pathlib import Path
from compare_pri import score
from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver


def main():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, truth = generate(scenario)
    tracker = StreamingDeinterleaver(bootstrap_pdws=600)
    start = time.perf_counter()
    for offset in range(0, len(pdws), 100):
        tracker.process_chunk(pdws[offset:offset + 100])
    elapsed = time.perf_counter() - start
    metrics = score(truth, tracker.assignments)
    coverage = metrics["coverage"]
    conditional = metrics["one_to_one_accuracy"] / coverage if coverage else 0
    print(f"Streaming fixed-PRI: all-PDW accuracy={metrics['one_to_one_accuracy']:.3%}, "
          f"coverage={coverage:.3%}, assigned-only accuracy={conditional:.3%}, "
          f"tracks={metrics['tracks']}, elapsed={elapsed:.3f}s")
    print(f"Snapshot: {json.dumps(tracker.snapshot(), indent=2)}")
    print("NOTE: experimental incremental association, not yet a robust streaming "
          "multi-mode deinterleaver or sustained throughput benchmark.")


if __name__ == "__main__":
    main()
