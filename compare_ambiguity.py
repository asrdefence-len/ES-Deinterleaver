"""Compare association correctness and coverage without hiding abstentions."""
import json
import time
from pathlib import Path
from compare_pri import score
from testbed.generator import generate
from testbed.phase_folding import PhaseFoldingDeinterleaver
from testbed.bounded_mht import BoundedMHTDeinterleaver
from testbed.ambiguity_aware import AmbiguityAwareDeinterleaver


def main():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, truth = generate(scenario)
    print(f"Observed PDWs: {len(pdws)}")
    for name, algorithm in [
        ("Phase folding", PhaseFoldingDeinterleaver()),
        ("Bounded MHT", BoundedMHTDeinterleaver()),
        ("Ambiguity aware", AmbiguityAwareDeinterleaver()),
    ]:
        start = time.perf_counter()
        assigned = algorithm.process(pdws)
        elapsed = time.perf_counter() - start
        metrics = score(truth, assigned)
        coverage = metrics["coverage"]
        conditional = metrics["one_to_one_accuracy"] / coverage if coverage else 0
        print(f"{name:16s} all-PDW accuracy={metrics['one_to_one_accuracy']:.3%} "
              f"coverage={coverage:.3%} assigned-only accuracy={conditional:.3%} "
              f"tracks={metrics['tracks']} time={elapsed:.3f}s")
        if isinstance(algorithm, AmbiguityAwareDeinterleaver):
            print(f"  Ambiguous PDWs withheld: {len(algorithm.ambiguous)}; "
                  f"total unassigned: {len(algorithm.unassigned_pdw_ids)}")
    print("Unassigned PDWs count against all-PDW accuracy; no truth used in association.")


if __name__ == "__main__":
    main()
