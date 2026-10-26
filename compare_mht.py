"""Compare baseline phase folding with experimental bounded MHT."""
import json
import time
from pathlib import Path
from compare_pri import score
from testbed.generator import generate
from testbed.phase_folding import PhaseFoldingDeinterleaver
from testbed.bounded_mht import BoundedMHTDeinterleaver


def main():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, truth = generate(scenario)
    print(f"Observed PDWs: {len(pdws)}")
    for name, algorithm in [
        ("Phase folding", PhaseFoldingDeinterleaver()),
        ("Bounded MHT (experimental)", BoundedMHTDeinterleaver()),
    ]:
        start = time.perf_counter()
        assigned = algorithm.process(pdws)
        elapsed = time.perf_counter() - start
        metrics = score(truth, assigned)
        print(f"{name}: accuracy={metrics['one_to_one_accuracy']:.3%}, "
              f"coverage={metrics['coverage']:.3%}, tracks={metrics['tracks']}, "
              f"time={elapsed:.3f}s")
        if isinstance(algorithm, BoundedMHTDeinterleaver):
            print(f"  Timing-ambiguous observations: {len(algorithm.ambiguous_pdw_ids)}")
            print(f"  Final hypotheses retained: {algorithm.hypotheses_retained}")
    print("NOTE: batch proof of concept; this does not benchmark streaming throughput.")


if __name__ == "__main__":
    main()
