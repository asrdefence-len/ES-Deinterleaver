"""Compare all three algorithms on the same observations and truth."""
import json
import time
from pathlib import Path
from testbed.generator import generate
from testbed.deinterleavers import FrequencyGateDeinterleaver
from testbed.pri_association import PRIGuidedDeinterleaver
from testbed.phase_folding import PhaseFoldingDeinterleaver
from compare_pri import score


def main():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, truth = generate(scenario)
    print(f"Scenario: {len(pdws)} observed PDWs; {len(set(truth.values()))} physical emitters")
    for name, algorithm in [
        ("Frequency only", FrequencyGateDeinterleaver()),
        ("PRI pulse-train", PRIGuidedDeinterleaver()),
        ("Phase folding", PhaseFoldingDeinterleaver()),
    ]:
        start = time.perf_counter()
        assignments = algorithm.process(pdws)
        elapsed = time.perf_counter() - start
        metrics = score(truth, assignments)
        print(f"{name:18s} accuracy={metrics['one_to_one_accuracy']:.3%} "
              f"coverage={metrics['coverage']:.3%} tracks={metrics['tracks']} "
              f"runtime={elapsed:.3f}s")
        if hasattr(algorithm, "tracks") and isinstance(algorithm, (PRIGuidedDeinterleaver, PhaseFoldingDeinterleaver)):
            for track in algorithm.tracks:
                print(f"  {track.track_id}: PRI={track.pri_model.mean_s*1e6:.3f} us; "
                      f"PDWs={len(track.associated_pdw_ids)}")


if __name__ == "__main__":
    main()
