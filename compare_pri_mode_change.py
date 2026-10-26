"""Evaluate fixed-PRI streaming baseline against a PRI-changing emitter.

No adaptive tracker changes here: this is a controlled failure-mode experiment.
Track-to-emitter mapping is computed from the first 0.35 seconds only to
measure continuity without remapping identities after the mode switch.
"""
import json
from collections import Counter
from pathlib import Path
from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver
from compare_pri import score


def main():
    scenario = json.loads(Path("scenarios/pri_mode_change.json").read_text())
    pdws, truth = generate(scenario)
    tracker = StreamingDeinterleaver(bootstrap_pdws=600)
    for offset in range(0, len(pdws), 100):
        tracker.process_chunk(pdws[offset:offset + 100])

    print(f"Observed PDWs: {len(pdws)}")
    print("Overall:", score(truth, tracker.assignments))
    early = [p for p in pdws if p.toa_s < 0.35]
    mapping = {}
    for track_id in tracker.tracks:
        counts = Counter(truth[p.pdw_id] for p in early
                         if tracker.assignments.get(p.pdw_id) == track_id)
        if counts:
            mapping[track_id] = counts.most_common(1)[0][0]
    print("Track identity mapping from pre-switch window (evaluation only):", mapping)
    for start, end, label in [(0, 0.4, "Before switch"),
                              (0.4, 0.6, "High PRF"),
                              (0.6, 1.0, "After return")]:
        window = [p for p in pdws if start <= p.toa_s < end]
        print(f"\\n{label} [{start:.1f}, {end:.1f})")
        for emitter in ("A", "B", "C"):
            subset = [p for p in window if truth[p.pdw_id] == emitter]
            assigned = [p for p in subset if p.pdw_id in tracker.assignments]
            correct = [p for p in assigned
                       if mapping.get(tracker.assignments[p.pdw_id]) == emitter]
            print(f"  {emitter}: observed={len(subset)}, associated={len(assigned)}, "
                  f"correct_identity={len(correct)}, "
                  f"coverage={len(assigned)/len(subset):.1%}, "
                  f"identity_recall={len(correct)/len(subset):.1%}")
    print("\\nThis simulator still emits separate PDWs for overlapping pulses.")


if __name__ == "__main__":
    main()
