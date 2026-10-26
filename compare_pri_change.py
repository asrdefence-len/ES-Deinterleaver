"""Four-arm controlled PRI change experiment, truth used only for scoring."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver
from testbed.mht_deinterleaver import MHTDeinterleaver
from testbed.pri_change_detector import PRIChangeDetector

PERIODS = [(0, .4, "before"), (.4, .6, "high_prf"), (.6, 1., "after")]


def run(path="scenarios/pri_mode_change.json"):
    pdws, truth = generate(json.loads(Path(path).read_text()))
    print(f"PDWs: {len(pdws)}")
    processors = [
        ("A fixed streaming", StreamingDeinterleaver(bootstrap_pdws=600)),
        ("B bounded MHT", MHTDeinterleaver(bootstrap_pdws=600)),
        ("C streaming + half-PRI", PRIChangeDetector(upstream="streaming")),
        ("D MHT + half-PRI", PRIChangeDetector(upstream="mht")),
    ]
    output = {}
    for name, model in processors:
        t0 = perf_counter()
        for i in range(0, len(pdws), 100):
            model.process_chunk(pdws[i:i+100])
        elapsed = perf_counter()-t0
        early = defaultdict(Counter)
        for p in pdws:
            if p.toa_s < .35 and p.pdw_id in model.assignments:
                early[model.assignments[p.pdw_id]][truth[p.pdw_id]] += 1
        mapping = {tid: c.most_common(1)[0][0] for tid, c in early.items()}
        print(f"\n{name}  runtime={elapsed:.3f}s  early_map={mapping}")
        print("period      radar  seen correct wrong withheld")
        rows = []
        for lo, hi, label in PERIODS:
            for emitter in ("A", "B", "C"):
                subset = [p for p in pdws if lo <= p.toa_s < hi
                          and truth[p.pdw_id] == emitter]
                assigned = [p for p in subset if p.pdw_id in model.assignments]
                correct = sum(mapping.get(model.assignments[p.pdw_id]) == emitter
                              for p in assigned)
                wrong = len(assigned)-correct
                withheld = len(subset)-len(assigned)
                print(f"{label:11} {emitter:5} {len(subset):5} "
                      f"{correct:7} {wrong:5} {withheld:8}")
                rows.append((label, emitter, len(subset), correct, wrong, withheld))
        if isinstance(model, PRIChangeDetector):
            print("Mode events:", model.mode_events)
            print("Pending:", len(model.pending_ids))
        output[name] = rows
    print("\nIMPORTANT: half-PRI is a targeted 2x-PRF detector, not an IMM.")
    print("Incorrect includes any train absent from the frozen early mapping.")
    print("The four arms do not have identical association decision rules.")
    return output


if __name__ == "__main__":
    run()
