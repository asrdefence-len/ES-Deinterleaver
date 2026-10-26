"""Compare frequency-only with experimental PRI extraction; unmatched pulses count as errors."""
import json
import time
from pathlib import Path
from testbed.generator import generate
from testbed.deinterleavers import FrequencyGateDeinterleaver
from testbed.pri_association import PRIGuidedDeinterleaver
from sweep_frequency import optimal_one_to_one_accuracy

def score(truth, assignments):
    if not truth: return {"coverage":0.0,"one_to_one_accuracy":0.0,"tracks":0}
    subset={i:truth[i] for i in assignments}
    accuracy=optimal_one_to_one_accuracy(subset,assignments)*len(subset)/len(truth) if subset else 0
    return {"coverage":len(subset)/len(truth),"one_to_one_accuracy":accuracy,"tracks":len(set(assignments.values()))}

def main():
    scenario=json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws,truth=generate(scenario)
    for name,algo in [("Frequency only",FrequencyGateDeinterleaver()),("PRI guided (experimental)",PRIGuidedDeinterleaver())]:
        start=time.perf_counter()
        assignments=algo.process(pdws)
        print(name, json.dumps({**score(truth,assignments),"runtime_s":round(time.perf_counter()-start,3)},indent=2))
        if isinstance(algo,PRIGuidedDeinterleaver):
            for track in algo.tracks:
                print(track.track_id,"PRI us:",round(track.pri_model.mean_s*1e6,3),"pulses:",len(track.associated_pdw_ids))

if __name__=="__main__":
    main()
