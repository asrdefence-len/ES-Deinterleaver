"""Smoke tests and invariants for conservative ambiguity handling."""
import json
from pathlib import Path
from testbed.generator import generate
from testbed.phase_folding import PhaseFoldingDeinterleaver
from testbed.ambiguity_aware import AmbiguityAwareDeinterleaver


def test_empty():
    algo = AmbiguityAwareDeinterleaver()
    assert algo.process([]) == {}
    assert algo.ambiguous == []


def test_bad_gate():
    try:
        AmbiguityAwareDeinterleaver(gate_s=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_ambiguous_withheld_and_baseline_preserved():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, _ = generate(scenario)
    baseline = PhaseFoldingDeinterleaver().process(pdws)
    algo = AmbiguityAwareDeinterleaver()
    assigned = algo.process(pdws)
    ambiguous = {item.pdw_id for item in algo.ambiguous}
    assert set(assigned).isdisjoint(ambiguous)
    assert set(assigned).issubset(set(baseline))
    assert all(assigned[i] == baseline[i] for i in assigned)
    assert len(assigned) + len(algo.unassigned_pdw_ids) == len(pdws)
    assert sum(len(t.associated_pdw_ids) for t in algo.tracks) == len(assigned)
