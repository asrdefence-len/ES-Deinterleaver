"""Smoke tests for the experimental bounded hypothesis stage."""
import json
from pathlib import Path
from testbed.generator import generate
from testbed.bounded_mht import BoundedMHTDeinterleaver


def test_empty_input():
    algorithm = BoundedMHTDeinterleaver()
    assert algorithm.process([]) == {}
    assert algorithm.hypotheses_retained == 0


def test_hypothesis_limit_validation():
    try:
        BoundedMHTDeinterleaver(max_hypotheses=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_association_invariants():
    scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
    pdws, _ = generate(scenario)
    algorithm = BoundedMHTDeinterleaver(max_hypotheses=4)
    assignments = algorithm.process(pdws)
    assert set(assignments).issubset({p.pdw_id for p in pdws})
    assert set(assignments.values()).issubset({t.track_id for t in algorithm.tracks})
    assert algorithm.hypotheses_retained <= 4
    assert sum(len(t.associated_pdw_ids) for t in algorithm.tracks) == len(assignments)
