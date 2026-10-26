"""Piecewise PRI simulator regression checks."""
import json
from pathlib import Path
from testbed.generator import generate


def test_mode_change_adds_pulses_without_truth_leakage():
    scenario = json.loads(Path("scenarios/pri_mode_change.json").read_text())
    pdws, truth = generate(scenario)
    assert len(pdws) > 2900
    assert len(pdws) == len(truth)
    assert all(not hasattr(p, "emitter_id") for p in pdws)
    assert [p.toa_s for p in pdws] == sorted(p.toa_s for p in pdws)


def test_mode_change_is_reproducible():
    scenario = json.loads(Path("scenarios/pri_mode_change.json").read_text())
    assert generate(scenario) == generate(scenario)
