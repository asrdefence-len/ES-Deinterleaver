import json
import unittest
from pathlib import Path
from testbed.generator import generate
from testbed.phase_folding import PhaseFoldingDeinterleaver
from compare_pri import score


class PhaseFoldingTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(PhaseFoldingDeinterleaver().process([]), {})

    def test_single_fixed_emitter(self):
        scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
        scenario["emitters"] = scenario["emitters"][:1]
        scenario["duration_s"] = 0.15
        pdws, truth = generate(scenario)
        algorithm = PhaseFoldingDeinterleaver(min_track_pdws=20)
        result = algorithm.process(pdws)
        self.assertGreater(score(truth, result)["coverage"], 0.5)

    def test_no_truth_leakage(self):
        scenario = json.loads(Path("scenarios/same_frequency.json").read_text())
        pdws, _ = generate(scenario)
        self.assertFalse(hasattr(pdws[0], "true_emitter"))
