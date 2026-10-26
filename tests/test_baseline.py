import json
import unittest
from pathlib import Path
from testbed.generator import generate
from testbed.deinterleavers import FrequencyGateDeinterleaver
from testbed.evaluation import evaluate

class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = json.loads(Path("scenarios/baseline.json").read_text())

    def test_reproducible_and_no_truth_in_pdw(self):
        pdws, truth = generate(self.scenario)
        pdws2, truth2 = generate(self.scenario)
        self.assertEqual((pdws, truth), (pdws2, truth2))
        self.assertFalse(hasattr(pdws[0], "true_emitter"))

    def test_baseline_association(self):
        pdws, truth = generate(self.scenario)
        assignments = FrequencyGateDeinterleaver().process(pdws)
        result = evaluate(truth, assignments)
        self.assertGreater(result["track_purity"], 0.95)
        self.assertEqual(result["true_emitters_observed"], 3)

    def test_invalid_pri(self):
        scenario = json.loads(json.dumps(self.scenario))
        scenario["emitters"][0]["pri_s"] = 0
        with self.assertRaises(ValueError):
            generate(scenario)

if __name__ == "__main__":
    unittest.main()
