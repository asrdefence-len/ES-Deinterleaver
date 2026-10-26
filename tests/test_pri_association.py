import json
import unittest
from pathlib import Path
from testbed.generator import generate
from testbed.pri_association import PRIGuidedDeinterleaver
from compare_pri import score

class PRIAssociationTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(PRIGuidedDeinterleaver().process([]),{})
    def test_single_emitter(self):
        scenario=json.loads(Path("scenarios/same_frequency.json").read_text())
        scenario["emitters"]=scenario["emitters"][:1]
        scenario["duration_s"]=0.1
        pdws,truth=generate(scenario)
        assignments=PRIGuidedDeinterleaver(min_train_length=10).process(pdws)
        self.assertTrue(set(assignments).issubset(truth))
        self.assertGreater(len(assignments),20)
    def test_unassigned_penalty(self):
        result=score({1:"A",2:"A",3:"B",4:"B"},{1:"T1",2:"T1"})
        self.assertEqual(result["coverage"],0.5)
        self.assertEqual(result["one_to_one_accuracy"],0.5)
