import json
import unittest
from pathlib import Path
from sweep_frequency import sweep, optimal_one_to_one_accuracy


class SweepTests(unittest.TestCase):
    def test_permutation_invariance(self):
        truth = {1: "A", 2: "A", 3: "B", 4: "B"}
        assignments = {1: "T9", 2: "T9", 3: "T2", 4: "T2"}
        self.assertEqual(optimal_one_to_one_accuracy(truth, assignments), 1.0)

    def test_merge_penalty(self):
        truth = {1: "A", 2: "A", 3: "B", 4: "B"}
        assignments = {1: "T1", 2: "T1", 3: "T1", 4: "T1"}
        self.assertEqual(optimal_one_to_one_accuracy(truth, assignments), 0.5)

    def test_fragmentation_penalty(self):
        truth = {1: "A", 2: "A", 3: "A", 4: "A"}
        assignments = {1: "T1", 2: "T1", 3: "T2", 4: "T2"}
        self.assertEqual(optimal_one_to_one_accuracy(truth, assignments), 0.5)

    def test_sweep_reproducibility(self):
        scenario = json.loads(Path("scenarios/baseline.json").read_text())
        self.assertEqual(sweep(scenario, [0, 200_000], 2, 500_000),
                         sweep(scenario, [0, 200_000], 2, 500_000))


if __name__ == "__main__":
    unittest.main()
