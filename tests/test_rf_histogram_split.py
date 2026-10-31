import unittest

import numpy as np

from diagnose_rf_histogram_split import fit_density_step


class RFHistogramSplitTests(unittest.TestCase):
    def test_detects_adjacent_density_bands_without_an_empty_gap(self):
        low = np.linspace(9000, 10000, 500, endpoint=False) + 1
        high = np.linspace(10000, 12000, 150, endpoint=False) + 1
        result = fit_density_step(np.r_[low, high], 9000, 12000)
        self.assertTrue(result["supported"])
        self.assertEqual(result["cut"], 10000)
        self.assertEqual((result["left_count"], result["right_count"]), (500, 150))

    def test_uniform_density_does_not_support_a_split(self):
        f = np.linspace(9000, 12000, 600, endpoint=False) + 2
        self.assertFalse(fit_density_step(f, 9000, 12000)["supported"])

    def test_all_observations_retained_including_display_underflow(self):
        f = np.r_[8999.9, np.linspace(9000, 9999, 500), np.linspace(10001, 11999, 150)]
        result = fit_density_step(f, 9000, 12000)
        self.assertEqual(result["left_count"] + result["right_count"], len(f))
        self.assertEqual(result["observation_min"], 8999.9)
        self.assertEqual(result["cut"], 10000)


if __name__ == "__main__":
    unittest.main()
