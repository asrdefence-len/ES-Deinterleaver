import unittest
from testbed.generator import PDW
from testbed.tracks import EmitterTrack, PRIModel, PRIType
from testbed.pri_discovery import difference_histogram, candidate_peaks


class PRIDiscoveryTests(unittest.TestCase):
    def test_fixed_pri_estimate_from_associated_intervals(self):
        model = PRIModel()
        model.update_from_intervals([0.001, 0.001001, 0.000999, 0.001])
        self.assertEqual(model.kind, PRIType.FIXED)
        self.assertAlmostEqual(model.mean_s, 0.001)

    def test_unknown_when_insufficient(self):
        model = PRIModel()
        model.update_from_intervals([0.001])
        self.assertEqual(model.kind, PRIType.UNKNOWN)

    def test_track_adds_pdws_in_order(self):
        track = EmitterTrack("T1")
        for i in range(3):
            track.add_pdw(PDW(i, i * 0.001, 9.4e9, 1e-5, 45, -30))
        self.assertEqual(track.associated_pdw_ids, [0, 1, 2])
        self.assertIsNone(track.pri_model.mean_s)

    def test_histogram_contains_repetition(self):
        pdws = [PDW(i, i * 0.001, 9.4e9, 1e-5, 45, -30) for i in range(10)]
        hist = difference_histogram(pdws)
        peaks = candidate_peaks(hist)
        self.assertTrue(any(abs(p.interval_s - 0.001) < 10e-6 for p in peaks))

    def test_bad_histogram_range(self):
        with self.assertRaises(ValueError):
            difference_histogram([], 0.003, 0.001)


if __name__ == "__main__":
    unittest.main()
