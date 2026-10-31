import unittest

import numpy as np
from scipy.stats import wasserstein_distance

from testbed.optimal_transport import BlobPair, score_blob_pairs, merge_blob_candidates


class OptimalTransportTests(unittest.TestCase):
    def test_histogram_distance_agrees_with_independent_scipy_reference(self):
        t = np.array([0., .12, .25, .33, .57, .89, 1.])
        labels = np.array([0, 1, 0, 1, 0, 1, 1])
        weights = np.array([1., 2., 4., 1., 3., 5., 1.])
        pair, = score_blob_pairs(t, labels, weights, amplitude_scale="linear", bins=20, min_pulses=2)
        # Independent reference on quantized time locations, not the CDF code.
        locations = (np.minimum((t * 20).astype(int), 19) + .5) / 20
        expected = wasserstein_distance(locations[labels == 0], locations[labels == 1],
            weights[labels == 0], weights[labels == 1])
        self.assertAlmostEqual(pair.wasserstein_fraction, expected)

    def test_common_clock_is_preserved_and_db_matches_linear_power(self):
        t = np.array([0., .1, .2, .7, .8, 1.])
        labels = np.array([0, 0, 0, 1, 1, 1])
        db = np.array([-60., -30., -50., -60., -30., -50.])
        p, = score_blob_pairs(t, labels, db, amplitude_scale="db", min_pulses=2)
        q, = score_blob_pairs(t + 5000, labels, 10**(db / 10), amplitude_scale="linear", min_pulses=2)
        self.assertAlmostEqual(p.wasserstein_fraction, q.wasserstein_fraction)
        self.assertGreater(p.wasserstein_fraction, .6)

    def test_circular_aoa_and_unknown_labels(self):
        t = np.tile(np.linspace(0, 1, 100), 2)
        labels = np.repeat([7, 9], 100)
        angles = np.repeat([359., 1.], 100)
        pairs = score_blob_pairs(t, labels, aoa_deg=angles)
        self.assertAlmostEqual(pairs[0].aoa_distance_deg, 2.)
        out = merge_blob_candidates(labels, pairs, aoa_gate_deg=3.)
        self.assertEqual(set(out), {7})
        labels[0] = -1
        out = merge_blob_candidates(labels, pairs, aoa_gate_deg=1.)
        self.assertEqual(out[0], -1)
        self.assertEqual(set(out), {-1, 7, 9})

    def test_sparse_or_zero_power_blobs_remain_unmerged(self):
        t = np.linspace(0, 1, 120)
        labels = np.repeat([0, 1], 60)
        pairs = score_blob_pairs(t, labels)
        self.assertEqual(pairs[0].reason, "sparse")
        self.assertTrue(np.array_equal(merge_blob_candidates(labels, pairs, max_distance=1), labels))
        pair, = score_blob_pairs(t, labels, np.zeros(120), amplitude_scale="linear", min_pulses=2)
        self.assertFalse(pair.eligible)
        self.assertEqual(pair.reason, "zero_weight")

    def test_complete_link_rejects_transitive_false_merge(self):
        pairs = [BlobPair(0, 1, 100, 100, .01, 0., True, "ok"),
                 BlobPair(1, 2, 100, 100, .02, 0., True, "ok"),
                 BlobPair(0, 2, 100, 100, .20, 0., True, "ok")]
        out = merge_blob_candidates(np.array([0, 1, 2, -1]), pairs, max_distance=.03)
        self.assertEqual(out.tolist(), [0, 0, 2, -1])

    def test_invalid_inputs_fail_instead_of_silently_scoring(self):
        with self.assertRaises(ValueError):
            score_blob_pairs([0, np.nan], [0, 1])
        with self.assertRaises(ValueError):
            score_blob_pairs([0, 1], [0, 1], [-1, 2], amplitude_scale="linear")
        with self.assertRaises(ValueError):
            score_blob_pairs([0, 0], [0, 1])
        self.assertEqual(score_blob_pairs([], np.array([], dtype=int)), [])


if __name__ == "__main__":
    unittest.main()
