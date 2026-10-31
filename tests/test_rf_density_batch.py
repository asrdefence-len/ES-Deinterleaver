from pathlib import Path
import tempfile
import unittest

import numpy as np

from trial_rf_density_split import automatic_step, discover_files, split_blobs, evaluate_splits


class RFDensityBatchTests(unittest.TestCase):
    @staticmethod
    def example(scale=1., offset=0.):
        return (np.r_[np.linspace(9000, 10000, 500, endpoint=False) + 1,
                       np.linspace(10000, 12000, 150, endpoint=False) + 1] * scale + offset)

    def test_automatic_grid_finds_boundary_without_fixed_frequency(self):
        f = self.example()
        r = automatic_step(f)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["cut"], 10000)
        transformed = automatic_step(self.example(1000., 2e8))
        self.assertTrue(transformed["accepted"])
        self.assertEqual(transformed["cut"], 210000000)

    def test_split_retains_unassigned_pdws_and_other_blobs(self):
        f = np.r_[self.example(), 5., 7.]
        labels = np.r_[np.full(650, 4), -1, 8]
        after, diagnostics = split_blobs(f, labels)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(after[-2:].tolist(), [-1, 8])
        self.assertEqual(int((after >= 0).sum()), int((labels >= 0).sum()))
        self.assertEqual(set(after[:500]), {4})
        self.assertEqual(set(after[500:650]), {9})

    def test_uniform_band_stays_one_group(self):
        f = np.linspace(9000, 12000, 600, endpoint=False) + 2
        self.assertFalse(automatic_step(f)["accepted"])

    def test_audit_exposes_pure_emitter_fragmentation(self):
        f = self.example()
        before = np.zeros(len(f), dtype=int)
        after, diagnostics = split_blobs(f, before)
        audit, = evaluate_splits(diagnostics, before, after, np.full(len(f), 17))
        self.assertTrue(audit["pure_parent_split_eval"])
        self.assertEqual(audit["minority_reduction_eval"], 0)
        self.assertEqual(audit["additional_fragments_eval"], 1)

    def test_directory_discovery_accepts_hdf5_extensions_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            for name in ("a.h5", "b.hdf5", "nested/c.hd5", "ignore.txt"):
                (root / name).touch()
            files = discover_files([tmp, str(root / "a.h5")])
            self.assertEqual(len(files), 3)
            self.assertEqual({p.suffix for p in files}, {".h5", ".hdf5", ".hd5"})


if __name__ == "__main__":
    unittest.main()
