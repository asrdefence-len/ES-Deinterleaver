# Local Turing recordings

The experiments accept local files such as `turing_data/config_101.h5`.
Raw downloads are excluded from ordinary Git commits. The recordings were
not available in the workspace for the initial optimal transport trial.

Before a recording-based result is considered reproducible, record:

- Official dataset download URL and release/version (not yet supplied here).
- File names, byte sizes and SHA-256 hashes.
- Dataset redistribution terms, if files will be shared through Git LFS or
  another store.
- Feature names and units, particularly ToA and received amplitude.
- Window indices and size, algorithm configuration and code commit.

Use `shasum -a 256 turing_data/*.h5` on macOS to record checksums. Do not
invent a download URL or infer amplitude units from the feature name alone.

The HDF5 trial expects `data` and `metadata/feature_names`; `labels` is
optional and used only after association for evaluation. At minimum,
`ToA`, `Frequency` and `AoA` are needed by this particular trial. A named
amplitude feature is also required unless using uniform weights.

Example (replace `Amplitude` and `db` with the actual feature and units):

```bash
python trial_optimal_transport.py --file turing_data/config_101.h5 \
  --amplitude-feature Amplitude --amplitude-scale db \
  --start-window 0 --windows 5 --output results/ot_turing_101
```

For an unweighted temporal-distribution comparison, use
`--amplitude-scale uniform`. This omits received-level weighting and must
be identified as a different experiment. For dB power measurements, the
trial uses `10**(dB/10)`; linear measurements must already represent
nonnegative power or energy weights.
