# Batch RF density-step experiment

This experiment asks whether a histogram density-change test can separate
mixed RF blobs. It is not enabled in the grouping baseline.

## Run on macOS

From your ES-Deinterleaver/DEINT checkout, with your usual Python environment:

```bash
git pull --ff-only
python3 -m pip install -r requirements-experiments.txt
python3 -u trial_rf_density_split.py turing_data --output results/rf_density
```

The final command recursively discovers `.h5`, `.hdf5` and `.hd5` files.
Files stay local and are processed sequentially in 10,000-PDW windows.
No uploads occur. Results are checkpointed after each completed file.

For a quick initial comparison over five windows per file:

```bash
python3 -u trial_rf_density_split.py turing_data --windows 5 \
  --output results/rf_density_first5
```

To reproduce the window-40 example, or supply files individually:

```bash
python3 -u trial_rf_density_split.py turing_data/config_101.h5 \
  --start-window 40 --windows 1 --output results/rf_density_w40
```

## Which grouping is tested?

The provided `mixed_rank05_w40_g40.png` comes from **AOA + multiscale RF
grouping**, via `diagnose_rf_contamination.py`. That is the default here.
It is separate from AOA + RF/PW HDBSCAN. To test the latter instead:

```bash
python3 -u trial_rf_density_split.py turing_data --grouping hdbscan \
  --output results/rf_density_hdbscan
```

Both before and after use the same initial labels. No existing grouping
module is edited, and no pulse timing or OT is involved in this experiment.

## Split proposal

Each blob gets a frequency grid based on its own observed span, targeting
120 bins with rounded widths. For the roughly 9000–12000 example this yields
25-unit bins; the boundary is not hardcoded to 10000. Grid spacing and the
observed span use the recording's raw frequency units.

The model compares one constant frequency density with two constant
densities separated at a grid edge. An approximate BIC improvement greater
than 10 and at least 30 pulses in each child are required. This threshold
is a heuristic model comparison, not a calibrated confidence probability.
Each original blob is split at most once. Every PDW is retained; an original
unassigned label remains unassigned. IDs are local to each window.

Fit and split decisions use observations only. Truth is read afterward for
evaluation. A file without `labels` can run but cannot establish whether
the extra splits separate emitters correctly.

## Assess the result

- `summary.csv`: pooled per-file purity, contamination, group counts and
  fragmentation; mean window ARI and completeness; unchanged coverage.
- `windows.csv`: the same before/after measurements per window.
- `splits.csv`: each proposed boundary, child sizes and BIC improvement.
  Evaluation columns identify splits of a pure emitter and quantify any
  reduction in contamination and increase in emitter fragmentation.
- `config.json`: run parameters and resolved input filenames.

**Purity alone cannot decrease when an existing group is split.** A useful
rule must also preserve emitter completeness and avoid excessive extra
fragments. In particular, a pure emitter can have a nonuniform frequency
distribution; density changes do not establish separate emitter identities.
The constant-density null is therefore an exploratory assumption.

Compare `fragment_excess`, `mean_ari_grouped`, and
`mean_completeness_grouped`, not just purity. Window and group counts are
summed across independent windows; they are not persistent emitter tracks.
The known window-40 example is development evidence. Do not tune parameters
to make every labelled example separate, then treat those same examples as
independent validation.

The existing grouping commands continue to use the baseline. This density
split is applied only when you explicitly run `trial_rf_density_split.py`.
