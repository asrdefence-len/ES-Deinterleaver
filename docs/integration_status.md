# Integration status

## Parallel experiment: sequential raw-PDW tracker

Work on `codex/sequential-pdw-tracker` introduces a separate experimental path:
raw time-ordered PDWs enter a causal probabilistic multi-emitter tracker without
AOA grouping, HDBSCAN, blobs or `BlobEmitterTracker`. This does not replace or
recalibrate the integrated E46-only baseline below. See
[`sequential_pdw_tracker.md`](sequential_pdw_tracker.md) for its explicit model,
dataset boundary, limitations and local run procedure. Performance acceptance
is open until a clean Config 101 replay is supplied and reviewed.

The v0 implementation and its focused tests pass; the full repository result
at its initial checkpoint was **243 passed**. A 500-PDW wiring smoke test used the explicitly different
`config_101_60s_gaps.h5`: it reached the configured 64-track limit, assigned
130 pulses and left 306 as clutter. This is recorded as an unaccepted smoke
result, not silently treated as the Config 101 benchmark. The next single
investigation was the user's 5,000-PDW run on the canonical `config_101.h5`.

That canonical replay used dataset SHA-256
`fae9e2a82108ca97c85b5a3501e3a4f1798493f30f7e3eb0cc961d1b537ff8ec`
and commit `90fdafe`. It created 64 tracks, assigned 1,010 later pulses and
rejected 3,926; including births, coverage was 1,074/5,000 (21.48%) with 99.81%
selected association purity. The cap was reached by row 94, the births covered
only 22 of the 45 truth emitters present, and all 64 tracks had one RF mode.
Posteriors show that 3,880 rejected pulses preferred a new emitter but could not
be born at capacity. The default unseen-mode score could not exceed the equal
0.08 birth prior after other likelihoods were applied, so the advertised RF-hop
path was structurally unable to win. The next isolated change lowers only the
birth prior to 0.02 and adds a default-configuration hopping regression; all
other association and state settings remain frozen for the matched rerun.

The matched 0.02-birth-prior replay at commit `13c5c76` raised coverage from
21.48% to 49.64% (2,482/5,000 including births), retained 99.88% selected
association purity, and produced multiple RF modes in 45/64 tracks. This is a
real improvement in the intended hop path, not performance acceptance: the cap
still filled by row 128, only 24 truth emitters received births, and 2,364
rejections preferred a new emitter. Analysis of all 40 duplicate births found
timing as the weakest component in 16, PW in 14 and RF in 10. The next single
change delays use of the online PRI likelihood until three assigned gaps support
it; all other components and settings remain fixed for another matched replay.

## Milestone: restore the intended processing chain

The experimental state was preserved at Git checkpoint `1465b5b`.
Integration work is on `codex/integrated-aoa-hdbscan-pri-tracker`.
No changes have been pushed to GitHub.

| Component | Supported integration | Acceptance evidence |
|---|---|---|
| AOA + HDBSCAN | Circular AOA populations followed by existing RF/PW HDBSCAN (8/8) | Default-runner test forbids entering legacy multiscale grouping |
| PRI estimator | Shared original local mean/spread and optional second hump | Known two-hump test; diagnostic and adapter share one implementation |
| Observation adapter | One bounded summary per frozen blob, or PRI-absent fallback | Membership, metadata and unit checks; no fixed-train/phase fabrication |
| Kalman tracker | AOA/PW/gap-mean prediction, covariance, joint NIS and update | Analytical mean/covariance check; no harmonic reinterpretation of gap means |
| Basic management | Birth, existence, coast, lost/stale archive, ID continuity and limits | Missed-opportunity, covariance growth, archival and non-reused-ID tests |
| Truth isolation | Truth used for evaluation only | Relabelling truth leaves complete tracking snapshots unchanged |
| Configuration control | Run manifests and Git milestones | Component names, settings, code/data hashes and dependency versions saved |

## Performance acceptance remains open

Connecting the requested components does not establish calibration or complete
emitter tracking. The full Config 101 E46/E38 integration replay exposes severe
fragmentation, especially for E38. This is recorded as a failed retention
benchmark, not described as successful tracking.

RF/PW HDBSCAN splits E46's distinct PW populations and partitions E38's broad RF
support. Observed-gap distributions can therefore differ between blobs from
the same emitter. One scalar gap-mean state does not automatically represent
those different sampling patterns. Sparse blobs may have no PRI estimate at
all. Both source emitters are configured with jittered timing. These are known
limits to investigate with the now-explicit chain; do not silently substitute
another grouping method or inflate gates to make retention appear better.

The general runner's default maximum gap is 0.3 s; seeded development runners
use 10 s. Their Bernoulli detection/lifecycle parameters are nominal. Dormant
identity memory and archived-track reacquisition remain outside this milestone.

## Working process

For each change, state which component changes and its acceptance criterion.
Run the relevant integration/regression checks, compare a recording and a
controlled fixture against the saved baseline, and commit the result.
Record implemented, integrated and performance-validated status separately.
Update the supported-path documentation and manifests whenever a component or
its measurement meaning changes. Historical scripts are experiments unless
listed as supported entry points in `blob_tracking.md`.

## Recorded baseline — 2 November 2026

Code revision: `4c3d5be` (clean working tree during all three replays).
Regression result: **239 passed**, including five integration tests.
The machine-readable [baseline record](integration_baseline.json) contains the
input/code hashes, dependency versions, artifact checksums and numerical results.

| Replay | Recorded result |
|---|---|
| All grouped data, first four Config 101 windows | 37,782 grouped PDWs; 457 whole-blob observations; 235 carry a PRI histogram; membership preserved |
| E46/E38 controlled families, all 85 windows | 586 blob observations; 348 carry a PRI histogram; zero canonical tracks contain both target truth labels |
| E46 single-family, all 85 windows | Runner completes; original primary has 139 updates and is absent at the recording end |

The two-family replay creates **33 canonical track IDs touching E46 pulses and
22 touching E38 pulses**. These are fragments, not counts of physical emitters.
Only 142/433 E46 observations and 6/153 E38 observations remain on the respective
original primary IDs after consolidation. The denominators are whole HDBSCAN
blobs, unlike the legacy fixed-train benchmark, so the old retention fractions
are not directly comparable.

Selection coverage is 93.43% for E46 and 64.50% for E38; selected-pulse purity is
100% and 99.66%, respectively. E38 starts from its Window 3 seed, whereas the
coverage denominator includes the entire recording. These metrics describe
this controlled selection and do not establish whole-scene detection quality.

Integration acceptance is recorded separately from the still-open emitter
retention and calibration acceptance. The next performance investigation is
identity fragmentation across RF/PW blobs and their differently sampled gap
statistics, using this saved configuration as the comparison point.

To reproduce the baseline plots:

```bash
python3 plot_integration_baseline.py results/integrated_baseline/moderate.json
```

`identity_fragmentation.png` shows all canonical IDs associated with each
truth emitter's pulses, including fragments outside the original primary.
`pri_histogram_inputs.png` shows the actual blob-level selected-gap mean,
standard deviation, hump centres and selected-gap support. These gap statistics
must not be interpreted as transmitter fundamental PRI.

## Confidence view and handoff correction

The integrated path is a connected **experimental** path, not an accepted
performance replacement for the earlier run. Grouping and PRI representation
changed together during the initial integration. That replay cannot attribute
a downstream regression to either individual change. Future comparisons must
change one stage at a time with matched inputs and explicit measurement accounting.

The confidence calculation remains in `BlobEmitterTracker`; the fragmentation
plots did not display it. The original E38 primary in the integrated replay is
last present at about 0.67 s. An absent primary is not a zero-confidence value,
and selecting a different surviving fragment would change the plotted identity.
The earlier saved confidence reference remains available locally at
`results/moderate_two_emitter_blob_tracker.json` (E38 56/75 retention).

```bash
python3 plot_confidence_comparison.py \
  results/moderate_two_emitter_blob_tracker.json \
  results/integrated_baseline/moderate.json
```

This restores both recorded confidence views with explicit processing-path
labels; it is not a like-for-like validation comparison. `AGENTS.md` records
project-level handoff instructions so future work starts from these files,
versions and known failures. The next investigation should isolate measurement
changes between grouping variants before changing filter/lifecycle parameters.

## Active version: E46-only v1

The user has set E38 aside to establish E46 first. The isolated version is on
`codex/e46-only-baseline`; [its run instructions](e46_only_baseline.md) and
[machine-readable record](e46_only_baseline.json) identify the result and checks.

The clean replay used code revision `16b0f64`. Only E46's candidate family
enters the tracker; grouping still sees the unchanged full Config 101 input.
Verification confirms the same dataset, grouping/PRI settings, runtime component
sources, E46 seed, all 433 E46 observation inputs and tracker parameters as the
preceding integrated two-family replay. No filter or lifecycle tuning occurred.
Targeted integration/tracker tests: **36 passed**.

The replay produces 32 canonical identities and retains 135/433 observations on
original primary E1. Selection purity is 100%, with 93.43% of labelled E46
pulses selected. The original primary is last present at about 5.91 s; the
last selected observation is at about 5.44 s. E46 truth pulses still occur in
the final recording window, so the late absence must not be described as a
verified emitter shutdown. Fragmentation and missed sparse coverage remain
open issues. Removing E38 changes beam competition, so identity assignments
may differ even though the E46 measurement inputs are identical.

`results/e46_only/v1/e46_overview.png` keeps confidence, every assigned identity,
PRI input statistics and pulse coverage together. Previous two-emitter results
remain untouched. Do not compare against legacy grouping as a success target;
continue investigation of this isolated, intended path one change at a time.
