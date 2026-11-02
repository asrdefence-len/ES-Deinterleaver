# Integration status

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
