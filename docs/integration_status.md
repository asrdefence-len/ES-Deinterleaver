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
