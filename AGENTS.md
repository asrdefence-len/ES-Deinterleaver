# S2B project handoff and integration rules

## Start of each coding task or new chat

Read `docs/integration_status.md`, `docs/blob_tracking.md`, and
`docs/integration_baseline.json`. Inspect the current branch, revision and
working changes. State the active processing path before modifying it. Chat
history alone is not a component inventory or evidence of performance.

## Current state and boundaries

The user's intended architecture is AOA density + RF/PW HDBSCAN, the shared
local gap mean/spread/one-or-two-hump estimator, bounded blob summaries, the
Kalman `BlobEmitterTracker`, and basic Bernoulli track management.

The four integrated blob runners default to that path. Its integration tests
pass, but it has NOT passed emitter-retention acceptance. In the recorded
Config 101 replay it creates 33 E46 and 22 E38 fragments. Do not describe a
connected or test-passing component as a validated tracking improvement.

Git checkpoint `1465b5b` preserves earlier experiments. The earlier saved
confidence reference is `results/moderate_two_emitter_blob_tracker.json`
(multiscale RF + fixed-train PRI, E38 primary retention 56/75). The current
experimental replay is `results/integrated_baseline/moderate.json` (HDBSCAN +
gap histogram). Result files are local/ignored; their paths do not guarantee
availability in another checkout. The versioned baseline record identifies
code, data and artifact hashes. Never silently substitute one result/path.

## Change and verification discipline

- Preserve existing work. Maintain local Git checkpoints and commits for the
  user; do not require them to operate Git. Do not push without authorization.
- Freeze the component choices during calibration. Do not change grouping,
  PRI estimator, tracker model and lifecycle policy together to improve a score.
- For a requested component change, compare one stage at a time against the
  same saved input. Identify differences in measurements before attributing
  downstream changes to the filter. Changed denominators are not comparable
  retention scores.
- Run relevant integration checks and benchmark comparisons. Passing unit
  tests alone is not performance acceptance. Record regressions explicitly.
- Show track existence/confidence, all identity fragments and measurement
  coverage together. Missing primary tracks must remain visibly missing, not
  be interpolated into reassuring histories or plotted as zero probability.
- Treat selected observed-gap mean, gap dispersion and transmitter fundamental
  PRI as different quantities. Preserve estimator metadata and SI units.
  Never use source truth/configuration to select algorithm decisions.
- Every benchmark must identify active component functions/settings, code
  revision/dirty state, source and dataset hashes, and dependency versions.
- Update the integration-status record at milestones with what changed, what
  passed, what failed, the active version, and the next single investigation.

## Active focus: E46-only isolation

The current user direction is to set E38 aside and get E46 right through the
intended integrated pipeline. Use `--pipeline integrated --scenario e46_only`
in the shared seeded runner and `docs/e46_only_baseline.md`. Do not return to
legacy grouping or use its reassuring confidence plots as an acceptance target.
`results/e46_only/v1/e46_only.json` is the isolated reference. Compare its input
isolation and settings only against the preceding integrated E46/E38 replay;
do not tune the filter, estimator or lifecycle merely to make plots look better.
