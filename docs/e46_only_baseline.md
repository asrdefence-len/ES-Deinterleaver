# E46-only v1

This version isolates the E46 candidate family using the existing seeded
benchmark runner. E38 is not selected or sent to the tracker. The original
Config 101 recording is unchanged: grouping still sees the full recording,
and seed RF/AOA/PW measurements select the E46 development family without
using truth. This is a single-target tracking experiment, not a truth-filtered
single-emitter recording.

The one experimental change from the integrated two-family case is removing
E38 selection. AOA density + RF/PW HDBSCAN, the shared histogram PRI estimator,
Kalman filter and basic track management remain unchanged. The benchmark
tracker settings remain beam width 32, maximum active tracks 16, maximum gap
10 s, consolidation support 2 and consolidation gate sigma 4. Other tracker
parameters retain their previous values; all are stored in the run manifest.

The existing runner is reused to avoid another implementation drifting from
the supported pipeline. Its historical filename does not imply two families
must be selected.

```bash
python3 run_two_emitter_blob_tracker.py turing_data/config_101.h5 \
  --pipeline integrated --scenario e46_only \
  --output results/e46_only/v1/e46_only.json \
  --consistency-log results/e46_only/v1/consistency.jsonl
python3 plot_e46_only.py results/e46_only/v1/e46_only.json
```

The E46-only scenario has its own default output path and cannot overwrite the
two-emitter result by merely omitting `--output`. The overview plots original
primary confidence with missing states left blank, all canonical track
identities, actual PRI histogram inputs, and evaluation-only E46 pulse
coverage. Multiple IDs must be reported as fragmentation even when primary
confidence is high. Source truth is not supplied to association logic.

Acceptance for this isolation step: the dataset, runtime components, tracker
parameters and E46 input observations must match the preceding integrated
benchmark; the result must contain only the E46 selected family. This step
establishes an isolated reference and does not claim to fix fragmentation.

## Recorded v1 result

Clean code version `16b0f64`, full 85-window replay. All ten isolation checks
in [the result record](e46_only_baseline.json) pass. There are 433 observations,
301 with PRI histograms, 32 canonical track IDs and 135/433 observations on the
original E1 primary. Selection purity is 100% and E46 pulse selection coverage
is 93.43%. The primary is last present at approximately 5.91 s. These are
unresolved tracking/coverage limitations, not a performance improvement claim.
