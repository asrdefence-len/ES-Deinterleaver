# Supported blob-tracking integration

The supported path is:

**PDWs → circular AOA density populations → RF/PW HDBSCAN → frozen blobs → local gap-PRI mean/spread/humps → bounded observation → Kalman tracker → basic track management.**

This is an integration baseline. It verifies which components run together;
it does not establish calibrated uncertainty or acceptable emitter retention.
See [integration status](integration_status.md) for acceptance checks and known failures.

## Entry points

```bash
python3 -m pip install -r requirements-experiments.txt
python3 run_blob_tracker.py turing_data/config_101.h5 \
  --windows 4 --output results/integrated_baseline/full_scene_first4
python3 run_two_emitter_blob_tracker.py turing_data/config_101.h5 \
  --scenario moderate --output results/integrated_baseline/moderate.json \
  --consistency-log results/integrated_baseline/moderate_consistency.jsonl
python3 run_live_blob_tracker.py turing_data/config_101_60s_gaps.h5 \
  --no-live --playback-rate 0 --output results/integrated_baseline/live
```

These entry points and `run_long_blob_tracker.py` default to the supported
integration. The general runner's defaults are `--grouping hdbscan` and
`--measurement-mode histogram`. The old combination requires explicit
`--grouping multiscale --measurement-mode train`; the seeded runners require
`--pipeline legacy`. Those alternatives are marked unsupported in the run
manifest. Other historical scripts remain experiments, not supported entry points.

Blob IDs belong to one grouping implementation and one window. The integrated
Config 101 two-emitter benchmark seeds E46 from Window 0 / HDBSCAN blobs 10–33,
E38 from Window 3 / blobs 14–25, or E33 from Window 0 / blobs 128–129. These are
controlled development selections; the tracker never reads truth to associate
measurements. Old multiscale blob IDs must not be reused with HDBSCAN.
The seeds retain RF and PW exemplars; later candidate selection is the union
of their measurement envelopes. HDBSCAN's RF/PW split changes the observations,
so measurement-retention counts cannot be compared directly with the legacy
fixed-train counts.

## Grouping

`compare_aoa_batch.batch_aoa_populations` estimates circular AOA-density peaks
in independent 10,000-PDW windows. `compare_aoa_batch.rf_pw_groups` then applies
`sklearn.cluster.HDBSCAN` to the existing scaled RF/PW features within each
AOA population (`min_cluster_size=8`, `min_samples=8`). Noise label -1 is not
promoted into a measurement. Window-local labels are frozen before timing
analysis and are never edited by the tracker.

This path can split different PW modes or parts of a broad RF distribution
into separate blobs. It is not a guarantee that each blob or each track
corresponds to one complete physical emitter.

## PRI estimator and input boundary

`testbed.pri_histogram.measure` is the original one/two-neighbouring-hump
algorithm moved out of `compare_two_humps.py`. The diagnostic imports that
same implementation, preventing a second copy from drifting. Defaults remain
2 µs bins, gaps from 10 to 400 µs, an optional second peak within 15% of the
primary peak, and contiguous half-height support. The general runner exposes
those settings as `--pri-bin-us`, `--pri-min-gap-us`, `--pri-max-gap-us` and
`--pri-neighbourhood`.

`testbed.blob_observation.summarize_blob_pri` exports one frozen observation
per blob, containing circular AOA mean/spread, RF and PW quantiles, activity
times, pulse count and, when supported, this bounded `GapPRI` summary:

- Selected observed-gap mean and sample standard deviation, in seconds.
- One or two peak centres, selected/in-range gap counts and estimator settings.
- Initial measurement sigma: `max(selected-gap SD, bin_width/sqrt(12), 0.05 µs)`.
  The tracker also applies its existing 1 µs PRI measurement floor.

The last rule is an explicit, conservative initial dispersion/binning proxy.
It is **not a calibrated standard error**; support does not shrink it by
`sqrt(N)`. Peak locations and the selected-bin SD are not proof of a true
fundamental PRI or an unbiased jitter distribution. Missing pulses, per-RF
sampling, PW splits and histogram truncation affect these statistics. The
original 10 µs lower limit can exclude short jittered gaps; it remains an
exposed estimator limitation, not a silently retuned parameter.

No fixed pulse-train extraction or phase estimate is inserted into the default
path. Unsupported timing produces an AOA/RF/PW observation with PRI absent.
Multiple humps are retained as metadata; they are not interpreted as separate
emitters. The tracker consumes only bounded summaries, never pulse lists,
truth IDs or TOA sequences. Default raw units are µs for TOA/PW, MHz for RF and
degrees for AOA; the adapter converts to SI.

## Kalman state and association

`BlobEmitterTracker` predicts diagonal random-walk covariance for circular
AOA, pulse width and available PRI. An observation forms innovation `ν` and
`S = P⁻ + R`; joint NIS is compared with the configured chi-square gate.
Accepted branches update the state with `K = P⁻ / S` and contract covariance.

Histogram inputs compare their selected-gap means **directly**, with harmonic
factors 1:1. Fixed-train harmonics remain available only on the explicit legacy
path. A fixed-train PRI and a histogram-gap mean are different statistics:
when changing models, no mixed-statistic PRI innovation is scored and the PRI
state is reinitialized. Track consolidation likewise does not harmonic-scale
histogram means. The last bounded histogram summary survives monitoring and
archival, separate from the filtered state estimate.

The bounded association beam retains known RF mode, new mode, birth and
capacity-unresolved explanations. Relative beam weights are not calibrated
probabilities. The optional consistency stream labels incoming hypothesis
rank; an `accepted` record means an update branch was constructed before beam
pruning, not necessarily that it became the winning association.

## Basic track management

The same tracker performs birth, Bernoulli existence updates, missed-opportunity
updates, confirmed/coasting/lost monitoring, stale-track archival, capacity
limits and fragment consolidation with aliases. IDs are never reused.
`track_monitor` exposes `lifecycle_status`; a new track is tentative, confidence
≥0.8 is confirmed, below 0.3 is lost, and other established tracks are coasting.
Archived records retain identity and last-state metadata but do not participate
in association. Dormant identity memory and archive reacquisition are not implemented.

Default archive policy remains existence below 0.3 and stale at least 0.5 s,
or expiry beyond `max_gap_s`. The general runner defaults to a 0.3 s maximum
gap; the seeded development runners use 10 s. These settings and every other
numeric tracker parameter are recorded. Detection opportunities, survival,
detection and clutter priors remain nominal; the integration does not retune
them or assert that their persistence policy is suitable for ELINT.

## Outputs and provenance

The general runner writes observation/association/evaluation CSVs, window
accounting, `blob_monitor.jsonl`, hypothesis history, final snapshot and
`config.json`. `--consistency-log` enables candidate-level diagnostics.
`run_manifest.json` identifies component functions, grouping/PRI/tracker
settings, Git revision/branch/dirty state, source-file hashes, dependency
versions and the input dataset SHA-256. Seeded runners embed this manifest in
their JSON result. Live playback writes it beside its history and snapshot.

A result produced from a dirty tree is identifiable by its source hashes; a
release benchmark should be run from a clean committed tree. Results/data are
kept outside Git. A compact, versioned benchmark record preserves the metrics,
input identity, code identity and local artifact locations.
