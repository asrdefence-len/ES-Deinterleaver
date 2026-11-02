# Whole-blob observations as tracker input

This is the selected tracking path: analyze a completed candidate blob, export
one bounded summary per discovered pulse train, then associate those summaries
with emitter hypotheses. A blob with no supported train falls back to one
whole-blob summary. The tracker cannot see individual PDWs or modify blob or
train membership; timing analysis remains upstream of the tracker boundary.

## Run using the usual workflow

From DEINT on `main`, in your normal Python environment:

```bash
git pull
python3 -m pip install -r requirements-experiments.txt
python3 -u run_blob_tracker.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/blob_tracker_focus
```

`--measurement-mode train` is the default and exports PRI, PRI uncertainty,
phase epoch, timing RMS residual and pulse support. Use `--measurement-mode
blob` only for the earlier timing-blind comparison.

Default grouping is the existing AOA + multiscale RF path, matching the
window-local IDs in the earlier plots. To use the separate AOA + RF/PW
HDBSCAN implementation, specify `--grouping hdbscan` and use its group IDs.

To carry hypotheses through two consecutive complete windows:

```bash
python3 -u run_blob_tracker.py turing_data/config_101.h5 \
  --windows 2 --output results/blob_tracker_first2
```

Omit `--blobs` for multiwindow runs: blob IDs do not identify the same
emitter in subsequent windows. `--windows 0` processes all remaining windows.
The default 10,000-PDW window size introduces grouping-window latency.

For a persistent single-signal development run over the full Config 101
duration, use `run_long_blob_tracker.py`. It retains the same identity while
evidence is absent, predicts covariance forward and applies explicit expected
detection opportunities through long quiet intervals.

## Input boundary

`summarize_blob_trains` is a grouping-stage adapter. It visits the discovered
train membership once to export a frozen `BlobObservation` containing:

- Unique observation/window/blob reference, start/end time and pulse count.
- Circular AOA mean and angular spread.
- Frequency 5th percentile, median and 95th percentile.
- Pulse-width 5th percentile, median and 95th percentile.
- Optional train ID, PRI and uncertainty, phase epoch, timing residual and
  timing support count.

There may be multiple train summaries from a mixed blob. The upstream timing
stage performs that split without changing the original blob labels; label -1
is never promoted. Unexplained pulses remain explicitly counted. The caller
retains the original data, blob labels and train memberships. The tracker only
receives summaries—not arrays, PDW references or TOA sequences.

Default raw units are microseconds for TOA/PW, MHz for RF, degrees for AOA.
`--toa-scale`, `--frequency-scale`, and `--pw-scale` convert other raw units
to seconds/Hz. Angular spread and PW width are feature dispersion measures,
not calibrated measurement-uncertainty or association probabilities.

## Association hypotheses

`BlobEmitterTracker` maintains a bounded beam of whole-blob associations.
Compatibility uses time continuity, circular AOA and pulse-width summaries.
Each retained track now carries AOA, pulse-width and PRI covariance. Covariance is
predicted across gaps, converted into dynamic association gates and contracted
with Kalman updates when a compatible blob arrives. The available AOA, PW and
PRI innovations form a joint diagonal innovation covariance `S`; their NIS is
gated against a chi-square threshold at the configured probability.

Track confidence is the Bernoulli posterior probability that the emitter track
exists. An accepted measurement updates its odds using the Gaussian innovation
likelihood, detection probability, RF-mode probability and expected clutter per
gate. Every opportunity first predicts existence with `r- = P_S r`. A completed
opportunity with no accepted measurement then applies
`r+ = (1-P_D)r- / (1-P_D r-)`. Long time gaps insert expected opportunities at
the configured cadence. There is no fixed hit reward or arbitrary time-decay
confidence curve. `Q`, `R`, `P_S`, `P_D`, clutter rate and RF-mode probabilities
remain explicit model parameters that must be calibrated.
An existing RF mode has a 5-MHz interval-gap gate; a distant frequency is
considered as a possible new mode of the same emitter as well as a possible
new emitter. PRI harmonic compatibility is evaluated across factors one to
four. Phase continuity is mode-specific: different RF modes may occupy
different phase offsets, while a returning known RF mode is checked against
its previous phase. Overlapping observations are permitted on one emitter track.

Measurement covariance combines track prediction covariance with blob-derived
measurement variance and fixed sensor floors. The default maximum temporal gap
is 0.3 seconds. The new-emitter branch retains an explicit birth prior cost.
Frequency hopping is represented by discrete known-mode/new-mode probabilities,
not forced into the continuous Kalman state. Phase remains mode metadata and is
not yet included in the Gaussian state likelihood.

Retained hypotheses within relative negative-log cost 5 of the best are called plausible.
If they disagree about an observation's identity, its status is AMBIGUOUS.
Even agreement is only PROVISIONAL. Similar AOA/PW and activity spans do not
prove a common physical emitter; co-located independent sources can be
indistinguishable from these summaries alone. No truth labels enter costs,
gates, births or decisions. Transmitter configurations are not consulted.

## Bounded work and memory

Defaults cap the tracker at 16 hypotheses, 64 active tracks per hypothesis,
8 RF-mode summaries per track, 4 existing-track match branches per update,
256 recent associations and 16 recent blob references per track. RF-mode
memory keeps the most recently added mode summaries when full; an additional
mode does not itself force a new emitter. Expired tracks leave active state.
When active capacity is exhausted, an unresolved alternative is retained
rather than requiring an incompatible association.

Tracks below the configured existence-probability threshold are removed from active
association after a stale interval and retained in a bounded audit archive.
Tentative fragments can consolidate into a mature parent after repeated
covariance-gated AOA/PW agreement, shared RF support and PRI/harmonic
compatibility. Consolidation records an alias; it never changes blob or train
membership, and distant tracks remain separate.

Tracker input size and per-update workload are independent of the number
of pulses inside a blob. Grouping and summary export still process those
pulses once. Trace files can grow on disk while internal tracker state stays
bounded. Hypothesis/mode/history truncation loses information; it does not
guarantee that the true interpretation survives every difficult scenario.

## Outputs

| File | Contents |
|---|---|
| `blob_observations.csv` | The fixed-size input summaries |
| `blob_associations.csv` | Status and candidate track IDs at the end of each processed window |
| `hypotheses.jsonl` | Bounded hypothesis snapshots after each window |
| `blob_monitor.jsonl` | Confidence, covariance and gates after each blob update |
| `tracker_snapshot.json` | Final retained state and configured limits |
| `windows.csv` | Blob/PDW accounting and possible track counts for the selected observations |
| `blob_evaluation.csv` | Truth composition, read only after inference |
| `config.json` | Run settings |

Hypotheses may revise associations as evidence arrives. This changes an
emitter explanation, never the pulses belonging to an input blob. Previously
written rows describe the decision at their output time; snapshots provide
the later retained explanation. Observations older than the bounded history
are reported as OUTSIDE_RETAINED_HISTORY instead of inventing a decision.

For the focused three-blob test, check whether a common-emitter explanation
is retained alongside alternatives. Retention is not confirmation. A mixed
input blob is still one observation here: this tracker cannot repair its
internal contamination. Ambiguity may require later evidence or an explicit
operator diagnostic, not automatic reopening of PDWs.
