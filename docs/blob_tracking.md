# Whole-blob observations as tracker input

This is the selected tracking path: summarize a completed candidate blob,
then associate that observation with emitter hypotheses. The tracker cannot
see individual PDWs, discover PRI trains, or modify blob membership. The
earlier pulse-association pipeline remains an optional historical diagnostic;
it is not invoked by this path.

## Run using the usual workflow

From DEINT on `main`, in your normal Python environment:

```bash
git pull
python3 -m pip install -r requirements-experiments.txt
python3 -u run_blob_tracker.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/blob_tracker_focus
```

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

## Input boundary

`summarize_blobs` is a grouping-stage adapter. It visits the members of each
existing blob once to export a frozen `BlobObservation` containing:

- Unique observation/window/blob reference, start/end time and pulse count.
- Circular AOA mean and angular spread.
- Frequency 5th percentile, median and 95th percentile.
- Pulse-width 5th percentile, median and 95th percentile.

There is one observation per existing blob. No split, merge of memberships,
PRI fitting or selection of a subset of pulses occurs. Original label -1
is not promoted to a blob. The caller retains the original data and labels.
The tracker only receives the summaries, not the array or member references.
The runner checks that the original label array remains unchanged.

Default raw units are microseconds for TOA/PW, MHz for RF, degrees for AOA.
`--toa-scale`, `--frequency-scale`, and `--pw-scale` convert other raw units
to seconds/Hz. Angular spread and PW width are feature dispersion measures,
not calibrated measurement-uncertainty or association probabilities.

## Association hypotheses

`BlobEmitterTracker` maintains a bounded beam of whole-blob associations.
Compatibility uses time continuity, circular AOA and pulse-width summaries.
An existing RF mode has a 5-MHz interval-gap gate; a distant frequency is
considered as a possible new mode of the same emitter as well as a possible
new emitter. Overlapping observations are permitted on one emitter track.

The feature cost is half the sum of squared scaled AOA/PW differences.
Scales combine observed spreads with fixed floors (0.5 degrees and 0.05 us).
Each feature is gated at four scale units. The default maximum temporal gap
is 0.3 seconds. New emitter and new RF mode explanations both incur cost 3,
deliberately leaving the two alternatives competitive when other features
agree. These are experimental heuristic costs, not fitted likelihoods.

Retained hypotheses within relative cost 2 of the best are called plausible.
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
