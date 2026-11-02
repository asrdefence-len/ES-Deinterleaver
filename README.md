# ES Deinterleaver Testbed

Standalone, truth-scored experimental testbed for emitter deinterleaving. Independent of S2B-ESM.

## First milestone
- Deterministic multi-emitter PDW scenario generator (fixed RF/PRI/PW/AOA with measurement noise and missed pulses).
- Ground truth held separately from the observations supplied to the algorithm.
- Replaceable frequency-gated baseline deinterleaver.
- Permutation-invariant scoring of emitter assignments, purity, completeness and fragmentation.
- CSV exports and a simple truth-versus-estimate plot.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 run_experiment.py --scenario scenarios/baseline.json --output results/baseline
python3 -m unittest discover -s tests -v
```

The output folder contains `observations.csv`, `truth.csv`, `assignments.csv`, `metrics.json`, and `assignments.png`.

**Important:** The baseline is deliberately simple: it does not solve PRI ambiguity, frequency hopping, overlapping IQ pulses, or waveform-assisted association. AOA is simulated, not measured by hardware. Scoring uses truth only after deinterleaving.

## Whole-blob emitter tracking

The [blob-derived tracker](docs/blob_tracking.md) consumes bounded train
summaries produced inside fixed-membership blobs. It associates observations
without exposing individual pulses to the tracker and retains ambiguity.

```bash
python3 -m pip install -r requirements-experiments.txt
python3 run_blob_tracker.py turing_data/config_101.h5 \
  --window 0 --blobs 9 11 12 --output results/blob_tracker_focus
python3 plot_blob_tracker_monitor.py \
  results/blob_tracker_focus/blob_monitor.jsonl
```

This is the primary tracker path. Each frozen blob is analyzed upstream and
exports one bounded summary per supported pulse train. PRI, harmonic and
mode-specific phase evidence contribute to association. Covariance, dynamic
gates and confidence update once per summary; raw-PDW gating is reserved for
optional sparse recovery and is not the normal tracker input.

Run the seeded E46 development track across the complete Config 101 recording,
including explicit 0.1-second coast monitoring through the long final gap:

```bash
python3 run_long_blob_tracker.py turing_data/config_101.h5 \
  --output results/long_blob_tracker_e46.json
python3 plot_long_blob_tracker.py results/long_blob_tracker_e46.json \
  --output results/long_blob_tracker_e46.png
```

The seed RF/AOA/PW envelope selects later candidate blobs without using truth.
Truth is read only after tracking for coverage and purity evaluation.

Run the first two-real-emitter baseline with E46 and the clearly separated
Config 101 E33 signal:

```bash
python3 run_two_emitter_blob_tracker.py turing_data/config_101.h5 \
  --output results/two_emitter_blob_tracker.json
python3 plot_two_emitter_blob_tracker.py \
  results/two_emitter_blob_tracker.json \
  --output results/two_emitter_blob_tracker.png
```

Run the moderate-separation scenario with E46 and real Config 101 E38. E38
is seeded from its clean Window 3 / Blob 6 observation rather than the mixed
Window 0 group:

```bash
python3 run_two_emitter_blob_tracker.py turing_data/config_101.h5 \
  --scenario moderate \
  --output results/moderate_two_emitter_blob_tracker.json
python3 plot_two_emitter_blob_tracker.py \
  results/moderate_two_emitter_blob_tracker.json \
  --output results/moderate_two_emitter_blob_tracker.png
```

## Sixty-second Config 101 playback with dropouts

The source data is not stored in Git. Place `config_101.h5` in
`turing_data/`, then extend its truth-labelled PDWs to a continuous,
zero-relative 60-second stress-test recording:

```bash
python3 extend_config_pdws.py turing_data/config_101.h5 \
  --duration 60 \
  --output turing_data/config_101_60s_gaps.h5
```

When no explicit schedule is supplied, four high-support truth emitters are
given staggered silence intervals. To choose the emitters and gaps yourself,
repeat `--dropout LABEL:START_S:END_S`, for example:

```bash
python3 extend_config_pdws.py turing_data/config_101.h5 \
  --duration 60 \
  --dropout 8:12:15 --dropout 22:30:34 \
  --output turing_data/config_101_60s_gaps.h5
```

Run the tracker with a live AOA-versus-time waterfall at true 1x simulated
time:

```bash
python3 run_live_blob_tracker.py turing_data/config_101_60s_gaps.h5 \
  --playback-rate 1 \
  --output results/config101_60s_live
```

Use `--playback-rate 10` for a ten-times-faster preview, `--playback-rate 0`
for the fastest possible run, or `--no-live` for unattended processing. The
waterfall draws provisional estimated tracks as solid histories and majority
ground truth as dashed histories. Truth is read only after association.

This extension repeats the measured Config 101 template while preserving its
truth IDs; it cannot reconstruct hidden simulator state that was not stored
in the HDF5 file. It is therefore suitable for tracker/dropout stress testing,
not a replacement for a future stateful scenario generator.

## Focused cross-RF train linking

To study one known development case without presenting truth to the
association logic, run the Config 101 emitter-46 candidate family in window 0:

```bash
python3 run_cross_rf_focus.py turing_data/config_101.h5 \
  --window 0 --blobs 8 9 10 11 12 13 14 15 \
  --tolerance-us 5 \
  --output results/emitter46_cross_rf_focus
```

The selected blobs define the controlled experiment. The linker uses AOA,
pulse width, activity time, PRI/harmonic compatibility and phase evidence. It
retains merged-emitter and separate-emitter hypotheses; truth is read only
after linking to score the candidate family. Multiple timing phases are
reported rather than being treated as proof of one physical emitter.

Create the focused performance plot and residual-metrics CSV:

```bash
python3 plot_cross_rf_performance.py turing_data/config_101.h5 \
  results/emitter46_cross_rf_focus/cross_rf_families.json \
  --output results/emitter46_cross_rf_focus/tracker_performance.png
```

Run the sequential association-score experiment across all Config 101
windows. The first window provides the eight measurement-only seed blobs;
truth labels are used only for the evaluation trace:

```bash
python3 run_cross_rf_confidence.py turing_data/config_101.h5 \
  --seed-window 0 --seed-blobs 8 9 10 11 12 13 14 15 \
  --tolerance-us 5 --output results/emitter46_confidence
python3 plot_association_confidence.py \
  results/emitter46_confidence/association_confidence.json
```

The persistent score combines timing (40%), RF behaviour (25%), AOA/pulse
width similarity (20%) and continuity (15%). It is deliberately reported as
an uncalibrated association score, not a probability.

The sequential experiment also has a conservative track-aided fallback. When
the group path supplies no observation, raw PDWs are compared with the active
track prediction. Five compatible pulses within 10 ms are required before the
batch is confirmed; ambiguous and rejected pulses remain available to normal
discovery. Reproduce the sparse final-window check directly with:

```bash
python3 run_sparse_track_reacquisition.py turing_data/config_101.h5 \
  --seed-window 0 --seed-blobs 8 9 10 11 12 13 14 15 \
  --target-window 84 --evaluation-truth-id 46
```

Stress ambiguity handling with a deliberately similar second emitter sharing
the same RF modes, plus midpoint pulses that must remain unassigned:

```bash
python3 run_competing_emitter_stress.py turing_data/config_101.h5 \
  --seed-window 0 --seed-blobs 8 9 10 11 12 13 14 15 \
  --target-window 84 --evaluation-truth-id 46
```

Run the harder full-recording test with a hopping competitor, two deliberate
gaps and a close AOA approach:

```bash
python3 run_long_competing_emitter_test.py turing_data/config_101.h5 \
  --seed-window 0 --seed-blobs 8 9 10 11 12 13 14 15 \
  --evaluation-truth-id 46 \
  --output results/emitter46_confidence/long_competing_emitter.json
python3 plot_long_competing_emitter_test.py \
  results/emitter46_confidence/long_competing_emitter.json
```

Run the first covariance-based stateful tracker and plot its confidence,
state covariance and prediction gates:

```bash
python3 run_stateful_tracker_monitor.py turing_data/config_101.h5 \
  --seed-blobs 8 9 10 11 12 13 14 15 \
  --evaluation-truth-id 46 \
  --output results/emitter46_confidence/stateful_tracker_monitor.json
python3 plot_stateful_tracker_monitor.py \
  results/emitter46_confidence/stateful_tracker_monitor.json
```

The tracker itself never reads truth. Truth is applied only after all PDW
decisions to measure correct, wrong, ambiguous and unclaimed associations.
The blob tracker uses Kalman state/covariance updates and joint NIS chi-square
gating for AOA, pulse width and PRI. Its displayed track confidence is a
Bernoulli existence probability updated from innovation likelihood, probability
of survival, probability of detection, RF-mode probability and expected clutter;
missed detection opportunities apply the full Bernoulli posterior update.
Association weights and existence probabilities are only as accurate as the
`Q`, `R`, survival, detection and clutter models, which still require calibration
against representative data.

## Kalman consistency monitor

Record candidate diagnostics while replaying the E46/E38 case:

```bash
python3 run_two_emitter_blob_tracker.py turing_data/config_101.h5 \
  --scenario moderate --output results/consistency/moderate.json \
  --consistency-log results/consistency/moderate.jsonl
python3 plot_kalman_consistency.py results/consistency/moderate.jsonl \
  --run-result results/consistency/moderate.json \
  --output results/consistency/moderate_consistency.png
```

The JSONL stream contains every evaluated observation/track pair in every
incoming beam hypothesis. It records innovation, diagonal predicted covariance
`P_predicted`, `R`, `S`, gain `K`, joint NIS, degrees of freedom, configured
chi-square threshold, selected PRI harmonics and before/after existence.
`accepted` means an update branch was generated, before beam pruning; it does
not mean that branch became the winning association. `rejected` means NIS
failed the gate; `not_selected` means it passed but exceeded the match-branch
limit. Missed opportunities record existence changes with null NIS and empty
measurement components, because no measurement exists. Miss timestamps mark
when the opportunity was scored. Births and track-fragment consolidation are
not observation innovation tests.

Diagnostics are opt-in through `consistency_sink`; the tracker retains no log
history. The plotting tool uses only incoming hypothesis rank zero and, with
`--run-result`, primary-track candidates belonging to that track's source
family. This family filter is the experiment's measurement-based selection,
not a ground-truth association. Cross-family rejection diagnostics remain in
the full log. The report does not pool alternative beam hypotheses.

Outputs include NIS histograms by dimension, pre-gate and accepted mean NIS,
95% and 99.73% gate fractions, standardized innovations and their means/event-lag
ACF, gains, covariance fractions and a separate native-unit `P`, `R`, `S` plot.
The JSON report includes numerical statistics and component NIS contributions.
PRI components use the selected common harmonic coordinate; multiply their
state and variance by the recorded track harmonic and its square to return to
the track's PRI coordinate.

Chi-square curves are references: upstream selection, harmonic minimization,
correlated summaries and beam selection prevent treating these records as IID
Gaussian innovations. Accepted-only statistics are also truncated by gating.
Do not tune all `Q`/`R` values from the pooled NIS mean. Inspect individual
components and rejected summaries first; detection/survival and lifecycle
settings remain separate calibration tasks.

Audit the E38 PRI candidates and run a timing-specificity control without
changing tracker parameters:

```bash
python3 diagnose_pri_consistency.py results/consistency/moderate.jsonl \
  results/consistency/moderate.json \
  --output results/consistency/e38_pri_audit
python3 run_pri_null_control.py turing_data/config_101.h5 \
  --window 10 --blob 7 --repetitions 20 \
  --output results/consistency/e38_pri_null_control.json
```

The audit exports candidate-level CSV/JSON and a plot, including reconstructed
incoming track PRI, chosen harmonics, component NIS, support and evaluation-only
truth purity. The control preserves the selected blob's pulse count, time span
and measurement marks, replaces TOAs with uniform random times, and reruns the
same fixed-PRI discovery and summary adapter. Its purpose is to measure how
readily this discovery setting reports precise periods from aperiodic timing;
it does not identify the real emitter's fundamental PRI.
