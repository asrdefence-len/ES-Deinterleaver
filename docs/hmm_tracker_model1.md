# HMM emitter tracker - Model 1

Model 1 is the first incremental proper HMM experiment. Its input is:

**Frequency + TOA + AOA**

`PulseWidth` and `Amplitude` are excluded. An invariance test changes both by
extreme amounts and requires the active filter result to remain identical.

## Controlled boundary

The canonical Config 101 truth label selects E46 outside inference. Original
PDW IDs and TOAs are preserved, while the filter receives the same truth-free
`PDW` type used by the raw reader. Every selected pulse updates one emitter
filter. There are no track births, clutter decisions, association gates or
rejected pulses in this controlled Model 1 run.

This tests emitter-state inference. It is not a multi-emitter deinterleaving
benchmark, and 100% update coverage is true by construction.

## Active state model

### RF hidden Markov chain

RF states are discovered from observations using a broad measurement gate; the
number and centres are not read from transmitter configuration. Each pulse
uses the complete HMM forward recursion:

```text
prediction[j] = sum_i posterior_previous[i] * transition[i,j]
posterior[j] proportional to prediction[j] * RF_emission[j]
```

Transitions are updated from the expected transition distribution `xi`, not a
hard winning state. Gaussian RF emission parameters are updated with posterior
weights. Probability and log-probability vectors are both exported.

### AOA continuous Markov state

AOA is a circular random-walk state with process variance, measurement
variance, innovation likelihood and Kalman update. Angular residuals cross the
-180/180 degree boundary correctly. TOA supplies the elapsed time for this
prediction step.

### Active evidence boundary

The active per-pulse log evidence is exactly:

```text
RF predictive log evidence + AOA predictive log evidence
```

No timing statistic is included. The result records this boundary as
`active_evidence_features = [Frequency, AoA]`.

## Detected-pulse timing shadow

TOA also feeds a separate online observer of the **received pulse train**. It
maintains:

- a Bayesian log-spaced histogram of adjacent detected-pulse gaps;
- a predictive density for the next received gap; and
- a generic two-state short-gap/long-gap visibility description.

These are properties of the accepted detections, not automatically properties
of the transmitter. A received gap may span missed pulses, mixed timing modes,
receiver visibility loss or several transmitter intervals. Therefore:

- `feedback_weight` is fixed at exactly zero;
- `transmitter_pri_estimate` is always `null`; and
- shadow timing density is never added to active HMM evidence.

The shadow is the safe place to inspect timing structure before deciding
whether a calibrated timing likelihood exists. A separate analyzer plots
adjacent gaps, lag differences and chronological stability. Peaks are explicitly
labelled as possible sums, harmonics, interleaving or detection effects—not as
automatically inferred PRI states.

## Why the earlier PRI HMM was removed from active inference

The complete E46 replay at commit `3eb8383` found one timing value near 10.633
microseconds, but did not represent all timing modes. The multi-mode revision at
commit `be15d5b` retained direct-support states at 4.796, 6.986, 8.756, 11.076,
16.871 and 32.157 microseconds. The 32.157-microsecond state accumulated strong
support even though such a gap can be explained by a path through shorter
intervals.

That result showed that the model was promoting observed PDW-to-PDW gaps into
transmitter PRI states before separating missed detections and summed paths.
Adding more pruning rules would have made this unvalidated assumption harder to
see. The active timing HMM was therefore removed rather than tuned. RF and AOA
were held fixed: their first 500 per-pulse posteriors, evidence, AOA means,
covariances and innovations are bit-for-bit unchanged from `be15d5b`.

## Verification

Focused tests require:

- RF forward probabilities and transition rows normalize to one;
- soft expected-transition updates learn a known RF sequence;
- changing PW and amplitude cannot change any Model 1 result;
- circular AOA filtering remains continuous across the wrap boundary;
- received-gap predictive densities remain finite;
- short/long shadow visibility probabilities normalize;
- timing feedback is zero for every update;
- no transmitter PRI is emitted; and
- total active evidence equals RF evidence plus AOA evidence exactly.

A 500-pulse canonical E46 smoke processed every selected pulse, learned eight
RF states, and ended at AOA 32.470 degrees with 0.053-degree posterior sigma.
It recorded 499 received gaps with zero timing feedback. This is implementation
and wiring evidence only; the clean 12,530-pulse replay is the next decision
point.

## Local E46 run

```bash
python3 run_hmm_emitter_model1.py turing_data/config_101.h5 \
  --truth-emitter 46 \
  --limit 12530 \
  --output results/sequential_pdw/hmm_model1/e46_timing_shadow.json \
  --diagnostics results/sequential_pdw/hmm_model1/e46_timing_shadow.jsonl
```

Plot active uncertainty/evidence and the timing shadow:

```bash
python3 plot_hmm_model1_performance.py \
  results/sequential_pdw/hmm_model1/e46_timing_shadow.jsonl \
  --output results/sequential_pdw/hmm_model1/e46_timing_shadow.png
```

Describe the received train independently:

```bash
python3 analyze_detected_pulse_train.py \
  results/sequential_pdw/hmm_model1/e46_timing_shadow.jsonl \
  --output results/sequential_pdw/hmm_model1/e46_detected_train.json \
  --plot results/sequential_pdw/hmm_model1/e46_detected_train.png
```

The next step is to review the full detected-train structure. Timing may later
enter association only as a separately calibrated, bounded soft likelihood
ratio. Until that validation exists, its feedback remains zero. PW addition and
multi-emitter association remain later milestones.
