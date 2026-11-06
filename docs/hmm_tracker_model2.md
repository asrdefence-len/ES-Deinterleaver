# HMM emitter tracker - Model 2

Model 2 makes one incremental change to the controlled E46 Model 1 experiment:

**Pulse Width is added as a discrete hidden Markov chain.**

The input is therefore Frequency + TOA + AOA + Pulse Width. Amplitude remains
excluded. TOA still supplies elapsed time to the AOA prediction and feeds the
zero-feedback timing shadow; timing is not active association evidence.

## Frozen Model 1 boundary

`HMMEmitterModel2` subclasses the Model 1 implementation. Every pulse first
passes through the unchanged Model 1 `process` method, then updates the new PW
chain. This prevents an accidental rewrite of RF, AOA or timing while adding PW.

A matched 500-pulse canonical E46 smoke requires every Model 1 field to equal
the saved Model 1 diagnostic exactly. The comparison covers RF posterior and
log posterior, RF evidence, AOA mean/covariance/innovation/evidence, timing gap,
timing density, visibility posterior and timing feedback weight. The current
smoke has zero difference in every field.

## Pulse-width hidden Markov chain

PW is positive and its observed populations have multiplicative spreads, so
emissions are Gaussian in natural-log seconds. State count and centres are not
read from source configuration. A new state is created only when an observation
is outside the log-distance gate of every learned state.

For every pulse the complete forward distribution is retained:

```text
PW prediction[j] = sum_i PW posterior_previous[i] * PW transition[i,j]
PW posterior[j] proportional to prediction[j] * log-PW emission[j]
```

PW transitions are learned from the soft expected transition distribution,
not from a hard winning-state sequence. Mode centres and spreads are updated
with posterior responsibilities. Model 2 active component evidence is:

```text
RF predictive evidence + AOA predictive evidence + log-PW predictive evidence
```

This is a factorial component model, not yet a calibrated multi-emitter
association score. The numerical log-density contribution depends on its stated
log-width measurement space and must not be read as an accuracy improvement by
itself. A pulse that creates a new PW state is explicitly marked in diagnostics:
that same-pulse value is state-discovery evidence, not a calibrated predictive
association likelihood.

## Initial verification

The 500-pulse canonical E46 wiring smoke discovers three PW states without
truth/configuration input:

- 0.101 µs;
- 0.271 µs; and
- 0.427 µs.

It also retains all eight RF states and exactly reproduces the Model 1 AOA result
of 32.470 degrees with 0.053-degree posterior sigma at pulse 500. Timing feedback
remains zero. This is implementation evidence only; the clean complete
12,530-pulse user replay remains the performance-review point.

Focused tests require:

- every Model 1 state/update field remains unchanged;
- PW posterior probabilities and transition rows normalize;
- a controlled three-mode PW sequence learns its centres and transition cycle;
- total Model 2 evidence equals frozen Model 1 evidence plus PW evidence;
- timing feedback remains zero; and
- changing amplitude cannot change any Model 2 result.

## Local E46 run

```bash
python3 run_hmm_emitter_model2.py turing_data/config_101.h5 \
  --truth-emitter 46 \
  --limit 12530 \
  --output results/sequential_pdw/hmm_model2/e46.json \
  --diagnostics results/sequential_pdw/hmm_model2/e46.jsonl
```

Plot the matched component behaviour:

```bash
python3 plot_hmm_model2_performance.py \
  results/sequential_pdw/hmm_model2/e46.jsonl \
  --output results/sequential_pdw/hmm_model2/e46_performance.png
```

The plot keeps AOA covariance, RF uncertainty, PW observations/states, the
zero-feedback timing shadow and incremental PW predictive evidence visible.
Small covariance or state entropy does not by itself establish accuracy or
association performance.

## Association milestones

The complete 12,530-pulse E46 replay learns eight RF states, three PW states at
approximately 0.100, 0.271 and 0.428 µs, and an AOA estimate of 32.238 degrees
with 0.149-degree posterior sigma. That component result is followed by two
held-out association experiments without changing the Model 2 state equations:

- [E46 timing association ablation](model2_timing_association.md) compares
  Frequency + AOA + PW against the same evidence plus bounded timing support.
- [E46/E38 two-emitter competition](model2_two_emitter_competition.md) allows
  two seeded filters and neither to compete for one update per pulse.

These are controlled seeded experiments. They do not yet supply automatic
track birth, unknown-emitter discovery or lifecycle management.

The next controlled milestone, [Model 2 automatic birth](model2_automatic_birth.md),
removes the pre-created HMMs. Two tentative filters are initiated and confirmed
from an interleaved, tracker-unlabelled E46/E38 stream. Background emitters are
still excluded, so this validates the initiation mechanism rather than full-
scene discovery or lifecycle management.

The following [frozen full-scene birth stress](model2_full_scene_birth.md)
passes all 844,999 Config 101 PDWs to the same manager. It fails: the eight-track
capacity is consumed by pulse 9, and neither E46 nor E38 is ever assigned. This
locates the next problem in tentative-birth capacity management, not in another
incremental change to the mature Model 2 observation equations.
