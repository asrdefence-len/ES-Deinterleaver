# HMM emitter tracker - Model 1

Model 1 is the first incremental implementation of the requested proper HMM
tracker. It uses exactly three incoming measurements:

**Frequency + TOA + AOA**

`PulseWidth` and `Amplitude` are excluded from the implementation and an
invariance test changes both by extreme amounts while requiring every HMM
update and final state to remain identical.

## Controlled boundary

The canonical Config 101 truth label selects E46 outside the filter. Original
PDW IDs and TOAs are preserved, while the filter receives the same truth-free
`PDW` type used by the raw reader. Every selected pulse updates exactly one
emitter filter. There are no track births, clutter decisions, association gates
or rejected pulses in Model 1.

This establishes emitter-state inference. It is not yet a multi-emitter
deinterleaving benchmark and 100% update coverage is true by construction.

## State model

### RF hidden Markov chain

RF states are discovered from observations using a broad measurement gate; the
number and centres are not read from transmitter configuration. For each pulse,
the filter calculates the complete HMM forward distribution:

```text
prediction[j] = sum_i posterior_previous[i] * transition[i,j]
posterior[j] proportional to prediction[j] * RF_emission[j]
```

Transitions are updated from the expected transition distribution `xi`, not a
hard winning state. Gaussian RF emission parameters are updated with posterior
weights. Both probability and log-probability vectors are exported so very
unlikely states remain inspectable without numerical underflow.

### AOA continuous Markov state

AOA is a circular random-walk state with process variance, measurement variance,
innovation likelihood and Kalman update. Angular residuals cross the -180/180
degree boundary correctly.

### TOA visibility-aware hidden semi-Markov model

The timing filter retains a normalized joint posterior over adaptive PRI modes
and two hidden visibility states:

- **Active:** the gap spans one or more transmitter intervals. The filter
  marginalizes possible elapsed interval counts using a geometric
  missed-detection prior. Each interval can occupy a different PRI mode.
- **Coast:** the receiver was not observing an ordinary pulse interval. A broad
  gap density explains scan or visibility gaps and the PRI state is preserved.

Active/coast probabilities pass through a two-state Markov transition matrix
before each gap likelihood. The pulse-detection probability has a Beta
posterior and is learned from the soft active-interval counts. PRI modes have a
separate learned transition matrix, updated from soft expected transitions in
the same manner as the RF HMM.

A multi-interval gap is evaluated as a Markov-additive path: the predicted gap
is the sum of the possibly different PRI modes traversed by the hidden pulses.
It is never approximated as `n * one PRI`. Only direct one-interval evidence
moves a PRI-mode centre, preventing long mixed-mode gaps from dragging states
toward subharmonics.

This structure gives long gaps a proper hidden-state explanation instead of
requiring thousands of ordinary missed pulses. It also makes a fundamental PRI
preferable to a submultiple: explaining each observed short gap with one
interval has higher prior probability than repeatedly requiring two, three or
more intervals.

The complete Model 1 state is therefore a Rao-Blackwellised/factorial state:
a discrete RF HMM posterior, a continuous circular AOA posterior and a
visibility-aware discrete PRI-mode HMM.

## Verification

Focused tests require:

- RF forward probabilities and every transition row normalize to one.
- Soft expected transition updates learn a known three-state sequence.
- Changing PW and amplitude cannot change any Model 1 result.
- Repeated short gaps select the fundamental PRI over its subharmonics.
- Long gaps switch to coast, preserve the PRI posterior and return to active.
- Visibility probabilities normalize and the detection probability is learned.
- A controlled 7 → 9 → 11 microsecond sequence learns three distinct PRI modes
  and the corresponding cyclic transition topology.
- Multi-interval likelihoods sum paths through possibly different PRI modes.
- Circular AOA filtering remains continuous across the wrap boundary.

The preceding complete E46 replay at commit `3eb8383` correctly found one
10.633-microsecond timing value but assigned it effectively all timing mass.
Evaluation-only metadata inspected afterward lists three E46 jittered timing
values: 6.697, 8.572 and 10.637 microseconds. The new multi-mode implementation
therefore remains a timing-only correction; those metadata values are not read
by inference.

On a 500-pulse dirty-tree smoke, the new HMM formed directly supported states
near 6.711, 8.658 and 10.966 microseconds. It also retained a supported
32.249-microsecond ambiguity, so this smoke is wiring evidence rather than
performance acceptance. The clean complete replay remains the decision point.

## Local E46 run

```bash
python3 run_hmm_emitter_model1.py turing_data/config_101.h5 \
  --truth-emitter 46 \
  --limit 12530 \
  --output results/sequential_pdw/hmm_model1/e46.json \
  --diagnostics results/sequential_pdw/hmm_model1/e46.jsonl
```

The result records enabled and excluded features, dataset/code provenance,
posterior-normalization checks, predictive log evidence, RF state emissions and
transition matrix, AOA estimate/uncertainty and the complete timing posterior.

The per-pulse diagnostic can be plotted without rerunning the filter:

```bash
python3 plot_hmm_model1_performance.py \
  results/sequential_pdw/hmm_model1/e46.jsonl \
  --output results/sequential_pdw/hmm_model1/e46_performance.png
```

The four panels deliberately use the uncertainty measure appropriate to each
state: posterior covariance for continuous AOA, Shannon entropy for the
discrete RF state, timing posterior mean/covariance with coast probability, and
rolling predictive log evidence. A small covariance is not by itself proof of
accuracy. In particular, the timing panel remains visible because the first
complete E46 run converged confidently to a harmonic/submultiple rather than
establishing the transmitter's fundamental PRI.

Model 2 will use this identical input and filter, changing only the addition of
PW observations. Multi-emitter association remains a later layer.
