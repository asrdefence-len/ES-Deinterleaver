# Bayesian emitter tracker - Model 3 association and birth foundation

Model 3 preserves the Model 2 RF, circular-AOA and log-PW HMMs. It replaces
Model 2's thresholded multi-track manager with a marked-Poisson global
multi-hypothesis posterior.

For each chronological PDW, every retained global hypothesis branches into:

- one explanation for each existing emitter;
- a new-emitter explanation; and
- a clutter explanation.

Existing-emitter mark density competes four explicit waveform explanations:

- known RF and known PW;
- new RF and known PW;
- known RF and new PW; and
- new RF and new PW.

Their weights combine Model 2 predictive evidence with declared new-state
priors and uniform base densities over sensor support. The new-state indicators
remain present through pruning, and a retained branch performs the matching
conditional HMM update. In a new-state branch, the current hidden RF or PW
state is the new state with probability one; uncertainty over its predecessor
updates the transition matrix fractionally. The standalone Model 2 new-state
prior is not applied again. Model 2's RF/PW distance gates are therefore
inactive in Model 3. Birth and clutter have declared rates and uniform mark
densities. Each emitter's PDW rate has a Gamma-Poisson posterior.

The branches are normalized into posterior probabilities. There is no
confirmation threshold, AOA birth gate, nearest-track assignment, or
truth/configuration input. The MAP action is reported only as a diagnostic; it
does not control the state update. Track existence is the posterior mass of
global hypotheses containing that track.

The current computational approximation retains the highest-probability 16
global hypotheses by default and reports the probability mass discarded at
each update. `--max-global-hypotheses 128` remains available for wider
diagnostics. This is explicit hypothesis pruning, not a hidden association
rule.

Global hypotheses are stored as a shared-state DAG. Identical conditional
emitter states share one Model 2 instance; predictive evidence is evaluated
once per unique state and only the emitter receiving a PDW is copied and
updated. Pulse-rate exposure is derived exactly from birth and current scene
time. This is a representation optimization: it adds no blob proposal, hard
gate or association threshold.

## Current boundary

This milestone implements probabilistic association, clutter, emitter birth
and waveform-state birth. Track death, survival branching, coasting and
reacquisition are not yet implemented; existence therefore does not yet
constitute lifecycle acceptance. The controlled E46/E38 selection is still
performed outside inference and is not a full-scene benchmark.

Model 3 uses a declared 64-state RF truncation because the preserved 16-state
Model 2 capacity fragments continuously agile E38. Pulse-width capacity is
eight states. These are finite representation limits rather than measurement
distance gates. Posterior uncertainty over state birth currently consumes the
same top-K global budget as emitter association; local Rao-Blackwellization of
that uncertainty is the next representation investigation.

## Controlled run

```bash
python3 run_bayesian_model3.py turing_data/config_101.h5 \
  --truth-emitters 46 38 \
  --limit-selected 24741 \
  --max-global-hypotheses 16 \
  --max-rf-states 64 \
  --output results/sequential_pdw/hmm_model3/e46_e38_full_bayesian_states.json \
  --diagnostics results/sequential_pdw/hmm_model3/e46_e38_full_bayesian_states.jsonl
```

Truth selects the controlled input outside inference and is added only after
each posterior update for fractional evaluation. The output reports posterior
association mass, marginal existence, fractional track composition and the
retained global-hypothesis mass. It deliberately does not convert these values
to hard precision/recall decisions.

The complete controlled replay contains 12,530 E46 and 12,211 E38 PDWs. It
finishes in 29.91 seconds on the recorded development machine and ends with
exactly two unit-existence identities: B58 for E46 and B39 for E38. Fractional
evaluation purity is 1.0 and 0.9999999995, and combined fractional coverage is
99.98323%.

This preserves the earlier aggregate-mixture result while removing RF/PW
distance gates from Model 3. It is not exact full posterior inference: the
top-16 bound retains 99.7770% per-event mass on average but has a 79.8482%
minimum. A top-64 2,000-PDW diagnostic improved the earlier pre-conditioning
minimum to only 85.58% and cost four times the wall time. Controlled identity
performance is strong, but the next step is to represent waveform-state
uncertainty locally rather than
spending the global association hypothesis budget. Lifecycle and full-scene
tests remain pending.
