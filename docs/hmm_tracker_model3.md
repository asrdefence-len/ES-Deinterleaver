# Bayesian emitter tracker - Model 3 association and birth foundation

Model 3 preserves the Model 2 RF, circular-AOA and log-PW HMMs. It replaces
Model 2's thresholded multi-track manager with a marked-Poisson global
multi-hypothesis posterior.

For each chronological PDW, every retained global hypothesis branches into:

- one explanation for each existing emitter;
- a new-emitter explanation; and
- a clutter explanation.

Within each existing-emitter track, the mark density competes four explicit
waveform explanations:

- known RF and known PW;
- new RF and known PW;
- known RF and new PW; and
- new RF and new PW.

Their weights combine Model 2 predictive evidence with declared new-state
priors and uniform base densities over sensor support. They are summed before
global association, so one existing-emitter branch carries the complete local
RF/PW-state probability. Conditional on that association, the local waveform
posterior is updated and independently truncated. In a new-state branch, the
current hidden RF or PW state is the new state with probability one;
uncertainty over its predecessor updates the transition matrix fractionally.
The standalone Model 2 new-state prior is not applied again. Model 2's RF/PW
distance gates are therefore inactive in Model 3. Birth and clutter have
declared rates and uniform mark densities. Each emitter's PDW rate has a
Gamma-Poisson posterior.

The branches are normalized into posterior probabilities. There is no
confirmation threshold, AOA birth gate, nearest-track assignment, or
truth/configuration input. The MAP action is reported only as a diagnostic; it
does not control the state update. Track existence is the posterior mass of
global hypotheses containing that track.

The computational approximation has two independent declared budgets. It
retains the highest-probability 16 global association hypotheses and, by
default, up to 16 local waveform-state histories per emitter. Both discarded
masses are reported. The full controlled development replay below uses a
four-history local profile to measure the speed/retained-mass tradeoff. This is
explicit posterior truncation, not a hidden association rule.

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
distance gates. RF/PW state-birth uncertainty now has its own local posterior
budget and no longer consumes the global emitter-association budget.

## Controlled run

```bash
python3 run_bayesian_model3.py turing_data/config_101.h5 \
  --truth-emitters 46 38 \
  --limit-selected 24741 \
  --max-global-hypotheses 16 \
  --max-local-waveform-hypotheses 4 \
  --max-rf-states 64 \
  --output results/sequential_pdw/hmm_model3/e46_e38_full_local_rb4.json \
  --diagnostics results/sequential_pdw/hmm_model3/e46_e38_full_local_rb4.jsonl
```

Truth selects the controlled input outside inference and is added only after
each posterior update for fractional evaluation. The output reports posterior
association mass, marginal existence, fractional track composition and the
retained global-hypothesis mass. It deliberately does not convert these values
to hard precision/recall decisions.

The complete controlled replay contains 12,530 E46 and 12,211 E38 PDWs. It
ends with exactly two unit-existence identities: B58 for E46 and B39 for E38.
Fractional evaluation purity is 1.0 and 0.9999999995, and combined fractional
coverage is 99.98280%, essentially unchanged from the preceding 99.98323%.

Separating the budgets resolves the observed global posterior starvation. The
top-16 global retained mass improves from a 99.7770% mean and 79.8482% minimum
to a 99.99993% mean and 99.4620% minimum. The four-history local waveform
posterior retains 99.7431% on average but has a 69.1935% minimum, so this is
still bounded approximate inference rather than an exact posterior.

The cost is material: wall time increases from 29.78 to 138.39 seconds on the
recorded machine because the local mixtures require 846,400 conditional HMM
materializations. A 2,000-PDW cap comparison measured 44.61 seconds at local
cap 16, 11.13 seconds at cap 4 and 5.47 seconds at cap 2; their mean minimum-
local retained masses were 99.4111%, 99.2573% and 99.1825%, respectively.
Identity purity was unchanged. Local Rao-Blackwellization is therefore a
posterior-representation improvement, not yet a scalability win. The next
single investigation is execution/representation optimization of this local
posterior without changing the event model, lifecycle or controlled input.
Lifecycle and full-scene tests remain pending.
