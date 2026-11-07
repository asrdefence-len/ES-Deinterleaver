# Bayesian emitter tracker - Model 3 association and birth foundation

Model 3 preserves the Model 2 RF, circular-AOA and log-PW HMMs. It replaces
Model 2's thresholded multi-track manager with a marked-Poisson global
multi-hypothesis posterior.

For each chronological PDW, every retained global hypothesis branches into:

- one explanation for each existing emitter;
- a new-emitter explanation; and
- a clutter explanation.

Existing-emitter mark density is the Model 2 RF/AOA/PW predictive density.
Explicit new-RF-state and new-PW-state mixture mass allows waveform agility
without forcing an incompatible observation to become a new emitter. Birth and
clutter have declared rates and uniform mark densities over declared sensor
support. Each emitter's PDW rate has a Gamma-Poisson posterior.

The branches are normalized into posterior probabilities. There is no
confirmation threshold, AOA birth gate, nearest-track assignment, or
truth/configuration input. The MAP action is reported only as a diagnostic; it
does not control the state update. Track existence is the posterior mass of
global hypotheses containing that track.

The current computational approximation retains the highest-probability 128
global hypotheses by default and reports the probability mass discarded at
each update. This is explicit hypothesis pruning, not a hidden association
rule.

Global hypotheses are stored as a shared-state DAG. Identical conditional
emitter states share one Model 2 instance; predictive evidence is evaluated
once per unique state and only the emitter receiving a PDW is copied and
updated. Pulse-rate exposure is derived exactly from birth and current scene
time. This is a representation optimization: it adds no blob proposal, hard
gate or association threshold.

## Current boundary

This milestone implements probabilistic association, clutter and birth. Track
death, survival branching, coasting and reacquisition are not yet implemented;
existence therefore does not yet constitute lifecycle acceptance. The first
test is a small controlled block, not a full Config 101 replay.

The per-emitter filter is still the preserved Model 2 implementation. Its RF
and PW latent-state creation uses fixed distance gates. Consequently Model 3
has Bayesian emitter association and birth, but is not yet a completely
Bayesian waveform-state learner. Replacing those gates requires posterior mass
over occupied and unoccupied RF/PW states, not another calibrated threshold.

## Controlled run

```bash
python3 run_bayesian_model3.py turing_data/config_101.h5 \
  --truth-emitters 46 38 \
  --limit-selected 250 \
  --max-global-hypotheses 128 \
  --output results/sequential_pdw/hmm_model3/e46_e38_250.json \
  --diagnostics results/sequential_pdw/hmm_model3/e46_e38_250.jsonl
```

Truth selects the controlled input outside inference and is added only after
each posterior update for fractional evaluation. The output reports posterior
association mass, marginal existence, fractional track composition and the
retained global-hypothesis mass. It deliberately does not convert these values
to hard precision/recall decisions.

The first 250-selected-PDW replay retained 0.999999984 of the represented
posterior mass after the 128-hypothesis bound. Its leading E46- and
E38-dominant identities had marginal existence 0.999999981 and 0.999615041,
with fractional identity purity 1.0 and 0.999999454 respectively. This block
contained approximately 234 E46 associations but only 12 E38 associations, so
it demonstrates posterior feasibility rather than balanced two-emitter
performance. A longer controlled chronological block and Bayesian lifecycle
model are required before a full-scene claim.

The 2,000-selected-PDW balanced replay contains 1,067 E46 and 933 E38 pulses.
The pre-optimization branch-and-clone implementation did not complete within
150 seconds. The shared-state representation completes in 60.03 seconds,
retains 0.999999953 represented posterior mass and ends with two unit-existence
identities. Fractional evaluation purity is 0.999999993 for E38-dominant B39
and 1.0 for E46-dominant B58. The saved 250-PDW association posterior changes
by at most 2.3e-16; its existence probabilities and retained mass are exactly
unchanged.
