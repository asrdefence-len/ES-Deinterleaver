# Model 2 raw-PDW track overlay audit

This is an evaluation-only view of the staged Model 2 full-scene replay. It
does not change association, evidence, timing, birth validation or lifecycle.
The runner optionally records each raw PDW beside the decision that the tracker
has already made. Truth labels are appended after that decision and are used
only to colour the plots.

## Exact run

```bash
python3 run_model2_full_scene_birth.py turing_data/config_101.h5 \
  --max-tracks 256 \
  --staged-birth-validation \
  --output results/sequential_pdw/hmm_model2_birth/full_scene_staged.json \
  --diagnostics results/sequential_pdw/hmm_model2_birth/full_scene_staged.jsonl \
  --audit-trace results/sequential_pdw/hmm_model2_birth/full_scene_staged.npz

python3 plot_model2_track_audit.py \
  results/sequential_pdw/hmm_model2_birth/full_scene_staged.json \
  results/sequential_pdw/hmm_model2_birth/full_scene_staged.npz \
  --tracks-per-page 5 \
  --output-dir results/sequential_pdw/hmm_model2_birth/full_scene_track_audit
```

The current 44-track replay produces nine PNG pages. Each track is overlaid on
the same sampled raw Frequency, AOA and pulse-width scene. Blue points are
assigned pulses from the track's dominant evaluation emitter, orange crosses
are pulses from that emitter missed by the track, and red crosses are other
emitters assigned to the track. Birth and confirmation times are marked. Each
row also reports the evaluation composition of the first eight AOA-only seed
pulses.

## What the overlay reveals

The visual audit supports the clean-target statistics. T19 follows E46's eight
RF levels and three PW modes across the recording with 99.94% evaluation
purity. T16 follows E38's broad RF behaviour with 99.78% purity. Their few red
marks are visible rather than hidden by rounded percentages.

It also exposes a specific remaining birth failure. T8's first eight assigned
pulses already contain E14, E26 and E35. These populations are visibly distinct
in RF/PW but lie within the small AOA neighbourhood. Once all three enter the
AOA-only seed, their RF/PW states are learned as if they were one agile emitter;
the later held-out block can validate that persistent mixture. T12 is likewise
seeded by E22 and E8, and T18 by E6, E10 and E28. This is direct replay evidence,
not an inference from the final purity number alone.

The orange-only rows distinguish a different limitation. T13 is a 100%-pure E40
candidate, but remains tentative because its later agility is not represented
well enough by the frozen early candidate. A pure tentative track and a mixed
confirmed track therefore need different remedies.

## Consequence

Lifecycle work should not be expected to repair T8/T12/T18: they are already
confirmed mixed identities. The next isolated birth investigation is the
composition of the first eight AOA-only pulses and whether incompatible RF/PW
subsets must branch before they become learned states of one candidate. HMM,
timing, held-out thresholds and lifecycle remain frozen during that test.

The complete repository regression after adding the trace and paginated plot is
**292 passed**. The audit output is diagnostic evidence, not a new performance
benchmark or a tracker improvement.
