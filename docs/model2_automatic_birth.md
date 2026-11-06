# Model 2 automatic birth — controlled mixture

This stage removes the pre-created E46 and E38 HMM filters. It tests whether two
identities can be initiated from a chronological, interleaved pulse stream
without passing a truth label into the tracker.

The input remains a component-isolation fixture: the runner selects all E46 and
E38 pulses outside the tracker, preserves their original TOA order, strips the
labels, and passes only raw `PDW` objects to the birth manager. Background
emitters are absent. Truth is reattached only after every decision for mapping
automatic IDs and calculating the confusion matrix.

## Staged initiation

A one-pulse HMM knows only one RF and PW state. Immediately applying the mature
Model 2 evidence would reject the next hop and create duplicate tracks. Birth is
therefore staged:

1. An unmatched pulse creates a tentative HMM.
2. Tentative HMMs use only circular AOA continuity within 3 degrees while RF and
   PW state discovery remains open.
3. After 256 assigned pulses, the most recent 128 one-step predictive scores
   freeze a 99%-recall evidence threshold for that track.
4. The confirmed track enters the unchanged Model 2 competition.
5. A rejected pulse near a confirmed track's AOA cannot create a duplicate
   identity. It remains visibly rejected.
6. When sufficient early active gaps exist, the already-defined positive-only,
   capped timing support is fitted and enabled. A timing-off arm uses the same
   birth and evidence settings.

The 3-degree tentative and duplicate-suppression gates are explicit initiation
settings based on the 1-degree Model 2 AOA measurement scale. They imply a known
limitation: this birth stage cannot resolve two simultaneous emitters within the
same AOA gate. That must be tested separately rather than hidden by retuning.

Run and plot the controlled birth replay with:

```bash
python3 run_model2_automatic_birth.py turing_data/config_101.h5 \
  --truth-emitters 46 38 \
  --output results/sequential_pdw/hmm_model2_birth/e46_e38.json \
  --diagnostics results/sequential_pdw/hmm_model2_birth/e46_e38.jsonl

python3 plot_model2_automatic_birth.py \
  results/sequential_pdw/hmm_model2_birth/e46_e38.json \
  results/sequential_pdw/hmm_model2_birth/e46_e38.jsonl \
  --output results/sequential_pdw/hmm_model2_birth/e46_e38.png
```

## Development replay

The development replay used Config 101 at base revision
`57347ffc4b4d4bdbcbf3fc0a2dbc65faad3ff224` with the birth files in a dirty
working tree. The result contains dataset, revision, dependency and active-source
hashes.

The controlled mixture contains all 12,530 E46 and 12,211 E38 pulses. Both arms
create exactly two tracks and confirm both. Evaluation maps automatic T1 to E38
and T2 to E46. Each track has 100% selected-pulse purity and there is exactly one
identity per emitter.

With timing off, E46 retains 12,389 pulses (98.87% recall) and E38 retains
11,868 (97.19%). With timing on, E46 retains 12,409 (99.03%) and E38 retains
11,877 (97.26%). Timing rescues 29 pulses and loses none. There are no E46/E38
cross-assignments. The remaining 121 E46 and 334 E38 pulses stay rejected and
do not create fragments.

The early online timing fit is not a transmitter-PRI claim. In particular, the
E38 tentative history selects a broad 12.542 µs phase lattice, whereas the later
seeded held-out experiment selected 10.502 µs. Timing remains bounded secondary
support; this disagreement must not be hidden or interpreted as timing-state
identification.

This milestone validates initiation only for a clean two-emitter mixture whose
AOAs are separated. It does not test background rejection, full-scene track
capacity, co-azimuth emitters, track deletion or reacquisition. The next matched
experiment should add background PDWs without changing birth, HMM, threshold or
timing settings.

That matched experiment is now recorded in
[Model 2 frozen full-scene birth stress](model2_full_scene_birth.md). It holds
those settings fixed and fails because eight one-pulse births consume the entire
track capacity by input pulse 9, before any confirmation. The controlled result
above remains valid for its stated mixture but is not full-scene acceptance.
