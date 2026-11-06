# Model 2 automatic birth — frozen full-scene stress replay

This replay makes one controlled change to the preceding automatic-birth
experiment: every chronological Config 101 PDW is passed to the tracker. The
Model 2 evidence, timing support, 3-degree tentative AOA gate, 256-pulse
confirmation rule, threshold calibration, duplicate suppression and eight-track
limit remain frozen.

No truth label is supplied to the birth manager or to either HMM. Truth is read
only after each decision to report automatic-track composition and E46/E38
coverage. This is therefore a full-scene stress test of the current initiation
and capacity policy, not a truth-selected tracking run.

## Exact run

```bash
python3 run_model2_full_scene_birth.py turing_data/config_101.h5 \
  --output results/sequential_pdw/hmm_model2_birth/full_scene.json \
  --diagnostics results/sequential_pdw/hmm_model2_birth/full_scene.jsonl

python3 plot_model2_full_scene_birth.py \
  results/sequential_pdw/hmm_model2_birth/full_scene.json \
  --output results/sequential_pdw/hmm_model2_birth/full_scene.png
```

The runner enables a mathematically safe score-pruning optimization for this
large replay. For a confirmed track it first calculates the exact AOA and timing
terms and an upper bound on the RF+PW predictive density. A full RF/PW score is
skipped only when even that upper bound cannot reach the frozen threshold. Tests
require pruning on and off to produce identical decisions and final models; it
does not change the tracker policy.

## Development replay result

The development replay used base revision
`3cfd8f91739d4d4e76c79cd9bf2575333e6a0ed3` with the full-scene files in a
dirty working tree. Its JSON output records that state together with dataset,
active-source and dependency hashes. The replay processed all 844,999 Config
101 PDWs. It created the maximum eight tentative tracks from the first nine
input pulses. The capacity was therefore full before any track had accumulated
the 256 updates required for confirmation. The first confirmations occurred
hundreds to thousands of pulses later, but no slot is ever released by this
implementation.

All eight tracks eventually confirmed and 507,338 PDWs were assigned, giving
60.04% overall assignment coverage. That aggregate is not a target-retention
success:

- E46: 0 of 12,530 pulses assigned;
- E38: 0 of 12,211 pulses assigned;
- 323,399 PDWs rejected because the eight-track capacity was already full;
- 14,262 additional rejected PDWs suppressed as near an existing confirmed
  track; and
- two confirmed tracks were materially mixed: T3 was 80.7% pure and T8 was
  71.1% pure by evaluation truth.

The controlled two-emitter result therefore does not survive the frozen
full-scene replay. The immediate cause is structural: a single unmatched pulse
consumes a permanent track slot, yet confirmation requires 256 assigned pulses
and there is no tentative-hypothesis eviction, confirmed-track deletion or
capacity recycling. E46 and E38 arrive after capacity is already exhausted, so
they cannot be born or reacquired.

The complete repository regression at this milestone is **283 passed**. This
verifies implementation consistency, including exact decision equivalence with
the safe score pruning enabled and disabled. It does not turn the failed replay
into performance acceptance.

## Next single investigation

Keep the Model 2 observation equations, timing policy and confirmed-track
competition frozen. Change only tentative birth admission and capacity
management: tentative pulse hypotheses must compete for a bounded pool and earn
a permanent confirmed slot from repeated support. Failed or stale tentative
hypotheses must release their capacity. This should be evaluated first on the
same full-scene input before adding confirmed-track deletion, changing the
eight-track operating limit or tuning evidence thresholds.

