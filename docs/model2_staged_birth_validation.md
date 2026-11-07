# Model 2 staged birth validation

This milestone changes only tentative birth assignment and confirmation. The
mature Model 2 Frequency/AOA/PW equations, bounded timing support, confirmed-
track competition and 256-track diagnostic capacity remain unchanged. No
confirmed-track retirement, expiry or reacquisition is added here.

The baseline comparison is the recorded 256-track full-scene diagnostic, which
assigned 98.84% of PDWs but achieved only 62.98% evaluation-weighted identity
purity. Its T13 absorbed 528,221 PDWs from 25 truth emitters at 47.03% purity.

## Staged design

Each truth-blind tentative candidate now progresses through these stages:

1. The first eight assigned pulses use the existing 3-degree circular AOA gate
   as a small state-discovery seed.
2. Subsequent tentative assignment requires AOA continuity and at least one
   known RF or PW component. A pulse with both RF and PW novel may create a
   separate candidate rather than being forced into the nearest-AOA candidate.
3. At 128 pulses the candidate model freezes. The last 64 adaptive scores form
   separate robust RF, AOA and PW reference floors.
4. The next 128 compatible pulses are scored without adapting the frozen model.
   Each component must independently meet its reference on at least 90% of the
   rolling held-out block. Requiring all three simultaneously on every pulse was
   tested and removed because its expected clean pass rate is approximately
   `0.95^3`, which fragmented E46.
5. An inconclusive block leaves the same candidate frozen and tests another
   rolling block. It does not create a new identity merely because one block is
   marginal.
6. Pathological training component floors reject the candidate. A pathological
   held-out one-percent threshold cannot promote it: the candidate remains
   frozen and tentative while the rolling held-out block advances. A later
   coherent, non-pathological block may confirm it. This prevents an extreme
   outlier from creating a permissive catch-all threshold without destroying a
   candidate that can subsequently demonstrate stable evidence.

Failed candidates remain in the result for audit but release their active birth
slot. Truth is never used by these decisions.

## Exact run

```bash
python3 run_model2_full_scene_birth.py turing_data/config_101.h5 \
  --max-tracks 256 \
  --staged-birth-validation \
  --output results/sequential_pdw/hmm_model2_birth/full_scene_staged.json \
  --diagnostics results/sequential_pdw/hmm_model2_birth/full_scene_staged.jsonl

python3 plot_model2_full_scene_birth.py \
  results/sequential_pdw/hmm_model2_birth/full_scene_staged.json \
  --output results/sequential_pdw/hmm_model2_birth/full_scene_staged.png
```

## Development replay

The development replay used base revision
`60a8dc31b45873fa7203328996d9162959a09887` with the staged-validation files in
a dirty working tree. The JSON result records the dataset, revision/dirty state,
dependencies and active-source hashes.

On all 844,999 Config 101 PDWs it creates 44 historical candidates: 32 confirm,
11 remain tentative and one pathological candidate is rejected. It assigns
819,410 PDWs (96.97%). Evaluation-weighted identity purity rises from 62.98% to
95.03%; distinct dominant emitters rise from 35 to 42; and the maximum number
of truth emitters in one automatic track falls from 25 to four. The giant T13
failure is eliminated. Seven tracks remain below 90% purity, so this is a large
improvement rather than full-scene acceptance.

E46 remains one identity: T19 receives 12,481/12,530 E46 pulses (99.61%
retention) and has 99.94% purity. E38 receives 12,142/12,211 pulses (99.43%
total coverage); primary T16 retains 12,056 (98.73%) at 99.78% purity, while 86
E38 pulses enter a secondary candidate.

All eleven previously identified strong emitters retain a high-purity primary
identity. Most primary retentions remain near their high-capacity reference.
The largest explicit regression is E33, from 97.01% to 94.23% primary retention;
E38 primary retention falls from 99.87% to 98.73%. These losses must remain
visible rather than being hidden by total assignment coverage.

An AOA-gate-only held-out variant was also replayed. It confirmed five more
candidates but did not resolve the large pure E37 tentative track and reduced
weighted purity from 95.03% to 94.69%. That regression was removed; the recorded
version retains frozen RF, AOA and PW held-out evidence.

## Boundary and next stage

Birth validation is improved but not perfect. Several same-feature emitters
still merge, and some pure, strongly supported candidates remain tentative
because a frozen early model does not represent their later agility. Those
limitations are not repaired here by truth-derived thresholds.

The next isolated component is lifecycle: tentative ageing/expiry, confirmed
coasting/retirement, released capacity and explicit reacquisition. Lifecycle
must preserve the staged birth result and report missing tracks as missing. It
must not retune the HMM, timing or validation thresholds at the same time.

The complete repository regression at this milestone is **289 passed**. This
verifies the staged state transitions, same-AOA RF/PW separation, rolling
held-out behavior, pathological-threshold rejection, JSON provenance boundary
and unchanged legacy/default behavior. It does not imply full-scene performance
acceptance.

