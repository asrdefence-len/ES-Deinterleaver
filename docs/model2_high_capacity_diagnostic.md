# Model 2 high-capacity birth diagnostic

This diagnostic follows the failed eight-track full-scene replay and changes
only one setting: `max_tracks` increases from 8 to 256. The 3-degree tentative
AOA gate, 256-pulse confirmation rule, threshold calibration, Model 2 evidence,
timing support and duplicate suppression remain unchanged. There is still no
track retirement or reacquisition.

The purpose is not to propose 256 as an operational limit. It asks whether the
existing birth/confirmation rule naturally separates durable emitter tracks
from weak one-pulse hypotheses when immediate capacity exhaustion is removed.

## Exact run

```bash
python3 run_model2_full_scene_birth.py turing_data/config_101.h5 \
  --max-tracks 256 \
  --output results/sequential_pdw/hmm_model2_birth/full_scene_256.json \
  --diagnostics results/sequential_pdw/hmm_model2_birth/full_scene_256.jsonl

python3 plot_model2_full_scene_birth.py \
  results/sequential_pdw/hmm_model2_birth/full_scene_256.json \
  --output results/sequential_pdw/hmm_model2_birth/full_scene_256.png
```

## Development result

The development replay used base revision
`9957f9b4c6f8635798c14ca97d67842c4a45d5de` with this diagnostic change in a
dirty working tree. Dataset and active-source hashes remain recorded in the
JSON result.

The tracker creates only 36 tracks, so the 256-track capacity is never reached.
Twenty-four tracks accumulate the existing 256-pulse support requirement and
confirm; twelve remain tentative. It assigns 835,179/844,999 PDWs (98.84%). E46
is assigned 12,480/12,530 pulses (99.60%) to one track. E38 is assigned
12,207/12,211 pulses (99.97%); 12 of those pulses enter a secondary identity.

Those target results do not establish full-scene identity accuracy. Across all
tracks, evaluation-weighted track purity is only 62.98%. Eighteen of 36 tracks
contain more than one truth emitter, eight are below 90% purity, and only 35 of
the 50 truth emitters are the dominant identity of any automatic track. The
largest failure is T13: it accumulates 528,221 PDWs from 25 truth emitters and
is only 47.03% pure.

T13 also demonstrates why pulse count is not sufficient birth validation. Its
tentative AOA-only history produces a frozen predictive threshold near
`-3477.4`, so a badly mixed candidate becomes an extremely permissive confirmed
track and continues to absorb pulses. High assignment coverage is therefore
not deinterleaving performance.

## Next single investigation

Keep capacity at 256 for diagnosis and keep mature Model 2 state equations and
timing frozen. Change only confirmation validation: a tentative candidate must
demonstrate coherent held-out Frequency + AOA + PW predictive evidence before
it becomes confirmed, and implausibly low calibrated thresholds must fail
confirmation rather than create a catch-all track. Compare the same full-scene
input and retain every mixed-track metric above.

After confirmation is trustworthy, add lifecycle as a separate change:
tentative expiry, confirmed coasting/retirement, released capacity and explicit
reacquisition. Confirmation and lifecycle must not be tuned together against
the same target scores.

That next component is now recorded in
[Model 2 staged birth validation](model2_staged_birth_validation.md). It removes
the giant catch-all and raises weighted identity purity to 95.03% while keeping
E46 on one identity. It does not yet add lifecycle.

The repository regression at this diagnostic milestone is **284 passed**.
Passing tests verifies implementation consistency; it does not override the
failed full-scene identity-purity result.

