# Model 2 timing association ablation

This is a controlled one-track experiment on raw PDWs. It is separate from the
supported AOA-density + HDBSCAN/blob pipeline and is not a multi-emitter tracker
acceptance result.

The experiment first trains Frequency, AOA and pulse-width Model 2 on the first
40% of E46 pulses. The next 20% calibrates one frozen association threshold for
99% target recall. The final 40% is held out. During that held-out interval,
every pulse within the learned E46 RF discovery envelope is passed to two
parallel association arms:

- timing off: Frequency + AOA + PW evidence only;
- timing on: the identical evidence plus non-negative, capped phase support.

The arms receive the same pulses in the same order. Truth labels are not passed
to either arm. They are used only to construct the controlled E46 seed and to
evaluate decisions after both arms have decided. Candidate selection uses RF
only; it does not use AOA, PW, timing or truth.

Run the comparison and its plot with:

```bash
python3 run_model2_timing_association.py turing_data/config_101.h5 \
  --truth-emitter 46 \
  --target-pulses 12530 \
  --output results/sequential_pdw/hmm_model2/e46_timing_association.json \
  --diagnostics results/sequential_pdw/hmm_model2/e46_timing_association.jsonl

python3 plot_model2_timing_association.py \
  results/sequential_pdw/hmm_model2/e46_timing_association.json \
  results/sequential_pdw/hmm_model2/e46_timing_association.jsonl \
  --output results/sequential_pdw/hmm_model2/e46_timing_association.png
```

## Development replay

The implementation replay used Config 101 SHA-256
`fae9e2a82108ca97c85b5a3501e3a4f1798493f30f7e3eb0cc961d1b537ff8ec`,
base revision `95def874f1668f994445a02dd079c40e2d5a8588` with the new files
present as a dirty working tree, Python 3.14.6, NumPy 2.5.4 and h5py 3.16.0.
The result records all component source hashes, settings and runtime provenance.

There were 10,224 RF-plausible held-out candidates: 5,012 E46 pulses and 5,212
pulses from other emitters. The Frequency + AOA + PW baseline accepted 4,961
E46 pulses and no distractors (precision 100%, recall 98.98%). Positive-only
timing support accepted 4,970 E46 pulses and no distractors (precision 100%,
recall 99.16%). It therefore rescued nine E46 pulses and promoted no distractor
in this replay.

This is evidence that timing can provide a small secondary association benefit
for near-threshold E46 pulses. It does not establish performance under emitter
overlap, an unseen emitter, a mode change, track birth/death, or multiple live
tracks. Frequency + AOA + PW already separated the available E46 candidates
very strongly: the best distractor log-evidence was -80.65 versus the calibrated
threshold of -21.76. The next single investigation is therefore a genuinely
ambiguous held-out case; the timing weight or candidate gate should not be
tuned merely to make this easy replay look better.
