# Model 2 controlled two-emitter competition

This experiment is the first mutually exclusive competition between two raw-PDW
HMM filters. It remains separate from the supported blob pipeline and does not
yet implement automatic emitter birth or a deployable multi-emitter tracker.

E46 and E38 each receive their own truth-isolated controlled seed. For each
emitter, the first 40% of its pulses train Frequency, AOA and PW states; the next
20% freezes a 99%-target-recall evidence threshold. Evaluation begins only after
both seeds end and stops at the earlier target endpoint. The two filters then
receive the same chronological candidate stream.

For every candidate pulse, both filters produce a non-mutating one-step Model 2
predictive score. Each score is centred by that filter's frozen threshold. The
highest non-negative margin wins; if both margins are negative, the pulse is
assigned to neither. At most one filter is updated. Softmax-normalized margins
are recorded as relative hypothesis weights, not claimed as calibrated posterior
probabilities.

The RF candidate envelope is the union of the learned E46 and E38 RF-state
discovery gates. It is a computational pruning boundary only: it uses no AOA,
PW, timing or truth. Pulses outside the envelope are assigned to neither. Truth
is used after the decision for the confusion matrix. A parallel arm adds the
same bounded, positive-only timing support previously tested for E46; all other
models, thresholds and candidates are identical.

Run and plot the experiment with:

```bash
python3 run_model2_two_emitter_competition.py turing_data/config_101.h5 \
  --truth-emitters 46 38 \
  --output results/sequential_pdw/hmm_model2_two_emitter/e46_e38.json \
  --diagnostics results/sequential_pdw/hmm_model2_two_emitter/e46_e38.jsonl

python3 plot_model2_two_emitter_competition.py \
  results/sequential_pdw/hmm_model2_two_emitter/e46_e38.json \
  results/sequential_pdw/hmm_model2_two_emitter/e46_e38.jsonl \
  --output results/sequential_pdw/hmm_model2_two_emitter/e46_e38.png
```

## Development replay

The development replay used the canonical Config 101 recording, base revision
`dc1c8bd92358816eb48442ff86a3523f5141d789` with the new competition files in
a dirty working tree, Python 3.14.6, NumPy 2.5.4 and h5py 3.16.0. The output
records the dataset identity, component source hashes, settings and runtime
provenance.

The common held-out interval contains 332,166 chronological PDWs. The union RF
envelope admits 113,363 candidates, including 4,713 E46 pulses, 4,882 E38 pulses
and 103,768 pulses from other emitters. All E46 and E38 pulses in the interval
are inside the envelope.

With timing off, E46 retains 4,665/4,713 pulses (98.98% recall) and E38 retains
4,816/4,882 (98.65%). With timing on, E46 retains 4,673 (99.15%) and E38 retains
4,831 (98.96%). Timing therefore rescues 23 target pulses: eight E46 and fifteen
E38. Neither arm cross-assigns E46 to E38 or E38 to E46, and neither accepts a
background pulse. Per-emitter precision is 100% in this controlled replay.

This establishes that two already-trained filters can compete sequentially and
update mutually exclusively in this held-out interval. It does not validate
track birth, unknown-emitter discovery, calibrated association probabilities,
long-term lifecycle, or more than two live filters. Those remain the next
architecture boundary; the existing evidence and timing settings should remain
frozen while track initiation is introduced.
