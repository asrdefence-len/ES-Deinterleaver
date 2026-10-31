# Temporal optimal transport trial — 1 November 2026

The feature helps regroup frequency fragments sharing a scan envelope, but
also falsely joins different emitters with similar temporal activity. Treat
it as supporting evidence, not an emitter-identity decision.

## Method

Existing batch circular AOA grouping and multiscale RF segmentation produce
frozen blobs. All methods use exactly those blobs. The added score compares
128-bin common-time histograms weighted by positive linear received power.
dB measurements are converted using 10**(dB/10). W1 is normalized by the
duration of the whole observation window, preserving temporal offsets.

Compared: frozen blobs, OT-only candidate merges, and AOA-gated OT merges.
The threshold is 0.03 of window duration, with a 3-degree circular AOA gate
and minimum 100 pulses per blob. Complete-link grouping requires every
cross-pair to pass, preventing simple transitive chaining. Sparse blobs
remain separate. Labels are used only for evaluation after candidate
grouping. No emitter count is supplied to the algorithm.

This is inspired by Mottier, Chardon and Pascal, *Deinterleaving RADAR
Emitters With Optimal Transport Distances*, TAES 60(3), 2024,
DOI 10.1109/TAES.2024.3367287. It is not a HACOT/IHACOT reproduction:
fixed bins, a fixed threshold, and complete-link grouping replace the
paper's bin selection, hierarchical aggregation and statistical pruning.
It does not yet combine OT with pulse-train evidence or tracker outputs.

## Initial results

Seven synthetic cases, each with two emitters and 20 seeds (100–119).
Parameters were fixed before this run; a preceding three-seed pilot used
seeds 0–2 with the same parameters. This is an exploratory stress test,
not a representative operational benchmark or a tuned performance claim.

Means across seeds:

| Case | Frozen groups | OT groups | OT purity % | AOA + OT groups | AOA + OT purity % |
|---|---:|---:|---:|---:|---:|
| Shared scan, frequency hopping | 6 | 4.10 | 100.00 | 4.10 | 100.00 |
| Same envelope, same AOA | 6 | 2.85 | 65.62 | 2.85 | 65.62 |
| Same envelope, different AOA | 6 | 2.85 | 65.62 | 4.15 | 100.00 |
| Sequential frequency changes | 6 | 6.00 | 100.00 | 6.00 | 100.00 |
| Continuous activity, same AOA | 6 | 1.35 | 55.88 | 1.35 | 55.88 |
| Sparse hopping | 6 | 6.00 | 100.00 | 6.00 | 100.00 |
| Mixed initial RF blobs | 3 | 1.70 | 51.13 | 1.70 | 51.13 |

Frozen purity is 100% except the deliberately mixed input case (51.39%).
Shared-scan mean fragments per emitter fall from 3 to 2.05; ARI rises from
0.333 to 0.583. This is partial recovery, not complete deinterleaving.
AOA gating removes false cross-bearing associations in the constructed
different-bearing case. It cannot resolve identical activity at the same
bearing. Sequential switches are missed because activity does not overlap.
Merging cannot repair a blob already containing two emitters.

Coverage stays fixed because this experiment only merges existing labels.
Purity alone rewards fragmentation; inspect groups, fragmentation and ARI
together. Millisecond score times on small synthetic windows do not establish
real-time throughput on operational streams.

## Reproduce and validate

```bash
python -m pip install -r requirements-experiments.txt pytest
python trial_optimal_transport.py --seeds 20 --seed-start 100
python -m pytest -q
```

The generated comparison, per-pair diagnostic and configuration are archived
under `docs/experiments/optimal_transport/`. Source baseline:
`cf3a29f86a04d26c064c0671e68fdd7ff674b1ca`.
Environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
scikit-learn 1.8.0, h5py 3.16.0. Validation: 176 tests passed,
including six new OT checks against an independent SciPy W1 reference,
circular AOA, dB-to-power equivalence, clock preservation, sparse inputs,
and transitive false merging. The HDF5 CLI was also exercised on synthetic
labelled and unlabelled files; this does not constitute Turing validation.

The real Turing recordings were unavailable. Next, run the same comparison
on fixed recording windows with their documented amplitude units, then
evaluate whether OT improves the existing timing/tracking association when
combined with it. Vary bin width and threshold on development data before
freezing parameters for held-out recordings. Retain ambiguous hypotheses
where temporal similarity alone cannot identify the emitter.
