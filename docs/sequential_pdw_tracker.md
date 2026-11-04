# Sequential probabilistic PDW tracker — baseline v0

This is a fresh experimental path, independent of AOA density grouping,
HDBSCAN, blobs, blob summaries and `BlobEmitterTracker`:

**raw time-ordered PDWs → causal probabilistic association → emitter tracks**

For every PDW, the hidden assignment alternatives are each existing emitter,
a new emitter, and clutter. Each emitter maintains online AOA, pulse-width and
PRI distributions plus a learned Markov chain over RF modes. An unseen RF can
therefore be treated as a probabilistic hop rather than automatically creating
a new identity. The number of emitters and their configured radar modes are not
given to the tracker.

This small baseline uses causal MAP filtering. It does **not** yet implement a
joint multi-hypothesis history, retrospective smoothing, track deletion, mode
dwell states, calibrated likelihoods or performance acceptance. Its output is
evidence for the next investigation, not a tracking-improvement claim.

## Dataset boundary

The expected HDF5 schema is:

| Dataset | Meaning |
|---|---|
| `data[N,5]` | columns named by `metadata/feature_names` |
| `metadata/feature_names` | `ToA`, `Frequency`, `PulseWidth`, `AoA`, `Amplitude` |
| `labels[N]` or `labels[N,1]` | optional emitter truth, evaluation only |

Raw units are µs, MHz, µs, degrees and dB-like amplitude respectively. The
reader converts time, frequency and pulse width to SI units. It yields truth as
a separate value after constructing the truth-free `PDW`; the tracker API has
no truth field.

The recorded Config 101 reference is `turing_data/config_101.h5` with SHA-256
`fae9e2a82108ca97c85b5a3501e3a4f1798493f30f7e3eb0cc961d1b537ff8ec`.
Do not silently replace it with `config_101_60s_gaps.h5`, which is a derived
recording with a different pulse count and extension manifest.

## Local run

Start with 5,000 PDWs so that every posterior is easy to inspect:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q tests/test_sequential_pdw_tracker.py
python3 run_sequential_pdw_tracker.py turing_data/config_101.h5 \
  --limit 5000 \
  --output results/sequential_pdw/baseline_5000.json \
  --diagnostics results/sequential_pdw/baseline_5000.jsonl
```

The JSON includes the active component path, settings, Git revision/dirty
state, dataset hash, dependency versions, every final track, truth coverage,
all identity fragments and per-track purity. The JSONL contains the posterior
and component likelihoods for every incoming PDW. Share the console output and
the JSON result for the next single-stage investigation.
