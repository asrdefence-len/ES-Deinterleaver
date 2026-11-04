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

The first canonical 5,000-PDW replay at commit `90fdafe` created 64 tracks,
assigned 1,010 later pulses and rejected 3,926 as clutter. Overall measurement
coverage, including track-birth pulses, was 1,074/5,000 (21.48%); labelled
association purity on that selected subset was 99.81%. The track cap was reached
by input row 94, the 64 births represented only 22 truth emitters, and every
track retained exactly one RF mode. Of the rejected pulses, 3,880 actually had
the new-emitter hypothesis as their highest posterior; they were reported as
clutter only because no track capacity remained.

That replay exposed a structural default-prior defect rather than demonstrating
frequency-hop tracking: the unseen-RF-mode score was at most 0.08, while the
new-emitter prior was also 0.08, before AOA, PW and timing likelihoods reduced
the existing-track score. A hop could therefore never beat birth under default
settings. The next isolated version lowers only the new-emitter prior to 0.02
and adds a default-settings regression requiring a compatible frequency hop to
update the original track. Timing, feature models, assignment threshold and
track capacity remain frozen pending the matched 5,000-PDW rerun.

The matched 0.02-prior replay at commit `13c5c76` increased total coverage to
2,482/5,000 (49.64%) while selected association purity remained 99.88%. It also
learned multiple RF modes in 45/64 tracks, so the intended RF-hop path was
actually exercised. The cap still filled by row 128: 64 births represented 24
truth emitters, and 2,364 rejected pulses preferred a new emitter.

Every one of the 40 duplicate births already had a same-truth track. The weakest
likelihood component was timing in 16 cases, pulse width in 14 and RF in 10.
The timing model was using its first assigned inter-arrival gap immediately as
a PRI, which produced near-zero likelihoods for later valid pulses when that
single gap spanned missed pulses or was otherwise unrepresentative. The next
isolated version keeps timing neutral until three assigned gaps support the PRI
estimate and exports that support count. RF/PW models, birth prior, association
threshold and track capacity remain unchanged for the next matched replay.

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

## Controlled E46-only state-estimation test

Before returning to multi-emitter association, isolate E46 from the canonical
recording and pass its raw PDWs directly to the same tracker:

```bash
python3 run_sequential_pdw_tracker.py turing_data/config_101.h5 \
  --isolate-truth-emitter 46 \
  --limit 12530 \
  --output results/sequential_pdw/e46_only/pri_support_3.json \
  --diagnostics results/sequential_pdw/e46_only/pri_support_3.jsonl
```

The label is used only by the runner to construct a controlled input stream.
Original PDW IDs and TOAs are preserved, and the truth-free `PDW` object is the
only object passed to `SequentialPDWTracker`. The output manifest marks the
selection as `evaluation_truth_isolation` and `truth_available_to_tracker` as
false. Purity is trivially 100% in this fixture and is not an acceptance metric;
inspect coverage, number of identities, fragment sizes, RF-mode continuity and
PRI support instead.

The earlier E46 cross-RF plot (71.4% family coverage, eight RF blobs and nine
pulse trains) belongs to a blob/train pipeline. It is historical context, not a
like-for-like acceptance target for this raw-PDW experiment.

The first controlled E46 run at commit `7810cc2` processed all 12,530 pulses,
created the maximum 64 identities, assigned 1,765 later pulses and rejected
10,701. Including births, coverage was 1,829/12,530 (14.60%). The cap filled by
selected pulse 1,170, within the first 0.489 s of E46 input. All 63 births after
the first were necessarily duplicate identities; their weakest likelihood was
RF in 57 cases, PW in three and timing in three.

E46 contains eight approximately 200 MHz-spaced RF populations from 10.0 to
11.4 GHz, but the implementation divided the unseen-mode probability by the
number of already learned modes. With a hop probability of 0.08 and birth prior
of 0.02, a track with four modes assigned at most 0.02 to a fifth before other
likelihoods reduced it. No E46 track learned more than four modes. The next
isolated version treats `rf_hop_probability` as the composite probability of
some previously unseen RF mode, without dividing it by known-mode count, and
adds an eight-mode default-settings regression. All other model stages remain
fixed for the matched E46 rerun.

The matched composite-unseen-mode run at commit `d189851` raised coverage from
14.60% to 18.62% (2,333/12,530) and allowed 12 tracks to learn all eight E46 RF
states. It still created 64 identities, with the last birth at selected pulse
2,071. RF remained the weakest component in 51/63 duplicate births. For a known
RF state, the association score used only the learned one-step transition
probability. A detected stream can skip transmitted states, so a rare observed
transition could have lower probability than track birth even when RF emission,
AOA and PW supported the existing emitter.

The next isolated version retains learned transition preferences but applies
the existing 0.08 hop probability as a backoff floor for rare or skipped
transitions between known RF modes. A regression learns three modes, makes one
transition artificially rare, and requires it to retain the original identity.
PW, timing, birth, threshold and capacity settings remain fixed.

The matched backoff run at commit `b2d190a` failed that acceptance test. E46
coverage fell from 18.62% to 17.87% (2,239/12,530), identities remained at 64,
and RF-driven duplicate births increased from 51 to 54. The backoff is therefore
removed rather than carried forward.

The next control fixes capacity at one track for the already truth-isolated
one-emitter stream. This is not a deployable assumption; it separates emitter
state estimation from duplicate-track competition. Pulses that the sole track
cannot explain remain visibly rejected, so coverage directly exposes whether
the RF/PW/PRI model can follow E46 without manufacturing reassuring fragments.

The one-track control at commit `1bc2283` retained only 205/12,530 E46 pulses
(1.64%). The sole track learned all eight RF modes, but its PW estimate collapsed
onto the approximately 0.27 µs population. In the first 1,000 selected pulses,
502 rejections were PW-driven. After the final accepted pulse at selected index
709, elapsed timing likelihood decayed to zero and the track never recovered;
11,794 total rejections had timing as their weakest component. This is a causal
failure chain: rejecting the other PW regimes starves timing updates, which then
makes even the learned PW regime impossible.

The next isolated version replaces the single log-Gaussian PW estimate with
observation-driven PW modes and a learned transition matrix, analogous to the RF
state model. No E46 configuration values or mode count enter inference. A
controlled regression cycles through three unknown PW populations and requires
one identity to learn all three. RF and timing models remain fixed for the
matched one-track rerun.

The matched PW-mode run at commit `fa1a930` learned the three E46 PW populations
without source configuration, but coverage regressed from 1.64% to 1.46%
(183/12,530) and the last accepted pulse moved from index 709 to 661. PW-driven
rejections in the first 1,000 fell from 502 to 113, while RF-driven rejections
rose from 29 to 400. The cause is the product of separate learned RF and PW
transition probabilities, which treats the two mode chains as independent and
over-penalises otherwise compatible joint changes.

The PW modes are retained, but PW association now uses emission compatibility
only. PW transition counts remain recorded for diagnosis and a future joint
RF/PW state model; they do not multiply the RF transition probability. A
regression makes a known PW transition artificially rare and requires identity
continuity. RF and timing remain unchanged for the next matched one-track run.
