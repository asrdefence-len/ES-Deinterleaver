# Integration status

## Active sequential architecture: Model 1 reference and Model 2 PW increment

Incremental patching of the causal MAP surrogate has stopped. It multiplied
uncalibrated compatibility scores, committed immediately to track/birth/clutter
and did not retain an HMM state posterior. Its results remain below as failure
evidence, not the active tracker architecture.

The active implementation is [HMM Model 1](hmm_tracker_model1.md): raw E46 PDWs
selected outside inference update one state filter using Frequency, TOA and AOA.
RF uses a full forward posterior and expected transition counts; AOA uses a
circular continuous Markov filter. TOA orders the sequence and supplies elapsed
time to AOA prediction. A separate zero-feedback timing shadow describes the
received pulse train but does not assert transmitter PRI or contribute active
evidence. Every E46 pulse updates the filter. PW and amplitude are explicitly
excluded and tested for invariance. This is emitter-state validation, not yet
multi-emitter association.

The user has reviewed the timing boundary and requested
[Model 2](hmm_tracker_model2.md). Model 2 changes only one component: it adds a
separate log-PW HMM with soft state and transition updates. It subclasses Model
1, so RF, AOA and the timing shadow execute through the unchanged implementation.
Amplitude remains excluded. Birth, clutter and bounded multi-emitter hypotheses
remain outside this milestone.

On a matched 500-pulse canonical E46 wiring smoke, Model 2 discovers three PW
states at 0.101, 0.271 and 0.427 µs. All saved Model 1 RF, AOA and timing fields
over the same 500 pulses are exactly unchanged. Focused Model 1/Model 2 tests
pass. This is component implementation evidence, not PW association acceptance;
the clean complete 12,530-pulse user replay remains pending.

The first complete Model 1 E46 run processed all 12,530 selected pulses, learned
eight RF states, retained a normalized RF transition matrix and ended with an
AOA estimate of 32.238 degrees with 0.149-degree posterior sigma. Its TOA bank
failed identification: the retained hypotheses collapsed near 2.032 microseconds,
a harmonic/submultiple with no model preference for the fundamental PRI. RF and
AOA are therefore frozen while timing is redesigned; PW remains blocked. The
diagnostic performance plot reports AOA covariance, RF posterior entropy, the
visible timing estimate and rolling log evidence. It must not present covariance
contraction alone as accuracy or performance acceptance.

The next timing-only implementation replaces hard nearest-interval updates with
a visibility-aware hidden semi-Markov posterior. An active branch marginalizes
plausible integer interval counts under a learned missed-detection probability;
a coast branch preserves PRI across scan/visibility gaps. Near-duplicate PRI
particles are moment-merged. RF, AOA and feature exclusions are unchanged. In a
500-pulse canonical E46 schema smoke, the leading timing posterior moved from
1.170 microseconds in the saved pre-change diagnostic at pulse 500 to 10.658
microseconds, with 11 distinct hypotheses and finite normalized probabilities.
At that same pulse, the RF posterior, RF evidence, AOA mean, AOA covariance,
AOA innovation and AOA evidence are exactly unchanged. The old complete run
later ended at the failed 2.032-microsecond submultiple. This is implementation
and wiring evidence only; the matched 12,530-pulse clean replay remains pending.

That clean replay at commit `3eb8383` processed all 12,530 pulses and retained a
single effective timing state at 10.633 microseconds with 0.027-microsecond
posterior sigma. This accurately matches one evaluation-only E46 configuration
value (10.637 microseconds), but the same metadata lists two additional jittered
values at 8.572 and 6.697 microseconds. The model therefore passed one-state
identification but failed the requested timing-agility topology.

The next timing-only revision at commit `be15d5b` treated retained candidates as
recoverable HMM states with learned soft transitions. The user's complete E46
run retained direct-support states at 4.796, 6.986, 8.756, 11.076, 16.871 and
32.157 microseconds. The 32.157-microsecond state acquired substantial support
even though an observed gap may be a sum of shorter intervals. This failed the
required distinction between detected inter-arrival time and transmitter PRI.

Commit `4738539` removes all timing feedback rather than adding state-pruning
rules. RF and AOA remain bit-for-bit unchanged over the first 500 E46 pulses. A
new received-train shadow records an empirical gap density and generic
short/long visibility state, with feedback fixed at zero and
`transmitter_pri_estimate` fixed to null. The user's clean complete replay
processed all 12,530 E46 pulses, learned eight RF states, ended at AOA 32.238
degrees with 0.149-degree posterior sigma, and recorded 12,529 shadow gaps. Its
received train shows an approximately 0.488-second visibility cycle and a broad
63.499 µs median adjacent gap. PW remains excluded.

The next single timing component is a held-out phase-coherence shadow. It fits
one dominant interval lattice on the first 60% of causally classified active
gaps, freezes the fit, and scores the last 40% against uniform phase. It does not
change the HMM or association evidence. The code-development replay selects
10.638 µs with 2.001 µs phase sigma. Of 4,090 held-out active gaps, 3,186 are
scoreable; mean phase log-likelihood ratio is +0.222, 76.0% are positive, and a
half-cycle negative control averages -1.116. This is modest predictive evidence,
not timing acceptance: 22.1% of active gaps are unscored and 24.0% of scored
gaps are negative. Timing feedback remains zero pending a bounded-weight
association ablation.

## Parallel experiment: sequential raw-PDW tracker

Work on `codex/sequential-pdw-tracker` introduces a separate experimental path:
raw time-ordered PDWs enter a causal probabilistic multi-emitter tracker without
AOA grouping, HDBSCAN, blobs or `BlobEmitterTracker`. This does not replace or
recalibrate the integrated E46-only baseline below. See
[`sequential_pdw_tracker.md`](sequential_pdw_tracker.md) for its explicit model,
dataset boundary, limitations and local run procedure. Performance acceptance
is open until a clean Config 101 replay is supplied and reviewed.

The v0 implementation and its focused tests pass; the full repository result
at its initial checkpoint was **243 passed**. A 500-PDW wiring smoke test used the explicitly different
`config_101_60s_gaps.h5`: it reached the configured 64-track limit, assigned
130 pulses and left 306 as clutter. This is recorded as an unaccepted smoke
result, not silently treated as the Config 101 benchmark. The next single
investigation was the user's 5,000-PDW run on the canonical `config_101.h5`.

That canonical replay used dataset SHA-256
`fae9e2a82108ca97c85b5a3501e3a4f1798493f30f7e3eb0cc961d1b537ff8ec`
and commit `90fdafe`. It created 64 tracks, assigned 1,010 later pulses and
rejected 3,926; including births, coverage was 1,074/5,000 (21.48%) with 99.81%
selected association purity. The cap was reached by row 94, the births covered
only 22 of the 45 truth emitters present, and all 64 tracks had one RF mode.
Posteriors show that 3,880 rejected pulses preferred a new emitter but could not
be born at capacity. The default unseen-mode score could not exceed the equal
0.08 birth prior after other likelihoods were applied, so the advertised RF-hop
path was structurally unable to win. The next isolated change lowers only the
birth prior to 0.02 and adds a default-configuration hopping regression; all
other association and state settings remain frozen for the matched rerun.

The matched 0.02-birth-prior replay at commit `13c5c76` raised coverage from
21.48% to 49.64% (2,482/5,000 including births), retained 99.88% selected
association purity, and produced multiple RF modes in 45/64 tracks. This is a
real improvement in the intended hop path, not performance acceptance: the cap
still filled by row 128, only 24 truth emitters received births, and 2,364
rejections preferred a new emitter. Analysis of all 40 duplicate births found
timing as the weakest component in 16, PW in 14 and RF in 10. The next single
change delays use of the online PRI likelihood until three assigned gaps support
it; all other components and settings remain fixed for another matched replay.

## Active sequential focus: controlled E46-only input

The user selected E46 for the next component-isolation stage. The canonical
recording's truth label is used outside inference to select E46 PDWs in original
TOA order; the tracker still receives only truth-free raw `PDW` objects. This is
a state-estimation and continuity test, not a deinterleaving benchmark, and its
purity is trivially 100%. Acceptance evidence must show coverage, every identity
fragment, RF-mode continuity and PRI support. The historical 71.4% E46 cross-RF
family plot came from the blob/train route and is not a like-for-like target.

The first controlled raw-PDW E46 run at commit `7810cc2` processed all 12,530
truth-isolated pulses and produced 64 identities with 14.60% total coverage
(1,829 pulses including births). The cap filled by selected pulse 1,170, within
0.489 s. All 63 births after the first were duplicate E46 identities; RF was the
weakest likelihood in 57, PW in three and timing in three. E46 contains eight RF
populations, while no track learned more than four because the unseen-mode mass
was divided by known-mode count: after four modes it could not beat the 0.02
birth prior. The next single change preserves the full 0.08 composite unseen-RF
mass and adds an eight-mode regression. Timing, PW, thresholds and capacity stay
fixed for a matched E46 rerun.

The matched run at commit `d189851` increased E46 coverage from 14.60% to
18.62% (2,333/12,530), and 12 tracks learned all eight RF modes. Fragmentation
remained at 64 identities; the last birth moved to selected pulse 2,071 and RF
was still the weakest component in 51/63 duplicate births. Known-mode scoring
used the learned one-step transition probability without allowance for skipped
transmitted states. The next RF-only change uses the existing 0.08 hop
probability as a backoff floor for rare known-mode transitions. No PW, timing,
birth, threshold or capacity setting changes for the matched rerun.

The RF-transition backoff at commit `b2d190a` regressed matched E46 coverage
from 18.62% to 17.87% (2,239/12,530), retained 64 identities and increased
RF-driven duplicate births from 51 to 54. It is removed. The next controlled
run sets `max_tracks=1` for the truth-isolated one-emitter stream to separate
state-model coverage from competition among duplicate fragments. Rejected
pulses remain missing; this control does not force association or claim
deployable knowledge of emitter count.

The one-track control at commit `1bc2283` retained 205/12,530 E46 pulses (1.64%).
The track learned all eight RF modes but only the approximately 0.27 µs PW
population. PW caused 502 rejections in the first 1,000 selected pulses; after
the last accepted pulse at index 709, timing decayed to zero and never recovered.
The next single component change replaces the scalar PW distribution with
learned PW modes and transitions. It uses no source configuration or truth mode
count. RF and timing models remain unchanged for the matched one-track rerun.

The matched PW-mode run at commit `fa1a930` learned all three E46 PW populations
but regressed coverage from 1.64% to 1.46% (183/12,530). In the first 1,000
pulses, PW-driven rejections fell from 502 to 113 while RF-driven rejections
rose from 29 to 400. Multiplying independent RF and PW transition probabilities
over-penalised joint mode changes. PW modes remain, but only PW emission
compatibility enters association; PW transitions remain diagnostic until a
joint state model is implemented. RF and timing stay fixed for the matched run.

## Milestone: restore the intended processing chain

The experimental state was preserved at Git checkpoint `1465b5b`.
Integration work is on `codex/integrated-aoa-hdbscan-pri-tracker`.
No changes have been pushed to GitHub.

| Component | Supported integration | Acceptance evidence |
|---|---|---|
| AOA + HDBSCAN | Circular AOA populations followed by existing RF/PW HDBSCAN (8/8) | Default-runner test forbids entering legacy multiscale grouping |
| PRI estimator | Shared original local mean/spread and optional second hump | Known two-hump test; diagnostic and adapter share one implementation |
| Observation adapter | One bounded summary per frozen blob, or PRI-absent fallback | Membership, metadata and unit checks; no fixed-train/phase fabrication |
| Kalman tracker | AOA/PW/gap-mean prediction, covariance, joint NIS and update | Analytical mean/covariance check; no harmonic reinterpretation of gap means |
| Basic management | Birth, existence, coast, lost/stale archive, ID continuity and limits | Missed-opportunity, covariance growth, archival and non-reused-ID tests |
| Truth isolation | Truth used for evaluation only | Relabelling truth leaves complete tracking snapshots unchanged |
| Configuration control | Run manifests and Git milestones | Component names, settings, code/data hashes and dependency versions saved |

## Performance acceptance remains open

Connecting the requested components does not establish calibration or complete
emitter tracking. The full Config 101 E46/E38 integration replay exposes severe
fragmentation, especially for E38. This is recorded as a failed retention
benchmark, not described as successful tracking.

RF/PW HDBSCAN splits E46's distinct PW populations and partitions E38's broad RF
support. Observed-gap distributions can therefore differ between blobs from
the same emitter. One scalar gap-mean state does not automatically represent
those different sampling patterns. Sparse blobs may have no PRI estimate at
all. Both source emitters are configured with jittered timing. These are known
limits to investigate with the now-explicit chain; do not silently substitute
another grouping method or inflate gates to make retention appear better.

The general runner's default maximum gap is 0.3 s; seeded development runners
use 10 s. Their Bernoulli detection/lifecycle parameters are nominal. Dormant
identity memory and archived-track reacquisition remain outside this milestone.

## Working process

For each change, state which component changes and its acceptance criterion.
Run the relevant integration/regression checks, compare a recording and a
controlled fixture against the saved baseline, and commit the result.
Record implemented, integrated and performance-validated status separately.
Update the supported-path documentation and manifests whenever a component or
its measurement meaning changes. Historical scripts are experiments unless
listed as supported entry points in `blob_tracking.md`.

## Recorded baseline — 2 November 2026

Code revision: `4c3d5be` (clean working tree during all three replays).
Regression result: **239 passed**, including five integration tests.
The machine-readable [baseline record](integration_baseline.json) contains the
input/code hashes, dependency versions, artifact checksums and numerical results.

| Replay | Recorded result |
|---|---|
| All grouped data, first four Config 101 windows | 37,782 grouped PDWs; 457 whole-blob observations; 235 carry a PRI histogram; membership preserved |
| E46/E38 controlled families, all 85 windows | 586 blob observations; 348 carry a PRI histogram; zero canonical tracks contain both target truth labels |
| E46 single-family, all 85 windows | Runner completes; original primary has 139 updates and is absent at the recording end |

The two-family replay creates **33 canonical track IDs touching E46 pulses and
22 touching E38 pulses**. These are fragments, not counts of physical emitters.
Only 142/433 E46 observations and 6/153 E38 observations remain on the respective
original primary IDs after consolidation. The denominators are whole HDBSCAN
blobs, unlike the legacy fixed-train benchmark, so the old retention fractions
are not directly comparable.

Selection coverage is 93.43% for E46 and 64.50% for E38; selected-pulse purity is
100% and 99.66%, respectively. E38 starts from its Window 3 seed, whereas the
coverage denominator includes the entire recording. These metrics describe
this controlled selection and do not establish whole-scene detection quality.

Integration acceptance is recorded separately from the still-open emitter
retention and calibration acceptance. The next performance investigation is
identity fragmentation across RF/PW blobs and their differently sampled gap
statistics, using this saved configuration as the comparison point.

To reproduce the baseline plots:

```bash
python3 plot_integration_baseline.py results/integrated_baseline/moderate.json
```

`identity_fragmentation.png` shows all canonical IDs associated with each
truth emitter's pulses, including fragments outside the original primary.
`pri_histogram_inputs.png` shows the actual blob-level selected-gap mean,
standard deviation, hump centres and selected-gap support. These gap statistics
must not be interpreted as transmitter fundamental PRI.

## Confidence view and handoff correction

The integrated path is a connected **experimental** path, not an accepted
performance replacement for the earlier run. Grouping and PRI representation
changed together during the initial integration. That replay cannot attribute
a downstream regression to either individual change. Future comparisons must
change one stage at a time with matched inputs and explicit measurement accounting.

The confidence calculation remains in `BlobEmitterTracker`; the fragmentation
plots did not display it. The original E38 primary in the integrated replay is
last present at about 0.67 s. An absent primary is not a zero-confidence value,
and selecting a different surviving fragment would change the plotted identity.
The earlier saved confidence reference remains available locally at
`results/moderate_two_emitter_blob_tracker.json` (E38 56/75 retention).

```bash
python3 plot_confidence_comparison.py \
  results/moderate_two_emitter_blob_tracker.json \
  results/integrated_baseline/moderate.json
```

This restores both recorded confidence views with explicit processing-path
labels; it is not a like-for-like validation comparison. `AGENTS.md` records
project-level handoff instructions so future work starts from these files,
versions and known failures. The next investigation should isolate measurement
changes between grouping variants before changing filter/lifecycle parameters.

## Active version: E46-only v1

The user has set E38 aside to establish E46 first. The isolated version is on
`codex/e46-only-baseline`; [its run instructions](e46_only_baseline.md) and
[machine-readable record](e46_only_baseline.json) identify the result and checks.

The clean replay used code revision `16b0f64`. Only E46's candidate family
enters the tracker; grouping still sees the unchanged full Config 101 input.
Verification confirms the same dataset, grouping/PRI settings, runtime component
sources, E46 seed, all 433 E46 observation inputs and tracker parameters as the
preceding integrated two-family replay. No filter or lifecycle tuning occurred.
Targeted integration/tracker tests: **36 passed**.

The replay produces 32 canonical identities and retains 135/433 observations on
original primary E1. Selection purity is 100%, with 93.43% of labelled E46
pulses selected. The original primary is last present at about 5.91 s; the
last selected observation is at about 5.44 s. E46 truth pulses still occur in
the final recording window, so the late absence must not be described as a
verified emitter shutdown. Fragmentation and missed sparse coverage remain
open issues. Removing E38 changes beam competition, so identity assignments
may differ even though the E46 measurement inputs are identical.

`results/e46_only/v1/e46_overview.png` keeps confidence, every assigned identity,
PRI input statistics and pulse coverage together. Previous two-emitter results
remain untouched. Do not compare against legacy grouping as a success target;
continue investigation of this isolated, intended path one change at a time.

## Separate raw-PDW HMM experiment: Model 2 timing ablation

The raw-PDW HMM work is a separate experimental path and does not replace or
validate the supported blob pipeline above. Model 2 adds three learned E46
pulse-width states to the Frequency + AOA model. Its first held-out association
test now compares the same trained one-track model on the same mixed RF-plausible
candidate stream with phase timing support off and on.

The Config 101 development replay contains 5,012 held-out E46 candidates and
5,212 distractors. Frequency + AOA + PW alone accepts 4,961 E46 pulses and zero
distractors. Bounded positive-only timing support rescues nine more E46 pulses
without promoting a distractor: recall changes from 98.98% to 99.16% and
precision remains 100%. Truth is used only for the controlled seed and
post-decision evaluation; both arms are truth-blind and candidate selection is
RF-only. The result is promising but easy because the best distractor remains
far below threshold. It is not multi-emitter acceptance.

The method, exact run commands, provenance and limitations are in
[Model 2 timing association ablation](model2_timing_association.md). The next
single investigation is a genuinely ambiguous held-out emitter/candidate case,
without retuning the timing weight, evidence model or candidate envelope.

## Separate raw-PDW HMM experiment: two-emitter competition

The next raw-PDW step freezes both Model 2 filters, their independently
calibrated thresholds and the timing policy, then changes only the decision
topology: E46, E38 and neither compete for one mutually exclusive update per
pulse. Controlled truth selects each training seed but is unavailable to both
competition arms. A union RF envelope performs candidate pruning without AOA,
PW, timing or truth.

The Config 101 development replay covers 332,166 chronological held-out PDWs;
113,363 enter both filters. Timing off retains 4,665/4,713 E46 pulses and
4,816/4,882 E38 pulses. Timing on rescues eight E46 and fifteen E38 pulses, for
4,673 and 4,831 retained respectively. Both arms have zero E46/E38 cross-
assignments and zero accepted background pulses. This is controlled competition
between seeded filters, not automatic multi-emitter tracking acceptance.

The exact boundary, commands, provenance and limitations are recorded in
[Model 2 controlled two-emitter competition](model2_two_emitter_competition.md).
The next single component is track initiation/birth from the unlabelled stream;
the evidence, timing and competition settings should remain frozen.

## Separate raw-PDW HMM experiment: automatic birth control

Automatic initiation is now isolated on a truth-selected but tracker-unlabelled
E46/E38 mixture. No HMM exists initially. Tentative tracks use only a fixed
3-degree AOA-continuity gate while their Model 2 RF/PW states are learned; after
256 pulses, each freezes a threshold from its own last 128 predictive scores and
enters the existing competition. Near-AOA rejections are not allowed to create
duplicate tracks.

The Config 101 development replay processes all 24,741 interleaved target
pulses. Both timing arms create exactly two confirmed identities: T1 maps to E38
and T2 to E46, with 100% assigned-pulse purity and one fragment per emitter.
Timing on retains 12,409/12,530 E46 pulses and 11,877/12,211 E38 pulses, with
zero cross-assignments. It rescues 29 pulses relative to timing off and loses
none. This is a strong initiation-control result, not a full-scene benchmark:
all background emitters were excluded before the tracker.

The exact staged model, commands, provenance, online-timing disagreement and
limitations are in [Model 2 automatic birth](model2_automatic_birth.md). The
next single change is adding background PDWs while holding birth, evidence,
threshold and timing settings fixed.

## Separate raw-PDW HMM experiment: frozen full-scene birth stress

The matched full-scene replay has now added all 844,999 chronological Config
101 PDWs without changing the Model 2 evidence, timing, initiation thresholds,
duplicate suppression or eight-track limit. Truth is unavailable to the
tracker and is used only after decisions for evaluation.

This replay fails target retention. The first nine input pulses consume all
eight track slots, before any tentative track can reach the 256-pulse
confirmation requirement. Although all eight tracks later confirm and the
tracker assigns 507,338 PDWs (60.04% overall coverage), it assigns zero of
12,530 E46 pulses and zero of 12,211 E38 pulses. Capacity rejects 323,399 PDWs,
and two retained tracks have only 80.7% and 71.1% evaluation purity. The overall
coverage must not be reported as tracking acceptance.

The exact boundary, commands, result and limitations are recorded in
[Model 2 frozen full-scene birth stress](model2_full_scene_birth.md). The next
single investigation is bounded tentative-hypothesis admission and eviction:
unconfirmed one-pulse births must earn a permanent slot, while Model 2 evidence,
timing and confirmed-track competition remain frozen.

## Separate raw-PDW HMM experiment: 256-track capacity diagnostic

Before changing lifecycle, a matched diagnostic changes only the full-scene
capacity from 8 to 256. Capacity is never reached: the tracker creates 36
identities, confirms 24 at the existing 256-pulse requirement and leaves 12
tentative. It assigns 98.84% of all PDWs, including 99.60% of E46 and 99.97% of
E38.

This apparent recovery is not full-scene tracking acceptance. Evaluation-
weighted identity purity is only 62.98%. Eighteen tracks contain multiple truth
emitters, eight fall below 90% purity, and the largest track contains 528,221
PDWs from 25 truth emitters at only 47.03% purity. Its frozen threshold is near
`-3477.4`, showing that a mixed AOA-only tentative history can confirm an
extremely permissive catch-all track.

The exact command, result and limitations are in
[Model 2 high-capacity birth diagnostic](model2_high_capacity_diagnostic.md).
The next single change is confirmation validation using held-out Frequency +
AOA + PW evidence. Lifecycle expiry, retirement and reacquisition follow as a
separate stage after confirmation is trustworthy.

## Separate raw-PDW HMM experiment: staged birth validation

Tentative birth now uses an eight-pulse AOA seed, early RF/PW discrimination,
a frozen 128-pulse candidate model and a rolling 128-pulse held-out validation
block. RF, AOA and PW each require coherent held-out evidence. Inconclusive
blocks keep the same frozen candidate rather than creating repeated identities;
pathological training references reject birth, and pathological held-out
quantiles are replaced by a robust threshold. Mature HMM equations, timing and
confirmed competition remain frozen.

The full Config 101 development replay creates 44 historical candidates: 32
confirmed, 11 tentative and one rejected. Assignment coverage is 96.97%.
Weighted identity purity improves from the matched high-capacity reference's
62.98% to 95.03%; distinct dominant emitters increase from 35 to 42; and the
maximum truth emitters in one track falls from 25 to four. Seven tracks remain
below 90% purity, so performance acceptance remains open.

E46 remains one identity with 99.61% primary retention and 99.94% purity. E38
has 99.43% total coverage; its primary retains 98.73% at 99.78% purity. The
eleven previously strong identities survive, although E33 primary retention
regresses from 97.01% to 94.23%. An AOA-gate-only validation variant reduced
weighted purity to 94.69% without resolving a large pure tentative track and
was removed.

The exact design, command and limitations are in
[Model 2 staged birth validation](model2_staged_birth_validation.md). The next
single component is lifecycle management; HMM, timing and validation settings
remain frozen.
