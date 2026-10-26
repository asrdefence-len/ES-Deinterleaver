# Stage A: observation-time nondetection model (experimental)

## Motivation

The earlier `joint_detection_probability` path counts other-emitter
nondetections when a PDW arrives, using a fixed `joint_guard_s`. A missing
pulse is not itself a received event, so this conflates observation-time
progress with association. It also cannot account for a silent interval
unless another PDW arrives.

## New opt-in API

`JointTimingTracker(..., observation_detection_probability=0.7, deadline_sigmas=3.)`

- `process(toa_s, snr_db)`: associate an incoming PDW. The selected
  emitter's intervening, unaccounted pulse slots are treated as misses.
  Other-emitter nondetections are **not** evaluated by this method.
- `advance_time(watermark_s)`: commit an observation interval after **all**
  PDWs with TOA at or before the watermark have been delivered. For every
  retained hypothesis and each emitter, evaluate successive unaccounted
  pulse indices. A slot is missed only when its deadline is strictly earlier
  than the watermark. Repeated advancement does not count the same slot twice.
- PDWs arriving at or before the committed watermark are rejected.
- Existing conditional and joint detection models are preserved unchanged.
  The new model is mutually exclusive with those alternatives.

## Timing deadline

For pulse slot n of an emitter, with Gaussian state (epoch, PRI) and
covariance P, predicted mean is epoch + n PRI and variance is

`P00 + 2*n*P01 + n*n*P11`.

The current **hard** deadline is mean + `deadline_sigmas` times standard
deviation. This is not the full Gaussian probability of nondetection;
it ignores the probability mass in the deadline tail.

## Detection likelihood

Each associated observed pulse contributes `log(Pd)`. Each intervening
or watermark-confirmed missed pulse contributes `log(1-Pd)`. The
`accounted_indices` state is carried independently per global hypothesis.
The beam posterior is renormalised after each observation or time advance.

## Assumptions and known limitations

1. Exactly two **established** emitters; no emitter births, clutter, mode
   changes, or merged/unresolved pulses.
2. Constant, independent Pd across emitters and pulse slots. SNR affects
   TOA variance but not Pd. No receiver blanking or dwell/antenna model.
3. Pulse timing uses a Gaussian epoch/PRI state and a hard uncertainty
   deadline. The deadlines can change after subsequent state updates.
4. An association may skip slots and account for their nondetections at
   the moment of association. This remains a conditional approximation;
   it does not yet implement a complete continuous-time point-process
   observation likelihood.
5. Watermarks must represent **completed** receiver observation windows,
   not simply wall-clock times when buffered PDWs may still arrive.
6. No pulse collisions or out-of-order arrivals. A late PDW after its
   watermark is an explicit error.
7. The beam truncates hypotheses and can affect posterior calibration.
8. No test here demonstrates improved crossing accuracy. That requires
   paired Monte Carlo with explicit watermarks and calibrated evaluation.

## Deterministic regression tests

`tests/test_observation_time.py` covers silent intervals, no duplicate
nondetection penalties, two close pulses, late-PDW rejection, and
mutually exclusive model selection.

## Next experiments

Introduce an event-driven crossing stress harness that advances a
watermark **between** incoming PDWs (never past an undelivered PDW),
then compare the legacy uniform, joint-guard and observation-time models
on the same receiver-generated observations. Measure accuracy, Brier
score, truth survival, and sensitivity to deadline sigma. Avoid claiming
a performance improvement until these paired experiments pass.
