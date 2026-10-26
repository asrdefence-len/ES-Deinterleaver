# NRL-inspired PDW deinterleaver performance measures

## Literature context

Thompson, Lin and Sciortino (2006), *Measures of effectiveness for analysis
of radar pulse train deinterleavers*, is a useful reference for evaluating
correct association, incorrect association, ambiguity, and track purity.
The metrics below are **our operational definitions inspired by that work**;
they are not asserted to reproduce the paper's exact formulas or protocol.

## Reporting partition (online, thresholded)

For each received PDW, calculate the marginal emitter probabilities from
the Bayesian beam. If the maximum probability is below the reporting
threshold, label the *reported* association ambiguous; otherwise report
the most probable emitter. With known simulated truth:

- `nrl_style_reporting_correct_fraction`: confidently reported and correct / all PDWs.
- `nrl_style_reporting_miscorrelated_fraction`: confidently reported and wrong / all PDWs.
- `nrl_style_reporting_ambiguous_fraction`: not confidently reported / all PDWs.

These three mutually exclusive fractions sum to 1. An ambiguous PDW is
**not rejected from tracking**: it still updates the Bayesian hypotheses.
The reporting threshold is not an empirically calibrated probability of
correctness. These are not measures of unknown-emitter rejection.

## Track purity (retrospective)

After the final PDW, take the most probable retained joint hypothesis
(MAP history). Within **each trial**, construct a 2x2 contingency matrix
between its emitter labels and the simulated true emitter labels.
Track purity is:

    sum_over_predicted_tracks(max_true_emitter_count_in_track) / total_PDWs

The final reported value is the PDW-weighted average across trials.
Track purity is permutation-invariant, unlike the online association
accuracy, which uses the established A/B identity labels. This is a
retrospective measure on the final MAP *history*, not a real-time score.
A swapped but internally pure pair of tracks can score high purity even
though its emitter identities are wrong. Report accuracy alongside purity.

`final_map_track_contingency` is pooled across trials for inspection,
but **must not** be used to recompute pooled purity: track labels may
permute independently across trials.

## Current limitations

Only two known emitters, ideal pulse resolution, no spurious PDWs, no
new-emitter births, no unknown-emitter hypothesis. Timing-only
measurements. Beam truncation and simplified skip priors may miscalibrate
posterior probabilities. The simulated ground truth is never used by the
tracker. No operational ESM performance claim follows from these tests.

## Reproduce

    git pull
    python -m pytest -q
    python paired_pdw_quality.py --scenario very_close --trials 100 --pd 0.70 --beam 128 --sigmas-ns 50 100 250

The three `nrl_style_reporting_*` fractions are disjoint, whereas
`accuracy` scores every PDW's maximum-posterior label, including those
whose reported association is ambiguous.
