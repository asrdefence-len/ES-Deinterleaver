import pytest

from testbed.pulse_association_pipeline import PulseMeasurement
from testbed.track_aided_pdw_association import (
    TrackAidedPDWAssociator, TrackPrediction,
)


def prediction(name="E1", *, aoa=32., last=0., phase=0.):
    return TrackPrediction(name, last, last, (10e9, 10.2e9), aoa, .27e-6,
        pri_s=85e-6, phase_epoch_s=phase, pulse_width_sigma_s=.14e-6)


def pulse(index, toa, *, rf=10e9, aoa=32., pw=.27e-6):
    return PulseMeasurement(index, toa, rf, pw, aoa)


def test_sparse_compatible_pdws_stay_provisional_until_batch_confirmation():
    gate = TrackAidedPDWAssociator(min_confirm_pulses=5)
    decisions = [gate.process(pulse(i, .5 + i * .001), [prediction()])
                 for i in range(1, 6)]
    assert [item.status for item in decisions] == [
        "provisional", "provisional", "provisional", "provisional", "confirmed"]
    assert decisions[-1].confirmed_pdw_ids == (1, 2, 3, 4, 5)


def test_long_gap_uses_signal_features_without_claiming_timing_evidence():
    gate = TrackAidedPDWAssociator()
    decision = gate.process(pulse(1, .5), [prediction()])
    assert decision.status == "provisional"
    assert decision.candidates[0].timing_score is None
    assert decision.candidates[0].score > .99


def test_incompatible_pulse_is_left_for_discovery():
    gate = TrackAidedPDWAssociator()
    decision = gate.process(pulse(1, .5, rf=12e9, aoa=70.), [prediction()])
    assert decision.status == "unclaimed"
    assert decision.selected_emitter_id is None


def test_ambiguous_track_match_is_withheld():
    gate = TrackAidedPDWAssociator(ambiguity_margin=.1)
    decision = gate.process(pulse(1, .5, aoa=33.), [
        prediction("E1", aoa=32.), prediction("E2", aoa=34.)])
    assert decision.status == "ambiguous"


def test_duplicate_and_out_of_order_pdws_are_rejected():
    gate = TrackAidedPDWAssociator()
    gate.process(pulse(1, .5), [prediction()])
    with pytest.raises(ValueError):
        gate.process(pulse(1, .6), [prediction()])
    with pytest.raises(ValueError):
        gate.process(pulse(2, .4), [prediction()])


def test_two_close_tracks_confirm_separately_and_midpoint_pulses_are_ambiguous():
    gate = TrackAidedPDWAssociator(min_confirm_pulses=5, ambiguity_margin=.08)
    left = prediction("E1", aoa=32.)
    right = TrackPrediction("E2", 0., 0., (10e9, 10.2e9), 34.4, .32e-6,
        pri_s=87e-6, phase_epoch_s=0., pulse_width_sigma_s=.14e-6)
    stream = []
    for i in range(5):
        stream.append(pulse(10 + i, .5 + i * .001, aoa=32., pw=.27e-6))
        stream.append(pulse(20 + i, .5004 + i * .001, aoa=34.4, pw=.32e-6))
    stream.append(pulse(30, .5022, aoa=33.2, pw=.295e-6))
    decisions = [gate.process(item, [left, right])
                 for item in sorted(stream, key=lambda item: item.toa_s)]
    confirmed = {item.selected_emitter_id for item in decisions
                 if item.status == "confirmed"}
    midpoint = next(item for item in decisions if item.pdw_id == 30)
    assert confirmed == {"E1", "E2"}
    assert midpoint.status == "ambiguous"
    assert midpoint.selected_emitter_id is None
