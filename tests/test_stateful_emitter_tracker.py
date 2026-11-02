import pytest

from testbed.pulse_association_pipeline import PulseMeasurement
from testbed.stateful_emitter_tracker import StatefulEmitterTracker


def pulse(index, toa, *, aoa=32., pw=.27e-6, rf=10e9):
    return PulseMeasurement(index, toa, rf, pw, aoa)


def tracker(two=False):
    item = StatefulEmitterTracker(min_confirm_pulses=5)
    item.add_track("E1", time_s=0., aoa_deg=32., pulse_width_s=.27e-6,
                   frequency_modes_hz=[10e9], existence_confidence=.7)
    if two:
        item.add_track("E2", time_s=0., aoa_deg=38., pulse_width_s=1e-6,
                       frequency_modes_hz=[10e9], existence_confidence=.7)
    return item


def test_covariance_prediction_widens_gate_and_confirmed_update_contracts_it():
    item = tracker()
    initial = item.gate("E1")
    widened = item.gate("E1", 1.)
    assert widened.aoa_half_width_deg > initial.aoa_half_width_deg
    decisions = [item.process_pdw(pulse(i, 1.001 + i * .001)) for i in range(1, 6)]
    assert decisions[-1].status == "confirmed"
    contracted = item.gate("E1")
    assert contracted.aoa_half_width_deg < widened.aoa_half_width_deg


def test_separated_tracks_confirm_without_cross_assignment():
    item = tracker(two=True)
    stream = []
    for index in range(5):
        stream.extend([pulse(index + 1, .01 + index * .001),
                       pulse(index + 11, .0104 + index * .001,
                             aoa=38., pw=1e-6)])
    decisions = [item.process_pdw(value)
                 for value in sorted(stream, key=lambda value: value.toa_s)]
    confirmed = {decision.selected_emitter_id for decision in decisions
                 if decision.status == "confirmed"}
    assert confirmed == {"E1", "E2"}
    assert item.tracks["E1"].associated_pdws == [1, 2, 3, 4, 5]
    assert item.tracks["E2"].associated_pdws == [11, 12, 13, 14, 15]


def test_ambiguous_measurement_is_withheld():
    item = StatefulEmitterTracker(min_confirm_pulses=5)
    item.add_track("E1", time_s=0., aoa_deg=32., pulse_width_s=.27e-6,
                   frequency_modes_hz=[10e9])
    item.add_track("E2", time_s=0., aoa_deg=34., pulse_width_s=.27e-6,
                   frequency_modes_hz=[10e9])
    decision = item.process_pdw(pulse(1, .01, aoa=33.))
    assert decision.status == "ambiguous"


def test_snapshot_exposes_confidence_covariance_gates_and_innovations():
    item = tracker()
    for index in range(5):
        item.process_pdw(pulse(index + 1, .01 + index * .001))
    row = item.snapshot()["tracks"][0]
    assert row["existence_confidence"] > .7
    assert row["covariance"]["aoa_variance_deg2"] > 0
    assert row["gates"]["aoa_half_width_deg"] > 0
    assert row["last_innovation"]["association_confidence"] > .7


def test_coasting_expands_covariance_and_reduces_existence_confidence():
    item = tracker()
    before = item.snapshot()["tracks"][0]
    item.coast_to(2.)
    after = item.snapshot()["tracks"][0]
    assert after["existence_confidence"] < before["existence_confidence"]
    assert after["covariance"]["aoa_variance_deg2"] > before["covariance"]["aoa_variance_deg2"]


def test_duplicate_and_out_of_order_pdws_rejected():
    item = tracker()
    item.process_pdw(pulse(1, .01))
    with pytest.raises(ValueError):
        item.process_pdw(pulse(1, .02))
    with pytest.raises(ValueError):
        item.process_pdw(pulse(2, .005))
