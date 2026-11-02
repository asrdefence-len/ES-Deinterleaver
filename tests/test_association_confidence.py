from testbed.association_confidence import AssociationConfidenceTracker
from testbed.pulse_association_pipeline import AssociatedTrain, PulseMeasurement
from testbed.pulse_train_observation import PulseTrainObservation


def pdw(index, toa, *, rf=10e9, aoa=32., pw=.27e-6):
    return PulseMeasurement(index, toa, rf, pw, aoa)


def train(count=20, *, pri=85e-6, rms=2e-6):
    ids = tuple(range(count))
    obs = PulseTrainObservation(1, "T1", 0., .1, pri, None, 10e9, None,
        32., None, .27e-6, None, count, ids)
    return AssociatedTrain(obs, 1, 0., rms, count, count)


def tracker():
    return AssociationConfidenceTracker(expected_rf_hz=[10e9, 10.2e9],
        expected_aoa_deg=32., expected_pw_s=.27e-6, expected_pri_s=85e-6)


def test_compatible_window_produces_strong_decomposed_evidence():
    pulses = [pdw(i, i * 85e-6) for i in range(20)]
    evidence = tracker().update(window_id="W0", time_s=0., window_end_s=.1,
        pdws=pulses, trains=[train()], candidate_groups=1,
        rf_transition_score=1.)
    assert evidence.timing_score > .9
    assert evidence.rf_score > .9
    assert evidence.signal_score > .9
    assert evidence.track_score > .9


def test_sparse_window_retains_score_and_empty_window_decays_it():
    state = tracker()
    first = state.update(window_id="W0", time_s=0., window_end_s=.1,
        pdws=[pdw(1, .05)], trains=[], candidate_groups=1)
    sparse = state.update(window_id="W1", time_s=.1, window_end_s=.2,
        pdws=[pdw(2, .15)], trains=[], candidate_groups=1)
    empty = state.update(window_id="W2", time_s=.2, window_end_s=.8,
        pdws=[], trains=[], candidate_groups=0)
    assert sparse.timing_score is None
    assert sparse.track_score > .5 * first.track_score
    assert empty.track_score < sparse.track_score


def test_rejects_out_of_order_windows():
    state = tracker()
    state.update(window_id="W0", time_s=1., window_end_s=2., pdws=[],
                 trains=[], candidate_groups=0)
    try:
        state.update(window_id="W1", time_s=.5, window_end_s=.7, pdws=[],
                     trains=[], candidate_groups=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected chronological validation")
