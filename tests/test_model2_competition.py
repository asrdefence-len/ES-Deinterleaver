import copy

import pytest

from sequential_pdw_tracker.hmm_emitter_model2 import HMMEmitterModel2
from sequential_pdw_tracker.model import PDW
from sequential_pdw_tracker.model2_competition import Model2CompetitiveAssociation


def pdw(index: int, toa_us: float, frequency_mhz: float, aoa_deg: float,
        pulse_width_us: float) -> PDW:
    return PDW(
        pdw_id=index,
        toa_s=toa_us * 1e-6,
        frequency_hz=frequency_mhz * 1e6,
        pulse_width_s=pulse_width_us * 1e-6,
        aoa_deg=aoa_deg,
        amplitude_db=-70.0,
    )


def trained_model(frequency_mhz: float, aoa_deg: float,
                  pulse_width_us: float) -> HMMEmitterModel2:
    model = HMMEmitterModel2()
    for index in range(100):
        model.process(pdw(
            index,
            index * 10,
            frequency_mhz,
            aoa_deg,
            pulse_width_us,
        ))
    return model


def test_competition_updates_only_the_winning_filter_and_normalizes_weights():
    left = trained_model(1000.0, 20.0, 0.10)
    right = trained_model(1200.0, 40.0, 0.30)
    left_score = left.predictive_evidence(
        pdw(200, 1000, 1000.0, 20.0, 0.10)
    ).total_log_evidence
    right_score = right.predictive_evidence(
        pdw(201, 1000, 1200.0, 40.0, 0.30)
    ).total_log_evidence
    association = Model2CompetitiveAssociation(
        {"left": left, "right": right},
        thresholds={"left": left_score - 1.0, "right": right_score - 1.0},
        last_accepted_toa_s={"left": 990e-6, "right": 990e-6},
    )
    left_before = copy.deepcopy(left.snapshot())
    right_before = copy.deepcopy(right.snapshot())

    decision = association.process(pdw(300, 1000, 1000.0, 20.0, 0.10))

    assert decision.selected_track == "left"
    assert left.snapshot() != left_before
    assert right.snapshot() == right_before
    assert decision.neither_weight + sum(
        score.normalized_hypothesis_weight
        for score in decision.track_scores.values()
    ) == pytest.approx(1.0)


def test_competition_selects_neither_when_every_track_is_below_threshold():
    left = trained_model(1000.0, 20.0, 0.10)
    right = trained_model(1200.0, 40.0, 0.30)
    association = Model2CompetitiveAssociation(
        {"left": left, "right": right},
        thresholds={"left": -20.0, "right": -20.0},
        last_accepted_toa_s={"left": 990e-6, "right": 990e-6},
    )
    snapshots = {"left": left.snapshot(), "right": right.snapshot()}

    decision = association.process(pdw(300, 1000, 1600.0, 100.0, 0.60))

    assert decision.selected_track is None
    assert left.snapshot() == snapshots["left"]
    assert right.snapshot() == snapshots["right"]
    assert association.neither_count == 1
