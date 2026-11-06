from dataclasses import asdict
import math

import numpy as np
import pytest

from sequential_pdw_tracker.hmm_emitter import HMMEmitterModel1
from sequential_pdw_tracker.hmm_emitter_model2 import HMMEmitterModel2
from sequential_pdw_tracker.model import PDW


def pdw(index, toa_us, frequency_mhz, pulse_width_us, aoa_deg=32.0,
        amplitude_db=-70.0):
    return PDW(
        pdw_id=index,
        toa_s=toa_us * 1e-6,
        frequency_hz=frequency_mhz * 1e6,
        pulse_width_s=pulse_width_us * 1e-6,
        aoa_deg=aoa_deg,
        amplitude_db=amplitude_db,
    )


def model1_projection(model2_update):
    values = asdict(model2_update)
    model1_fields = {
        key: value
        for key, value in values.items()
        if key not in {
            "pulse_width_s",
            "pulse_width_posterior",
            "pulse_width_log_posterior",
            "pulse_width_map_state",
            "pulse_width_map_mean_s",
            "pulse_width_log_evidence",
            "pulse_width_state_created",
            "model1_log_evidence",
        }
    }
    model1_fields["total_log_evidence"] = model2_update.model1_log_evidence
    return model1_fields


def test_model2_preserves_every_model1_update_outside_total_evidence():
    model1 = HMMEmitterModel1()
    model2 = HMMEmitterModel2()
    for index in range(120):
        item = pdw(
            index,
            index * 11,
            (1000, 1200, 1400)[index % 3],
            (0.10, 0.27, 0.43)[index % 3],
            aoa_deg=30.0 + 0.1 * math.sin(index),
        )
        expected = model1.process(item)
        actual = model2.process(item)
        assert model1_projection(actual) == asdict(expected)

    left = model1.snapshot()
    right = model2.snapshot()
    for key in (
        "rf_states",
        "rf_transition_matrix",
        "aoa",
        "detected_train_timing_shadow",
    ):
        assert right[key] == left[key]


def test_pulse_width_hmm_learns_modes_and_soft_transition_topology():
    tracker = HMMEmitterModel2()
    widths = (0.10, 0.27, 0.43)
    updates = [
        tracker.process(pdw(index, index * 10, 1000, widths[index % 3]))
        for index in range(300)
    ]
    snapshot = tracker.snapshot()
    means_us = sorted(
        state["mean_s"] * 1e6 for state in snapshot["pulse_width_states"]
    )
    assert means_us == pytest.approx(widths, abs=0.005)
    assert all(
        sum(update.pulse_width_posterior) == pytest.approx(1.0)
        for update in updates
    )
    transition = np.asarray(snapshot["pulse_width_transition_matrix"])
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert transition[0, 1] > 0.9
    assert transition[1, 2] > 0.9
    assert transition[2, 0] > 0.9
    assert np.isfinite(updates[-1].pulse_width_log_posterior).all()
    assert sum(update.pulse_width_state_created for update in updates) == 3


def test_model2_total_evidence_adds_pw_but_timing_remains_zero_feedback():
    tracker = HMMEmitterModel2()
    updates = [
        tracker.process(pdw(index, index * 12, 1000, (0.10, 0.27)[index % 2]))
        for index in range(80)
    ]
    for update in updates:
        assert update.total_log_evidence == pytest.approx(
            update.model1_log_evidence + update.pulse_width_log_evidence
        )
        assert update.timing_feedback_weight == 0.0
    snapshot = tracker.snapshot()
    assert snapshot["active_evidence_features"] == [
        "Frequency", "AoA", "PulseWidth",
    ]
    assert snapshot["shadow_features"] == ["ToA"]
    assert snapshot["excluded_features"] == ["Amplitude"]


def test_model2_ignores_amplitude():
    left = HMMEmitterModel2()
    right = HMMEmitterModel2()
    for index in range(40):
        common = dict(
            index=index,
            toa_us=index * 10,
            frequency_mhz=(1000, 1200)[index % 2],
            pulse_width_us=(0.10, 0.27, 0.43)[index % 3],
        )
        left_update = left.process(pdw(**common, amplitude_db=-20.0))
        right_update = right.process(pdw(**common, amplitude_db=-150.0))
        assert left_update == right_update
    assert left.snapshot() == right.snapshot()


def test_predictive_evidence_scores_without_mutating_trained_model():
    tracker = HMMEmitterModel2()
    for index in range(120):
        tracker.process(pdw(
            index,
            index * 10,
            (1000, 1200, 1400)[index % 3],
            (0.10, 0.27, 0.43)[index % 3],
            aoa_deg=32.0,
        ))

    before = tracker.snapshot()
    expected = tracker.predictive_evidence(
        pdw(200, 1200, 1000, 0.10, aoa_deg=32.0)
    )
    wrong = tracker.predictive_evidence(
        pdw(201, 1200, 1040, 0.20, aoa_deg=40.0)
    )

    assert tracker.snapshot() == before
    assert expected.total_log_evidence > wrong.total_log_evidence
    assert expected.total_log_evidence == pytest.approx(
        expected.rf_log_evidence
        + expected.aoa_log_evidence
        + expected.pulse_width_log_evidence
    )
