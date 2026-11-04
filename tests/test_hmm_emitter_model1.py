import math

import numpy as np
import pytest

from sequential_pdw_tracker.hmm_emitter import HMMEmitterModel1
from sequential_pdw_tracker.model import PDW


def pdw(index, toa_us, frequency_mhz, aoa_deg, *, pulse_width_us=1.0, amplitude_db=-70.0):
    return PDW(
        pdw_id=index,
        toa_s=toa_us * 1e-6,
        frequency_hz=frequency_mhz * 1e6,
        pulse_width_s=pulse_width_us * 1e-6,
        aoa_deg=aoa_deg,
        amplitude_db=amplitude_db,
    )


def test_forward_probabilities_and_transition_rows_are_normalized():
    tracker = HMMEmitterModel1()
    updates = [tracker.process(pdw(
        index,
        index * 10,
        (1000, 1200, 1400)[index % 3],
        32.0,
    )) for index in range(90)]
    assert tracker.update_count == 90
    assert all(sum(update.rf_posterior) == pytest.approx(1.0) for update in updates)
    transition = np.asarray(tracker.snapshot()["rf_transition_matrix"])
    assert transition.shape == (3, 3)
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.isfinite([update.total_log_evidence for update in updates]).all()


def test_frequency_hmm_uses_soft_forward_state_and_learns_transitions():
    tracker = HMMEmitterModel1()
    updates = [tracker.process(pdw(
        index,
        index * 10,
        (1000, 1200, 1400)[index % 3],
        20.0,
    )) for index in range(120)]
    state = tracker.snapshot()
    means_mhz = sorted(item["mean_hz"] / 1e6 for item in state["rf_states"])
    assert means_mhz == pytest.approx([1000, 1200, 1400], abs=2.0)
    # The complete posterior is retained rather than a single hard state label.
    assert all(len(update.rf_posterior) == len(state["rf_states"])
               for update in updates[-10:])
    assert all(math.isfinite(value) for value in updates[-1].rf_log_posterior)


def test_model1_provably_ignores_pulse_width_and_amplitude():
    left = HMMEmitterModel1()
    right = HMMEmitterModel1()
    for index in range(30):
        common = dict(index=index, toa_us=index * 12,
                      frequency_mhz=(1000, 1200)[index % 2], aoa_deg=45.0)
        left_update = left.process(pdw(**common, pulse_width_us=0.1, amplitude_db=-20.0))
        right_update = right.process(pdw(**common, pulse_width_us=500.0, amplitude_db=-150.0))
        assert left_update == right_update
    assert left.snapshot() == right.snapshot()
    assert left.snapshot()["enabled_features"] == ["Frequency", "ToA", "AoA"]
    assert left.snapshot()["excluded_features"] == ["PulseWidth", "Amplitude"]


def test_semi_markov_timing_survives_large_latent_interval_count():
    tracker = HMMEmitterModel1()
    times_us = [0, 10, 20, 30, 40, 50, 10_050, 10_060]
    updates = [tracker.process(pdw(index, toa, 1000, 15.0))
               for index, toa in enumerate(times_us)]
    after_gap = updates[-2]
    assert after_gap.timing_map_intervals is not None
    assert after_gap.timing_map_intervals > 20
    assert math.isfinite(after_gap.timing_log_evidence)
    assert math.isfinite(updates[-1].timing_log_evidence)
    timing = tracker.snapshot()["timing_hypotheses"]
    assert timing
    assert sum(item["posterior"] for item in timing) == pytest.approx(1.0)


def test_circular_aoa_markov_filter_crosses_wrap_boundary():
    tracker = HMMEmitterModel1()
    for index, angle in enumerate((179.0, -179.0, 178.5, -178.5)):
        tracker.process(pdw(index, index * 10, 1000, angle))
    mean = tracker.snapshot()["aoa"]["mean_deg"]
    assert abs(abs(mean) - 180.0) < 2.0
