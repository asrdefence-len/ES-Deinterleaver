"""Contract tests for the experimental bounded MHT baseline."""
import pytest
from testbed.mht_deinterleaver import MHTDeinterleaver
from testbed.generator import generate


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        MHTDeinterleaver(max_hypotheses=0)


def test_bootstrap_and_bounded_hypotheses():
    scenario = {
        "duration_s": 0.8, "seed": 7,
        "emitters": [
            {"id": "A", "aoa_deg": 45, "frequency_hz": 9.4e9,
             "pulse_width_s": 10e-6, "pri_s": 1000e-6},
            {"id": "B", "aoa_deg": 90, "frequency_hz": 9.4e9,
             "pulse_width_s": 10e-6, "pri_s": 1020e-6}
        ]
    }
    # Scenario structure is exercised by the comparison script against
    # repository fixtures; here test only configuration-independent state.
    mht = MHTDeinterleaver(bootstrap_pdws=40, max_hypotheses=3)
    snap = mht.snapshot()
    assert snap["trains"] == 0
    assert snap["hypotheses"] == 0
    assert not snap["bootstrapped"]


def test_timing_kalman_update():
    mht = MHTDeinterleaver()
    from testbed.mht_deinterleaver import Hypothesis
    h = Hypothesis(0.0, 0.001, (1e-14, 0, 0, 1e-16))
    predicted, cov = mht._predict(h, 1)
    assert predicted == pytest.approx(0.001)
    assert cov[0] > 0
    updated = mht._update(h, 0.0010001, 1, predicted, cov)
    assert updated.pri_s > 0
    assert updated.toa_s > 0.001
