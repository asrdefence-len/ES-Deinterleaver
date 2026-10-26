"""Controlled checks of latent-index Gaussian timing regression."""
import random
import pytest
from testbed.bayesian_timing_state import BayesianTimingEstimator


def test_exact_pri_with_complete_detection():
    rng = random.Random(7)
    e = BayesianTimingEstimator(.00101, .00005, detection_probability=1)
    for i in range(40):
        e.process(i*.001+rng.gauss(0, 50e-9))
    s = e.snapshot()
    assert abs(s["pri_s"]-.001) < 30e-9
    assert s["missed_pulses"] == 0
    assert s["pulse_indices"] == tuple(range(40))


@pytest.mark.parametrize("pd", [1.0, .95, .70, .50])
def test_single_emitter_missed_detections(pd):
    rng = random.Random(23)
    e = BayesianTimingEstimator(.001, .00001, detection_probability=pd)
    actual_indices = []
    for i in range(100):
        if i == 0 or rng.random() < pd:
            actual_indices.append(i)
            e.process(i*.001+rng.gauss(0, 50e-9))
    s = e.snapshot()
    assert abs(s["pri_s"]-.001) < 50e-9
    assert s["pulse_indices"] == tuple(actual_indices)
    # Only gaps between observed pulses are identifiable. Trailing\n    # undetected transmissions after the final detection are not observed.\n    assert s["missed_pulses"] == actual_indices[-1]+1-len(actual_indices)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        BayesianTimingEstimator(.001, 0)
    e = BayesianTimingEstimator(.001, .00001)
    e.process(0.0)
    with pytest.raises(ValueError):
        e.process(0.0)
