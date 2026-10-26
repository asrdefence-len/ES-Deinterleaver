"""Check per-PDW SNR-weighted timing updates and legacy compatibility."""
import random
import pytest
from testbed.bayesian_timing_state import BayesianTimingEstimator
from two_emitter_snr_baseline import toa_sigma


def test_first_pulse_covariance_matches_snr():
    for snr in (0., 10., 30.):
        e = BayesianTimingEstimator(.001, .00001)
        s = e.process(0., snr_db=snr)
        assert s["epoch_sigma_s"] == pytest.approx(toa_sigma(snr))
        assert s["last_measurement_sigma_s"] == pytest.approx(toa_sigma(snr))


def test_low_snr_pulse_receives_smaller_kalman_update():
    # Identical priors and innovations, different measurement quality.
    high = BayesianTimingEstimator(.001, 1e-6, detection_probability=1)
    low = BayesianTimingEstimator(.001, 1e-6, detection_probability=1)
    high.process(0., snr_db=20)
    low.process(0., snr_db=20)
    high.process(.001 + 300e-9, snr_db=30)
    low.process(.001 + 300e-9, snr_db=0)
    assert abs(high.snapshot()["pri_s"]-.001) > abs(low.snapshot()["pri_s"]-.001)


def test_varying_snr_keeps_constant_pri():
    rng = random.Random(73)
    e = BayesianTimingEstimator(.001, 10e-6, detection_probability=1)
    snrs = [30.]*25 + [0.]*25 + [30.]*25
    for n, snr in enumerate(snrs):
        e.process(n*.001+rng.gauss(0,toa_sigma(snr)),snr_db=snr)
    assert abs(e.snapshot()["pri_s"]-.001) < 50e-9
    assert e.snapshot()["pulse_indices"] == tuple(range(len(snrs)))


def test_bad_snr_is_rejected():
    e = BayesianTimingEstimator(.001, .00001)
    with pytest.raises(ValueError):
        e.process(0., snr_db=float("nan"))
