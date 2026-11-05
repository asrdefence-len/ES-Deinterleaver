import math

import pytest

from sequential_pdw_tracker.detected_train_timing import (
    DetectedTrainTimingEstimator,
)


def test_shadow_histogram_predicts_received_gaps_without_asserting_pri():
    estimator = DetectedTrainTimingEstimator()
    toa = 0.0
    first = estimator.process(toa)
    assert first.gap_s is None
    updates = []
    for gap_us in [10, 12, 10, 1000] * 20:
        toa += gap_us * 1e-6
        updates.append(estimator.process(toa))
    assert all(math.isfinite(update.log_predictive_density) for update in updates)
    assert all(update.feedback_weight == 0.0 for update in updates)
    snapshot = estimator.snapshot()
    assert snapshot["transmitter_pri_estimate"] is None
    assert sum(snapshot["histogram_counts"]) == pytest.approx(80)


def test_visibility_shadow_separates_short_and_long_detected_gaps():
    estimator = DetectedTrainTimingEstimator()
    toa = 0.0
    estimator.process(toa)
    short_updates = []
    long_updates = []
    for _ in range(30):
        for gap_us in (10, 11, 12, 1500):
            toa += gap_us * 1e-6
            update = estimator.process(toa)
            (long_updates if gap_us > 100 else short_updates).append(update)
    assert short_updates[-1].active_posterior > 0.9
    assert long_updates[-1].coast_posterior > 0.9
