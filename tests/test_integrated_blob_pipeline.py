"""End-to-end contracts for the supported processing chain, not just its parts."""
from dataclasses import asdict, replace
import json
import sys

import h5py
import numpy as np
import pytest

import run_blob_tracker
from testbed.blob_observation import summarize_blob_pri
from testbed.blob_tracker import BlobEmitterTracker
from testbed.pri_histogram import estimate_gap_pri


NAMES = ['ToA', 'Frequency', 'PulseWidth', 'AoA']


def emitter_window(offset=0., pri_offset=0.):
    # Two stable populations, each with two neighbouring gap humps. All units
    # are raw µs/MHz/µs/degrees until the grouping-stage adapter exports SI.
    rng = np.random.default_rng(47)
    rows, truth = [], []
    for identity, bearing, rf, pw in [(1, 20., 9400., .3), (2, 70., 10400., .6)]:
        times = offset + np.cumsum(np.resize([100.25 + pri_offset, 112.25 + pri_offset], 240))
        rows.extend(zip(times, rng.normal(rf, .01, 240), rng.normal(pw, .001, 240), rng.normal(bearing, .03, 240)))
        truth.extend([identity] * 240)
    order = np.argsort(np.asarray(rows)[:, 0], kind='stable')
    return np.asarray(rows)[order], np.asarray(truth)[order]


def write_recording(path, *, relabel=False):
    first, truth = emitter_window()
    second, second_truth = emitter_window(40000., .4)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('data', data=np.concatenate([first, second]))
        labels = np.concatenate([truth, second_truth])
        handle.create_dataset('labels', data=99 - labels if relabel else labels)
        metadata = handle.create_group('metadata')
        metadata.create_dataset('feature_names', data=np.asarray(NAMES, dtype='S'))


def test_two_humps_export_mean_dispersion_and_bounded_metadata():
    toa = np.r_[0., np.cumsum(np.tile([100.25, 112.25], 100))] * 1e-6
    histogram = estimate_gap_pri(toa)
    assert histogram.mean_s == pytest.approx(106.25e-6)
    assert sorted(histogram.peaks_s) == pytest.approx([101e-6, 113e-6])
    assert histogram.used_gaps == 200
    assert histogram.spread_s == pytest.approx(np.std(np.tile([100.25, 112.25], 100), ddof=1) * 1e-6)
    # Export spread, rather than claiming SEM from independently sampled pulses.
    assert histogram.measurement_sigma_s >= histogram.spread_s
    assert estimate_gap_pri([0., 100e-6, 200e-6]) is None


def test_histogram_adapter_preserves_whole_blob_and_filter_uses_direct_mean():
    data, _ = emitter_window()
    labels = np.zeros(len(data), dtype=int)
    labels[data[:, 3] > 50] = -1
    original = labels.copy()
    observation = summarize_blob_pri(data, NAMES, labels, 'W0')[0]
    assert np.array_equal(labels, original)
    assert observation.pulse_count == 240
    assert observation.train_id is None and observation.phase_epoch_s is None
    assert observation.pri_histogram is not None
    assert not {'pdw_ids', 'toa_sequence', 'truth'} & asdict(observation).keys()
    tracker = BlobEmitterTracker(consistency_sink=lambda event: None)
    tracker.process(observation)
    prior = tracker.hypotheses[0].tracks[0]
    histogram = replace(observation.pri_histogram, mean_s=observation.pri_s + 1e-6)
    second = replace(observation, observation_id='W1:B0', window_id='W1',
                     start_s=.04, end_s=.06, pri_s=histogram.mean_s,
                     pri_sigma_s=histogram.measurement_sigma_s, pri_histogram=histogram)
    _, _, stats = tracker._compatibility(second, prior)
    assert stats['dof'] == 3
    assert stats['timing']['observed_harmonic'] == stats['timing']['track_harmonic'] == 1
    p = prior.pri_variance_s2 + tracker.pri_q * max(0., second.start_s - prior.last_end_s)
    r = max(second.pri_sigma_s ** 2, tracker.pri_measurement_floor_s ** 2)
    tracker.process(second)
    updated = next(t for t in tracker.hypotheses[0].tracks if t.track_id == prior.track_id)
    assert updated.pri_s == pytest.approx(prior.pri_s + p/(p+r)*(second.pri_s-prior.pri_s), rel=1e-12, abs=0)
    assert updated.pri_variance_s2 == pytest.approx((1-p/(p+r))*p, rel=1e-12, abs=0)
    assert tracker.track_monitor(updated)['last_pri_histogram']['peaks_s'] == histogram.peaks_s
    # An observed-gap mean twice as large must not be rescued as a harmonic.
    far_histogram = replace(histogram, mean_s=histogram.mean_s * 2)
    far = replace(second, pri_s=far_histogram.mean_s, pri_histogram=far_histogram)
    assert tracker._compatibility(far, updated) is None


def test_default_runner_connects_hdbscan_histogram_kalman_and_is_truth_independent(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy multiscale/fixed-train path entered')
    monkeypatch.setattr(run_blob_tracker, 'group_window', forbidden)
    monkeypatch.setattr(run_blob_tracker.FixedPRIPulseAssociation, 'process', forbidden)
    snapshots = []
    for index in range(2):
        recording, output = tmp_path / f'{index}.h5', tmp_path / f'run{index}'
        write_recording(recording, relabel=bool(index))
        monkeypatch.setattr(sys, 'argv', ['run_blob_tracker.py', str(recording),
            '--windows', '2', '--window-size', '480', '--beam-width', '4',
            '--consistency-log', '--output', str(output)])
        run_blob_tracker.main()
        manifest = json.loads((output / 'run_manifest.json').read_text())
        assert manifest['pipeline']['supported_integration'] is True
        assert manifest['recording']['sha256']
        assert 'HDBSCAN' in manifest['pipeline']['grouping']
        monitor = [json.loads(line) for line in (output / 'blob_monitor.jsonl').read_text().splitlines()]
        assert any(row['track']['last_innovation']['degrees_of_freedom'] == 3 for row in monitor)
        assert any(row['measurement']['pri_histogram'] is not None for row in monitor)
        for row in monitor:
            histogram = row['measurement']['pri_histogram']
            if histogram is None:
                assert row['measurement']['pri_s'] is None
            else:
                assert row['measurement']['pri_s'] == histogram['mean_s']
                assert row['track']['pri_model'] == 'local_gap_humps_v1'
                assert row['track']['last_pri_histogram'] == histogram
        snapshot = json.loads((output / 'tracker_snapshot.json').read_text())
        assert all('lifecycle_status' in t['monitor'] for h in snapshot['hypotheses'] for t in h['active_tracks'])
        snapshots.append(snapshot)
    assert snapshots[0] == snapshots[1]


def test_basic_lifecycle_predicts_coasts_archives_and_never_reuses_identity():
    data, _ = emitter_window()
    labels = np.where(data[:, 3] < 50, 0, -1)
    observation = summarize_blob_pri(data, NAMES, labels, 'W0')[0]
    tracker = BlobEmitterTracker(max_gap_s=10., detection_probability=.9, archive_after_s=.5)
    tracker.process(observation)
    first = tracker.hypotheses[0].tracks[0]
    monitor = tracker.track_monitor(first)
    assert monitor['lifecycle_status'] == 'TENTATIVE'
    predicted = tracker.track_monitor(first, first.last_end_s + .2)
    assert predicted['covariance']['pri_variance_s2'] > monitor['covariance']['pri_variance_s2']
    tracker.complete_window('W0', .03)
    tracker.complete_window('W1', .25)
    coasted = tracker.hypotheses[0].tracks[0]
    assert tracker.track_monitor(coasted)['existence_probability'] < monitor['existence_probability']
    tracker.complete_window('W2', .8)
    assert first.track_id in tracker.archived_tracks
    assert tracker.archived_tracks[first.track_id]['reason'] == 'LOST_CONFIDENCE'
    tracker.process(replace(observation, observation_id='W3:B0', window_id='W3', start_s=1., end_s=1.03))
    assert tracker.hypotheses[0].tracks[0].track_id != first.track_id


def test_missing_histogram_measurement_still_predicts_existing_pri_covariance():
    data, _ = emitter_window()
    observation = summarize_blob_pri(data, NAMES, np.where(data[:, 3] < 50, 0, -1), 'W0')[0]
    tracker = BlobEmitterTracker()
    tracker.process(observation)
    prior = tracker.hypotheses[0].tracks[0]
    missing = replace(observation, observation_id='W1:B0', window_id='W1',
        start_s=.05, end_s=.08, pri_s=None, pri_sigma_s=None, pri_histogram=None)
    tracker.process(missing)
    updated = next(t for t in tracker.hypotheses[0].tracks if t.track_id == prior.track_id)
    assert updated.pri_s == prior.pri_s
    assert updated.pri_variance_s2 == pytest.approx(
        prior.pri_variance_s2 + tracker.pri_q * (.05 - prior.last_end_s), rel=1e-12, abs=0)
    assert updated.pri_histogram == prior.pri_histogram
