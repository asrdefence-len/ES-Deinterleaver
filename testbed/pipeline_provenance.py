"""Reproducible component, source, settings and data identities for a run."""
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_manifest(recording, *, grouping, measurement_mode, settings, tracker):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=root,
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    sources = sorted([*root.glob('*.py'), *root.joinpath('testbed').glob('*.py')])
    fingerprints = {str(p.relative_to(root)): sha256_file(p) for p in sources}
    source_digest = hashlib.sha256(''.join(f'{p}:{h}\n' for p, h in fingerprints.items()).encode()).hexdigest()
    return {
        'schema_version': 1,
        'pipeline': {
            'aoa': 'compare_aoa_batch.batch_aoa_populations',
            'grouping': ('compare_aoa_batch.rf_pw_groups:sklearn.cluster.HDBSCAN'
                         if grouping == 'hdbscan' else 'compare_multiscale_rf.group_window'),
            'pri_estimator': {'histogram': 'testbed.pri_histogram.estimate_gap_pri:local_gap_humps_v1',
                              'train': 'deinterleaver_v0.discover:fixed_train',
                              'blob': None}[measurement_mode],
            'tracker': 'testbed.blob_tracker.BlobEmitterTracker',
            'filter': 'diagonal random-walk Kalman with joint chi-square NIS gate',
            'track_management': 'Bernoulli existence, birth, coast, consolidation, bounded archive',
            'supported_integration': grouping == 'hdbscan' and measurement_mode == 'histogram'},
        'grouping_settings': {
            'aoa_bin_deg': .25, 'aoa_smooth_deg': .5,
            'aoa_min_peak_count': 8, 'aoa_min_separation_deg': 1.5,
            'hdbscan_min_cluster_size': 8, 'hdbscan_min_samples': 8,
            'features': 'AOA density populations, then existing scaled RF/PW features'},
        'pri_histogram_settings': ({'bin_width_us': settings.get('pri_bin_us', 2.),
            'min_gap_us': settings.get('pri_min_gap_us', 10.),
            'max_gap_us': settings.get('pri_max_gap_us', 400.),
            'neighbourhood': settings.get('pri_neighbourhood', .15),
            'measurement_sigma': 'max(selected-gap SD, bin_width/sqrt(12), 0.05 us); tracker also applies its PRI floor',
            'harmonic_scaling': False} if measurement_mode == 'histogram' else None),
        'git_revision': git('rev-parse', 'HEAD'),
        'git_branch': git('branch', '--show-current'),
        'git_dirty': bool(git('status', '--porcelain')),
        'source_snapshot_sha256': source_digest,
        'source_files_sha256': fingerprints,
        'recording': {'name': Path(recording).name, 'sha256': sha256_file(recording),
                      'size_bytes': Path(recording).stat().st_size},
        'settings': settings,
        'tracker_parameters': {k: v for k, v in vars(tracker).items()
            if not k.startswith('_') and isinstance(v, (int, float, str, bool))},
        'package_versions': {p: importlib.metadata.version(p) for p in
                             ('numpy', 'scipy', 'scikit-learn', 'h5py')},
        'truth_is_evaluation_only': True,
        'calibration_status': 'Integration baseline; Q/R, gap-mean uncertainty and lifecycle priors remain nominal'}
