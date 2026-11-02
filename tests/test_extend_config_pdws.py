import json

import h5py
import numpy as np

from extend_config_pdws import extend_recording


def make_recording(path):
    # Raw Config 101 units: ToA/PW in microseconds and RF in MHz.
    data = np.array([
        [0., 9400., .30, 10.],
        [250_000., 9600., .40, 20.],
        [500_000., 9401., .30, 10.1],
        [750_000., 9601., .40, 20.1],
        [1_000_000., 9402., .30, 10.2],
    ])
    labels = np.array([1, 2, 1, 2, 1])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset("labels", data=labels)
        metadata = handle.create_group("metadata")
        metadata.create_dataset("feature_names",
            data=np.array([b"ToA", b"Frequency", b"PulseWidth", b"AoA"]))
        handle.attrs["original"] = "retained"
    return data, labels


def test_extension_is_continuous_truth_labelled_and_has_requested_gap(tmp_path):
    source, output = tmp_path / "config_101.h5", tmp_path / "config_101_6s.h5"
    original_data, original_labels = make_recording(source)
    manifest = extend_recording(source, output, duration_s=6.,
        dropouts=[(1, 2., 4.)])

    with h5py.File(output) as handle:
        data = handle["data"][:]
        labels = handle["labels"][:]
        toa_s = data[:, 0] * 1e-6
        assert np.all(np.diff(toa_s) >= 0)
        assert toa_s[0] == 0.
        assert toa_s[-1] < 6.
        assert toa_s[-1] > 5.5
        assert not np.any((labels == 1) & (toa_s >= 2.) & (toa_s < 4.))
        assert np.any((labels == 2) & (toa_s >= 2.) & (toa_s < 4.))
        assert np.any((labels == 1) & (toa_s < 2.))
        assert np.any((labels == 1) & (toa_s >= 4.))
        assert handle.attrs["original"] == "retained"
        saved = json.loads(handle.attrs["s2b_extension_manifest_json"])
        assert saved["dropouts"][0]["truth_emitter"] == 1
        assert saved["template_repetition"] is True

    with h5py.File(source) as handle:
        assert np.array_equal(handle["data"][:], original_data)
        assert np.array_equal(handle["labels"][:], original_labels)
    assert manifest["output_pdws"] == len(data)
    assert manifest["cycles"] >= 5


def test_automatic_dropouts_choose_high_support_emitters(tmp_path):
    source, output = tmp_path / "source.h5", tmp_path / "output.h5"
    make_recording(source)
    manifest = extend_recording(source, output, duration_s=4., automatic_dropouts=1)
    assert len(manifest["dropouts"]) == 1
    assert manifest["dropouts"][0]["truth_emitter"] == 1


def test_nonzero_source_toa_is_normalised_to_zero(tmp_path):
    source, output = tmp_path / "source.h5", tmp_path / "output.h5"
    make_recording(source)
    with h5py.File(source, "r+") as handle:
        handle["data"][:, 0] += 15_000_000.
    manifest = extend_recording(source, output, duration_s=3., dropouts=[])
    with h5py.File(output) as handle:
        assert handle["data"][0, 0] == 0.
        assert handle["data"][-1, 0] < 3_000_000.
    assert manifest["source_toa_origin_raw"] == 15_000_000.
