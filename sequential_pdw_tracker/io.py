"""Streaming reader for the Turing Config 101 HDF5 layout."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .model import PDW


REQUIRED_FIELDS = ("ToA", "Frequency", "PulseWidth", "AoA", "Amplitude")


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    total_pdws: int
    feature_names: tuple[str, ...]
    labels_present: bool


def _feature_names(handle: h5py.File) -> tuple[str, ...]:
    try:
        raw = handle["metadata/feature_names"][:]
    except KeyError as error:
        raise ValueError("Missing HDF5 dataset 'metadata/feature_names'") from error
    names = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in raw)
    missing = sorted(set(REQUIRED_FIELDS) - set(names))
    if missing:
        raise ValueError(f"Missing required PDW fields: {', '.join(missing)}")
    return names


def inspect_dataset(path: str | Path) -> DatasetInfo:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "data" not in handle or handle["data"].ndim != 2:
            raise ValueError("Expected two-dimensional HDF5 dataset 'data'")
        names = _feature_names(handle)
        data = handle["data"]
        if data.shape[1] != len(names):
            raise ValueError("Feature-name count does not match PDW column count")
        labels = handle.get("labels")
        if labels is not None and (labels.ndim not in (1, 2) or labels.shape[0] != len(data)):
            raise ValueError("Labels must contain one entry for every PDW")
        return DatasetInfo(path, len(data), names, labels is not None)


def iter_pdws(
    path: str | Path,
    *,
    limit: int | None = None,
    start: int = 0,
    chunk_size: int = 4096,
) -> Iterator[tuple[PDW, int | str | None]]:
    """Yield time-ordered PDWs and separately carried evaluation labels.

    The tracker receives only the ``PDW`` value. The second tuple item is for
    evaluation after the association decision has been made.
    """
    if start < 0 or chunk_size < 1 or (limit is not None and limit < 1):
        raise ValueError("start must be non-negative; limit and chunk_size must be positive")
    with h5py.File(path, "r") as handle:
        names = _feature_names(handle)
        data = handle["data"]
        labels = handle.get("labels")
        columns = {name: names.index(name) for name in REQUIRED_FIELDS}
        stop = len(data) if limit is None else min(len(data), start + limit)
        previous_toa = -float("inf")
        for chunk_start in range(start, stop, chunk_size):
            chunk_stop = min(stop, chunk_start + chunk_size)
            values = np.asarray(data[chunk_start:chunk_stop], dtype=float)
            truth = None if labels is None else np.asarray(labels[chunk_start:chunk_stop]).reshape(-1)
            for offset, row in enumerate(values):
                toa_s = float(row[columns["ToA"]] * 1e-6)
                if not np.isfinite(row).all():
                    raise ValueError(f"PDW row {chunk_start + offset} contains a non-finite value")
                if toa_s < previous_toa:
                    raise ValueError("PDWs are not in non-decreasing ToA order")
                previous_toa = toa_s
                pdw = PDW(
                    pdw_id=chunk_start + offset,
                    toa_s=toa_s,
                    frequency_hz=float(row[columns["Frequency"]] * 1e6),
                    pulse_width_s=float(row[columns["PulseWidth"]] * 1e-6),
                    aoa_deg=float(row[columns["AoA"]]),
                    amplitude_db=float(row[columns["Amplitude"]]),
                )
                label = None if truth is None else truth[offset].item()
                yield pdw, label
