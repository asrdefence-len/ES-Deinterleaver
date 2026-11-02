"""Extend a truth-labelled PDW recording to a longer continuous timeline.

The source recording is used as a deterministic template.  Each template
cycle retains the original emitter labels and measured features; only ToA is
shifted.  Optional truth-emitter dropout intervals remove selected pulses so
that a tracker must expire and reacquire those emitters.

This is intentionally a playback/stress-test generator, not a claim that the
source scenario's hidden kinematics or random generator state were recovered.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def feature_names(handle: h5py.File) -> list[str]:
    if "metadata/feature_names" not in handle:
        raise ValueError("Missing metadata/feature_names")
    return [value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["metadata/feature_names"][:]]


def parse_dropout(spec: str) -> tuple[int, float, float]:
    """Parse LABEL:START:END where START and END are seconds."""
    try:
        label, start, end = spec.split(":")
        result = int(label), float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "dropout must be LABEL:START_S:END_S") from exc
    if result[0] < 0 or not 0 <= result[1] < result[2]:
        raise argparse.ArgumentTypeError(
            "dropout requires LABEL >= 0 and 0 <= START_S < END_S")
    return result


def choose_default_dropouts(labels: np.ndarray, duration_s: float,
                            count: int = 4) -> list[tuple[int, float, float]]:
    """Choose high-support emitters and stagger visible silence intervals."""
    values, counts = np.unique(labels.reshape(-1), return_counts=True)
    valid = [(int(label), int(n)) for label, n in zip(values, counts)
             if int(label) >= 0]
    selected = [label for label, _ in sorted(valid, key=lambda x: (-x[1], x[0]))[:count]]
    starts = np.linspace(.20, .74, len(selected)) if selected else []
    widths = (.055, .070, .060, .080)
    return [(label, duration_s * start, min(duration_s, duration_s * (start + width)))
            for index, (label, start) in enumerate(zip(selected, starts))
            for width in (widths[index % len(widths)],)]


def _copy_non_stream_data(source: h5py.File, target: h5py.File) -> None:
    for key in source:
        if key not in {"data", "labels"}:
            source.copy(key, target)
    for key, value in source.attrs.items():
        target.attrs[key] = value


def extend_recording(source_path: str | Path, output_path: str | Path, *,
                     duration_s: float = 60., toa_scale: float = 1e-6,
                     dropouts: list[tuple[int, float, float]] | None = None,
                     automatic_dropouts: int = 4,
                     continuation_mode: str = "per-emitter") -> dict:
    """Create a time-sorted extended recording and return its manifest."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(toa_scale) or toa_scale <= 0:
        raise ValueError("toa_scale must be finite and positive")
    if continuation_mode not in {"per-emitter", "whole-recording"}:
        raise ValueError("Invalid continuation_mode")
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Output must not overwrite the source recording")

    with h5py.File(source_path, "r") as source:
        names = feature_names(source)
        if "ToA" not in names or "data" not in source or "labels" not in source:
            raise ValueError("Require data, labels and a ToA feature")
        data = np.asarray(source["data"])
        labels = np.asarray(source["labels"])
        if data.ndim != 2 or len(data) == 0 or labels.shape[0] != len(data):
            raise ValueError("Invalid or empty data/labels datasets")
        toa_index = names.index("ToA")
        order = np.argsort(data[:, toa_index], kind="stable")
        data, labels = data[order], labels[order]
        raw_toa = np.asarray(data[:, toa_index], dtype=float)
        if not np.all(np.isfinite(raw_toa)):
            raise ValueError("ToA contains non-finite values")
        relative_s = (raw_toa - raw_toa[0]) * toa_scale
        positive_steps = np.diff(relative_s)
        positive_steps = positive_steps[positive_steps > 0]
        boundary_step_s = float(np.median(positive_steps)) if len(positive_steps) else toa_scale
        template_period_s = float(relative_s[-1] + boundary_step_s)
        if template_period_s <= 0:
            raise ValueError("Source recording has no positive duration")

        schedule = (choose_default_dropouts(labels, duration_s, automatic_dropouts)
                    if dropouts is None else list(dropouts))
        for label, start, end in schedule:
            if end > duration_s + 1e-12:
                raise ValueError(f"Dropout for emitter {label} exceeds output duration")

        flat_labels = labels.reshape(len(labels), -1)[:, 0]
        emitter_templates = []
        if continuation_mode == "per-emitter":
            for emitter in np.unique(flat_labels):
                mask = flat_labels == emitter
                emitter_times = relative_s[mask]
                local_times = emitter_times - emitter_times[0]
                steps = np.diff(local_times)
                steps = steps[steps > 0]
                boundary = float(np.median(steps)) if len(steps) else template_period_s
                period = float(local_times[-1] + boundary)
                if period <= 0:
                    period = template_period_s
                emitter_templates.append({
                    "label": int(emitter),
                    "data": data[mask],
                    "labels": labels[mask],
                    "offset_s": float(emitter_times[0]),
                    "local_times_s": local_times,
                    "period_s": period,
                    "cycles": max(0, int(math.ceil(
                        max(0., duration_s - float(emitter_times[0])) / period))),
                })
            cycles = max(item["cycles"] for item in emitter_templates)
        else:
            cycles = int(np.ceil(duration_s / template_period_s))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as target:
            _copy_non_stream_data(source, target)
            source_data = source["data"]
            source_labels = source["labels"]
            data_kwargs = {"compression": source_data.compression,
                           "compression_opts": source_data.compression_opts,
                           "shuffle": source_data.shuffle,
                           "fletcher32": source_data.fletcher32}
            label_kwargs = {"compression": source_labels.compression,
                            "compression_opts": source_labels.compression_opts,
                            "shuffle": source_labels.shuffle,
                            "fletcher32": source_labels.fletcher32}
            extended = target.create_dataset("data", shape=(0, data.shape[1]),
                maxshape=(None, data.shape[1]), dtype=source_data.dtype,
                chunks=source_data.chunks or True, **data_kwargs)
            extended_labels = target.create_dataset("labels",
                shape=(0,) + labels.shape[1:], maxshape=(None,) + labels.shape[1:],
                dtype=source_labels.dtype, chunks=source_labels.chunks or True,
                **label_kwargs)
            output_pdws = 0

            def append(block, block_labels):
                nonlocal output_pdws
                next_size = output_pdws + len(block)
                extended.resize(next_size, axis=0)
                extended_labels.resize(next_size, axis=0)
                extended[output_pdws:next_size] = block
                extended_labels[output_pdws:next_size] = block_labels
                output_pdws = next_size

            if continuation_mode == "whole-recording":
                for cycle in range(cycles):
                    shifted_s = relative_s + cycle * template_period_s
                    keep = shifted_s < duration_s
                    for label, start, end in schedule:
                        keep &= ~((flat_labels == label) & (shifted_s >= start) & (shifted_s < end))
                    if not np.any(keep):
                        continue
                    block = data[keep].copy()
                    block[:, toa_index] = shifted_s[keep] / toa_scale
                    append(block, labels[keep])
            else:
                dropout_by_label = {}
                for label, start, end in schedule:
                    dropout_by_label.setdefault(label, []).append((start, end))
                # Merge one second at a time so the full extended recording
                # never needs to reside in memory and remains globally sorted.
                for chunk_start in np.arange(0., duration_s, 1.):
                    chunk_end = min(duration_s, float(chunk_start + 1.))
                    blocks, block_label_sets = [], []
                    for item in emitter_templates:
                        local = item["local_times_s"]
                        offset, period = item["offset_s"], item["period_s"]
                        first_cycle = max(0, int(math.ceil(
                            (chunk_start - (offset + local[-1])) / period)))
                        last_cycle = int(math.floor(
                            (chunk_end - offset - 1e-12) / period))
                        for cycle in range(first_cycle, last_cycle + 1):
                            shifted_s = offset + local + cycle * period
                            keep = ((shifted_s >= chunk_start)
                                    & (shifted_s < chunk_end)
                                    & (shifted_s < duration_s))
                            for start, end in dropout_by_label.get(item["label"], []):
                                keep &= ~((shifted_s >= start) & (shifted_s < end))
                            if not np.any(keep):
                                continue
                            block = item["data"][keep].copy()
                            block[:, toa_index] = shifted_s[keep] / toa_scale
                            blocks.append(block)
                            block_label_sets.append(item["labels"][keep])
                    if not blocks:
                        continue
                    block = np.concatenate(blocks)
                    block_labels = np.concatenate(block_label_sets)
                    chunk_order = np.argsort(block[:, toa_index], kind="stable")
                    append(block[chunk_order], block_labels[chunk_order])
            target.attrs["s2b_extension_manifest_json"] = json.dumps({
                "source": source_path.name,
                "duration_s": duration_s,
                "toa_scale": toa_scale,
                "source_toa_origin_raw": float(raw_toa[0]),
                "template_period_s": template_period_s,
                "template_repetition": True,
                "continuation_mode": continuation_mode,
                "emitter_template_periods_s": ({str(item["label"]): item["period_s"]
                    for item in emitter_templates} if emitter_templates else {}),
                "dropouts": [{"truth_emitter": label, "start_s": start, "end_s": end}
                             for label, start, end in schedule],
            }, sort_keys=True)

    return {
        "source": str(source_path),
        "output": str(output_path),
        "source_pdws": int(len(data)),
        "output_pdws": int(output_pdws),
        "duration_s": duration_s,
        "source_toa_origin_raw": float(raw_toa[0]),
        "template_period_s": template_period_s,
        "cycles": cycles,
        "continuation_mode": continuation_mode,
        "dropouts": [{"truth_emitter": label, "start_s": start, "end_s": end}
                     for label, start, end in schedule],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", default="turing_data/config_101_60s_gaps.h5")
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--toa-scale", type=float, default=1e-6)
    parser.add_argument("--dropout", action="append", type=parse_dropout,
                        help="truth emitter and interval as LABEL:START_S:END_S; repeatable")
    parser.add_argument("--automatic-dropouts", type=int, default=4,
                        help="number of high-support truth emitters to silence when no --dropout is supplied")
    parser.add_argument("--continuation-mode", choices=("per-emitter", "whole-recording"),
                        default="per-emitter")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    if args.automatic_dropouts < 0:
        parser.error("--automatic-dropouts must be non-negative")
    manifest = extend_recording(args.source, args.output, duration_s=args.duration,
        toa_scale=args.toa_scale, dropouts=args.dropout,
        automatic_dropouts=args.automatic_dropouts,
        continuation_mode=args.continuation_mode)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output).with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
