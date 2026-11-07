"""Compact evaluation-only trace for Model 2 full-scene decisions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import PDW
from .model2_birth import Model2BirthDecision


TRACE_SCHEMA_VERSION = 1


class Model2AuditTraceRecorder:
    """Record raw measurements beside already-made tracker decisions.

    Truth labels are stored only after ``Model2AutomaticBirthTracker.process``
    returns. They are evaluation metadata and cannot affect association.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("audit trace capacity must be positive")
        self.capacity = capacity
        self.count = 0
        self.pdw_id = np.empty(capacity, dtype=np.int32)
        self.toa_s = np.empty(capacity, dtype=np.float64)
        self.frequency_mhz = np.empty(capacity, dtype=np.float32)
        self.aoa_deg = np.empty(capacity, dtype=np.float32)
        self.pulse_width_us = np.empty(capacity, dtype=np.float32)
        self.truth_code = np.empty(capacity, dtype=np.int16)
        self.track_code = np.empty(capacity, dtype=np.int16)
        self.outcome_code = np.empty(capacity, dtype=np.uint8)
        self.confirmed_after_update = np.empty(capacity, dtype=np.bool_)
        self.assignment_margin = np.empty(capacity, dtype=np.float32)
        self.timing_boost = np.empty(capacity, dtype=np.float32)
        self._truth_labels: list[str] = []
        self._truth_codes: dict[str, int] = {}
        self._track_ids: list[str] = []
        self._track_codes: dict[str, int] = {}
        self._outcome_labels: list[str] = []
        self._outcome_codes: dict[str, int] = {}

    @staticmethod
    def _code_for(
        value: str,
        labels: list[str],
        codes: dict[str, int],
    ) -> int:
        code = codes.get(value)
        if code is None:
            code = len(labels)
            labels.append(value)
            codes[value] = code
        return code

    def record(
        self,
        pdw: PDW,
        evaluation_truth_label: int | str | None,
        decision: Model2BirthDecision,
    ) -> None:
        if self.count >= self.capacity:
            raise ValueError("audit trace exceeded its declared capacity")
        index = self.count
        truth = "None" if evaluation_truth_label is None else str(
            evaluation_truth_label
        )
        truth_code = self._code_for(
            truth, self._truth_labels, self._truth_codes,
        )
        track_code = -1
        margin = float("nan")
        boost = 0.0
        if decision.track_id is not None:
            track_code = self._code_for(
                decision.track_id, self._track_ids, self._track_codes,
            )
            margin = decision.confirmed_margins.get(
                decision.track_id, float("nan"),
            )
            boost = decision.timing_boosts.get(decision.track_id, 0.0)
        outcome_code = self._code_for(
            decision.outcome, self._outcome_labels, self._outcome_codes,
        )

        self.pdw_id[index] = pdw.pdw_id
        self.toa_s[index] = pdw.toa_s
        self.frequency_mhz[index] = pdw.frequency_hz / 1e6
        self.aoa_deg[index] = pdw.aoa_deg
        self.pulse_width_us[index] = pdw.pulse_width_s * 1e6
        self.truth_code[index] = truth_code
        self.track_code[index] = track_code
        self.outcome_code[index] = outcome_code
        self.confirmed_after_update[index] = (
            decision.track_confirmed_after_update
        )
        self.assignment_margin[index] = margin
        self.timing_boost[index] = boost
        self.count += 1

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("audit trace path must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        stop = self.count
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(
                    [TRACE_SCHEMA_VERSION], dtype=np.int16,
                ),
                pdw_id=self.pdw_id[:stop],
                toa_s=self.toa_s[:stop],
                frequency_mhz=self.frequency_mhz[:stop],
                aoa_deg=self.aoa_deg[:stop],
                pulse_width_us=self.pulse_width_us[:stop],
                truth_code=self.truth_code[:stop],
                truth_labels=np.asarray(self._truth_labels, dtype="U"),
                track_code=self.track_code[:stop],
                track_ids=np.asarray(self._track_ids, dtype="U"),
                outcome_code=self.outcome_code[:stop],
                outcome_labels=np.asarray(self._outcome_labels, dtype="U"),
                confirmed_after_update=self.confirmed_after_update[:stop],
                assignment_margin=self.assignment_margin[:stop],
                timing_boost=self.timing_boost[:stop],
            )
