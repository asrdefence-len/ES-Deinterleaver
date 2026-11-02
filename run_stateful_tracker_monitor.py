"""Run and monitor the covariance-based tracker on the long baseline test."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from run_cross_rf_confidence import circular_mean_deg, group_stats, make_pdws
from testbed.pulse_association_pipeline import PulseMeasurement
from testbed.stateful_emitter_tracker import StatefulEmitterTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--seed-blobs", type=int, nargs="+", default=list(range(8, 16)))
    parser.add_argument("--evaluation-truth-id", type=int, default=46)
    parser.add_argument("--monitor-s", type=float, default=.1)
    parser.add_argument("--random-seed", type=int, default=4602)
    parser.add_argument("--output", default="results/emitter46_confidence/stateful_tracker_monitor.json")
    args = parser.parse_args()

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        data = np.asarray(handle["data"][:], float)
        truth = np.asarray(handle["labels"][:]).reshape(-1)
    n = len(data)
    raw = make_pdws(data, names, np.arange(n), 0)
    origin_s, end_s = raw[0].toa_s, raw[-1].toa_s
    duration_s = end_s - origin_s

    seed_data = data[:10000]
    populations, count = batch_aoa_populations(seed_data[:, names.index("AoA")])
    labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
    indices = np.flatnonzero(np.isin(labels, args.seed_blobs))
    seed_pdws = make_pdws(seed_data, names, indices, 0)
    stats = group_stats(seed_data, names, labels)
    modes = tuple(sorted(stats[blob]["frequency_hz"] for blob in args.seed_blobs))
    e46_aoa = circular_mean_deg([pdw.aoa_deg for pdw in seed_pdws])
    e46_pw = float(np.median([pdw.pulse_width_s for pdw in seed_pdws]))

    rng = np.random.default_rng(args.random_seed)
    relative = np.arange(.02, duration_s - .02, 100e-6)
    detected = rng.random(len(relative)) < .24
    gaps = ((relative >= 1.40) & (relative < 1.80)) | (
            (relative >= 3.65) & (relative < 4.15))
    relative = relative[detected & ~gaps]
    approach = 2.0 * np.exp(-.5 * ((relative - 2.75) / .24) ** 2)
    hop = np.asarray([0, 3, 6, 1, 4, 7, 2, 5])
    competitor = [PulseMeasurement(n + index + 1,
        origin_s + float(time_s + rng.normal(0, 1.2e-6)),
        float(modes[hop[index % len(hop)]] + rng.normal(0, 3e6)),
        float(max(.02e-6, 1e-6 + rng.normal(0, .05e-6))),
        float(e46_aoa + 6. - approach[index] + rng.normal(0, .18)))
        for index, time_s in enumerate(relative)]
    competitor.sort(key=lambda pdw: (pdw.toa_s, pdw.pdw_id))

    tracker = StatefulEmitterTracker(
        gate_sigma=3., min_confirm_pulses=5, confirmation_span_s=.010,
        confirmation_confidence=.70, ambiguity_margin=.10,
        # Thousands of five-pulse confirmations occur in one run, so each
        # batch contributes only a small existence-evidence increment.
        hit_log_odds=.002, miss_log_odds_s=6.)
    tracker.add_track("E46_TRACK", time_s=origin_s, aoa_deg=e46_aoa,
        pulse_width_s=e46_pw, frequency_modes_hz=modes,
        pri_s=85.089e-6, phase_epoch_s=origin_s,
        existence_confidence=.8, timing_enabled=False)
    tracker.add_track("COMPETITOR_TRACK", time_s=origin_s, aoa_deg=e46_aoa + 6.,
        pulse_width_s=1e-6, frequency_modes_hz=modes,
        pri_s=100e-6, phase_epoch_s=origin_s,
        aoa_sigma_deg=1., pulse_width_sigma_s=.08e-6,
        existence_confidence=.8, timing_enabled=False)

    raw_code = np.zeros(n, np.int8)
    competitor_code = np.zeros(len(competitor), np.int8)
    competitor_index = {pdw.pdw_id: index for index, pdw in enumerate(competitor)}
    code = {None: 0, "E46_TRACK": 1, "COMPETITOR_TRACK": 2}
    monitor = []
    next_monitor = 0.
    i = j = 0
    while i < len(raw) or j < len(competitor):
        if j >= len(competitor) or (i < len(raw) and
                (raw[i].toa_s, raw[i].pdw_id) <=
                (competitor[j].toa_s, competitor[j].pdw_id)):
            pdw, raw_index, competitor_idx = raw[i], i, None
            i += 1
        else:
            pdw, raw_index = competitor[j], None
            competitor_idx = competitor_index[pdw.pdw_id]
            j += 1
        relative_time = pdw.toa_s - origin_s
        while next_monitor <= relative_time:
            tracker.coast_to(origin_s + next_monitor)
            snapshot = tracker.snapshot(origin_s + next_monitor)
            snapshot["time_from_start_s"] = next_monitor
            monitor.append(snapshot)
            next_monitor += args.monitor_s
        decision = tracker.process_pdw(pdw)
        outcome = 3 if decision.status == "ambiguous" else code[decision.selected_emitter_id]
        if raw_index is not None:
            raw_code[raw_index] = outcome
        else:
            competitor_code[competitor_idx] = outcome
    while next_monitor <= duration_s:
        tracker.coast_to(origin_s + next_monitor)
        snapshot = tracker.snapshot(origin_s + next_monitor)
        snapshot["time_from_start_s"] = next_monitor
        monitor.append(snapshot)
        next_monitor += args.monitor_s

    # Truth-only evaluation starts after all tracker decisions are complete.
    e46 = truth == args.evaluation_truth_id
    background = ~e46
    summary = {
        "duration_s": duration_s,
        "e46_pulses": int(np.sum(e46)),
        "competitor_pulses": len(competitor),
        "e46_correct": int(np.sum(e46 & (raw_code == 1))),
        "e46_wrong_track": int(np.sum(e46 & (raw_code == 2))),
        "e46_ambiguous": int(np.sum(e46 & (raw_code == 3))),
        "competitor_correct": int(np.sum(competitor_code == 2)),
        "competitor_wrong_track": int(np.sum(competitor_code == 1)),
        "competitor_ambiguous": int(np.sum(competitor_code == 3)),
        "background_false_claims": int(np.sum(background & np.isin(raw_code, [1, 2]))),
    }
    result = {"truth_is_evaluation_only": True,
        "confidence_note": "relative model confidence; not yet calibrated",
        "design": {"e46_pri_s": 85.089e-6, "competitor_pri_s": 100e-6,
            "e46_pulse_width_s": e46_pw, "competitor_pulse_width_s": 1e-6,
            "initial_aoa_separation_deg": 6., "minimum_aoa_separation_deg": 4.,
            "gaps_s": [[1.40, 1.80], [3.65, 4.15]], "shared_rf_modes": True},
        "summary": summary, "monitor": monitor}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
