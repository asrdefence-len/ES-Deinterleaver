"""Run a multi-second, two-track raw-PDW association stress test."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import h5py
import numpy as np

from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from run_cross_rf_confidence import circular_mean_deg, group_stats, make_pdws
from testbed.pulse_association_pipeline import PulseMeasurement
from testbed.track_aided_pdw_association import (
    TrackAidedPDWAssociator, TrackPrediction,
)


def circular_blend(old, new, amount):
    a = (1 - amount) * np.exp(1j * np.deg2rad(old))
    b = amount * np.exp(1j * np.deg2rad(new))
    return float(np.rad2deg(np.angle(a + b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--seed-window", type=int, default=0)
    parser.add_argument("--seed-blobs", type=int, nargs="+", default=list(range(8, 16)))
    parser.add_argument("--evaluation-truth-id", type=int, default=46)
    parser.add_argument("--bin-s", type=float, default=.1)
    parser.add_argument("--random-seed", type=int, default=4602)
    parser.add_argument("--competitor-pri-us", type=float, default=100.)
    parser.add_argument("--competitor-aoa-offset-deg", type=float, default=6.)
    parser.add_argument("--competitor-min-aoa-offset-deg", type=float, default=4.)
    parser.add_argument("--competitor-pw-us", type=float, default=1.)
    parser.add_argument("--competitor-pw-noise-us", type=float, default=.05)
    parser.add_argument("--output", default="results/emitter46_confidence/long_competing_emitter.json")
    args = parser.parse_args()

    with h5py.File(args.file) as handle:
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["metadata/feature_names"][:]]
        data = np.asarray(handle["data"][:], float)
        truth = np.asarray(handle["labels"][:]).reshape(-1)
    n = len(data)
    origin_s = float(data[0, names.index("ToA")] * 1e-6)
    end_s = float(data[-1, names.index("ToA")] * 1e-6)
    duration_s = end_s - origin_s
    raw = make_pdws(data, names, np.arange(n), 0)

    seed_data = data[:10000]
    populations, count = batch_aoa_populations(seed_data[:, names.index("AoA")])
    seed_labels = group_window(seed_data, names, populations, count, 8, .20, 6., 8)
    seed_indices = np.flatnonzero(np.isin(seed_labels, args.seed_blobs))
    seed_pdws = make_pdws(seed_data, names, seed_indices, 0)
    stats = group_stats(seed_data, names, seed_labels)
    modes = tuple(sorted(stats[blob]["frequency_hz"] for blob in args.seed_blobs))
    base_aoa = circular_mean_deg([pdw.aoa_deg for pdw in seed_pdws])
    pw_values = np.asarray([pdw.pulse_width_s for pdw in seed_pdws])
    base_pw = float(np.median(pw_values))
    pw_sigma = max(.08e-6, float(np.std(pw_values)))

    rng = np.random.default_rng(args.random_seed)
    relative = np.arange(.02, duration_s - .02, args.competitor_pri_us * 1e-6)
    detected = rng.random(len(relative)) < .24
    gaps = ((relative >= 1.40) & (relative < 1.80)) | (
            (relative >= 3.65) & (relative < 4.15))
    relative = relative[detected & ~gaps]
    # A modest approach tests continuity without making the baseline signals
    # nearly indistinguishable.
    approach = args.competitor_aoa_offset_deg - args.competitor_min_aoa_offset_deg
    offset = args.competitor_aoa_offset_deg - approach * np.exp(
        -.5 * ((relative - 2.75) / .24) ** 2)
    hop = np.asarray([0, 3, 6, 1, 4, 7, 2, 5])
    synthetic = []
    for index, (time_s, aoa_offset) in enumerate(zip(relative, offset)):
        mode = modes[hop[index % len(hop)]]
        synthetic.append(PulseMeasurement(
            n + index + 1, origin_s + float(time_s + rng.normal(0, 1.2e-6)),
            float(mode + rng.normal(0, 3e6)),
            float(max(.02e-6, args.competitor_pw_us * 1e-6
                      + rng.normal(0, args.competitor_pw_noise_us * 1e-6))),
            float(base_aoa + aoa_offset + rng.normal(0, .18))))
    synthetic.sort(key=lambda pdw: (pdw.toa_s, pdw.pdw_id))

    predictions = {
        "E46_TRACK": TrackPrediction("E46_TRACK", origin_s, origin_s, modes,
            base_aoa, base_pw, pulse_width_sigma_s=pw_sigma),
        "COMPETITOR_TRACK": TrackPrediction("COMPETITOR_TRACK", origin_s, origin_s,
            modes, base_aoa + args.competitor_aoa_offset_deg,
            args.competitor_pw_us * 1e-6,
            pulse_width_sigma_s=max(.08e-6, args.competitor_pw_noise_us * 1e-6)),
    }
    gate = TrackAidedPDWAssociator(score_threshold=.8, ambiguity_margin=.08,
        min_confirm_pulses=5, confirmation_span_s=.010)
    raw_code = np.zeros(n, np.int8)
    raw_score = np.full(n, np.nan)
    synthetic_code = np.zeros(len(synthetic), np.int8)
    synthetic_score = np.full(len(synthetic), np.nan)
    synthetic_index = {pdw.pdw_id: index for index, pdw in enumerate(synthetic)}
    pdw_lookup = {pdw.pdw_id: pdw for pdw in synthetic}
    prediction_history = {name: [(0., prediction.aoa_deg)]
                          for name, prediction in predictions.items()}
    confirmations = {name: 0 for name in predictions}
    raw_competitor_audit = []
    code = {None: 0, "E46_TRACK": 1, "COMPETITOR_TRACK": 2}

    i = j = 0
    while i < len(raw) or j < len(synthetic):
        if j >= len(synthetic) or (i < len(raw) and
                (raw[i].toa_s, raw[i].pdw_id) <= (synthetic[j].toa_s, synthetic[j].pdw_id)):
            pdw, raw_index = raw[i], i
            i += 1
            is_raw = True
        else:
            pdw, raw_index = synthetic[j], None
            j += 1
            is_raw = False
        prediction_before = {name: (value.aoa_deg, value.pulse_width_s)
                             for name, value in predictions.items()}
        decision = gate.process(pdw, tuple(predictions.values()))
        outcome = 3 if decision.status == "ambiguous" else code[decision.selected_emitter_id]
        top_score = decision.candidates[0].score if decision.candidates else np.nan
        if is_raw:
            raw_code[raw_index] = outcome
            raw_score[raw_index] = top_score
            if decision.selected_emitter_id == "COMPETITOR_TRACK":
                raw_competitor_audit.append({
                    "raw_index": raw_index, "pdw_id": pdw.pdw_id,
                    "time_s": pdw.toa_s - origin_s,
                    "frequency_hz": pdw.frequency_hz,
                    "aoa_deg": pdw.aoa_deg,
                    "pulse_width_s": pdw.pulse_width_s,
                    "decision_status": decision.status,
                    "candidate_scores": {item.emitter_id: item.score
                                         for item in decision.candidates},
                    "candidate_components": {item.emitter_id: {
                        "timing": item.timing_score, "rf": item.rf_score,
                        "aoa": item.aoa_score, "pulse_width": item.pulse_width_score}
                        for item in decision.candidates},
                    "prediction_before": {name: {"aoa_deg": value[0],
                        "pulse_width_s": value[1]}
                        for name, value in prediction_before.items()},
                })
        else:
            index = synthetic_index[pdw.pdw_id]
            synthetic_code[index] = outcome
            synthetic_score[index] = top_score
        if decision.status == "confirmed":
            emitter_id = decision.selected_emitter_id
            batch = []
            for pdw_id in decision.confirmed_pdw_ids:
                if pdw_id <= n:
                    batch.append(raw[pdw_id - 1])
                else:
                    batch.append(pdw_lookup[pdw_id])
            measured_aoa = circular_mean_deg([item.aoa_deg for item in batch])
            measured_pw = float(np.median([item.pulse_width_s for item in batch]))
            old = predictions[emitter_id]
            predictions[emitter_id] = replace(old,
                generated_at_s=pdw.toa_s, last_toa_s=pdw.toa_s,
                aoa_deg=circular_blend(old.aoa_deg, measured_aoa, .12),
                pulse_width_s=.88 * old.pulse_width_s + .12 * measured_pw)
            prediction_history[emitter_id].append(
                (pdw.toa_s - origin_s, predictions[emitter_id].aoa_deg))
            confirmations[emitter_id] += 1

    # Evaluation begins here; truth labels were not consulted above.
    bins = np.arange(0, duration_s + args.bin_s, args.bin_s)
    centers = .5 * (bins[:-1] + bins[1:])
    raw_time = np.asarray([pdw.toa_s - origin_s for pdw in raw])
    synthetic_time = np.asarray([pdw.toa_s - origin_s for pdw in synthetic])
    e46 = truth == args.evaluation_truth_id
    background = ~e46

    def binned(time, mask, outcomes, wanted):
        total = np.histogram(time[mask], bins)[0]
        chosen = np.histogram(time[mask & np.isin(outcomes, wanted)], bins)[0]
        return total, chosen

    e46_total, e46_correct = binned(raw_time, e46, raw_code, [1])
    _, e46_wrong = binned(raw_time, e46, raw_code, [2])
    _, e46_ambiguous = binned(raw_time, e46, raw_code, [3])
    competitor_mask = np.ones(len(synthetic), bool)
    competitor_total, competitor_correct = binned(
        synthetic_time, competitor_mask, synthetic_code, [2])
    _, competitor_wrong = binned(synthetic_time, competitor_mask, synthetic_code, [1])
    _, competitor_ambiguous = binned(synthetic_time, competitor_mask, synthetic_code, [3])
    background_claims = int(np.sum(background & np.isin(raw_code, [1, 2])))
    hijacks = [row for row in raw_competitor_audit
               if truth[row["raw_index"]] == args.evaluation_truth_id]
    e46_values = data[e46]
    e46_aoa_values = e46_values[:, names.index("AoA")]
    e46_pw_values = e46_values[:, names.index("PulseWidth")]

    def rate(values, total):
        return np.divide(values, total, out=np.full_like(values, np.nan, dtype=float),
                         where=total > 0)

    history = {}
    for name, events in prediction_history.items():
        event_time = np.asarray([item[0] for item in events])
        event_aoa = np.asarray([item[1] for item in events])
        history[name] = np.asarray([event_aoa[max(0, np.searchsorted(
            event_time, time, side="right") - 1)] for time in centers]).tolist()
    true_competitor_aoa = (base_aoa + args.competitor_aoa_offset_deg
        - approach * np.exp(-.5 * ((centers - 2.75) / .24) ** 2)).tolist()

    result = {
        "truth_is_evaluation_only": True,
        "duration_s": duration_s,
        "bin_s": args.bin_s,
        "design": {"shared_rf_modes": True,
            "competitor_pri_s": args.competitor_pri_us * 1e-6,
            "detection_probability": .24,
            "gaps_s": [[1.40, 1.80], [3.65, 4.15]],
            "closest_approach_s": 2.75,
            "initial_aoa_offset_deg": args.competitor_aoa_offset_deg,
            "minimum_aoa_offset_deg": args.competitor_min_aoa_offset_deg,
            "competitor_pulse_width_s": args.competitor_pw_us * 1e-6},
        "signal_parameters": {
            "frequency_modes_hz": modes,
            "e46": {"seed_aoa_deg": base_aoa,
                "measured_aoa_min_deg": float(np.min(e46_aoa_values)),
                "measured_aoa_median_deg": float(np.median(e46_aoa_values)),
                "measured_aoa_max_deg": float(np.max(e46_aoa_values)),
                "seed_pulse_width_s": base_pw,
                "measured_pw_min_s": float(np.min(e46_pw_values) * 1e-6),
                "measured_pw_median_s": float(np.median(e46_pw_values) * 1e-6),
                "measured_pw_max_s": float(np.max(e46_pw_values) * 1e-6),
                "reference_pri_s": 85.089e-6},
            "competitor": {"initial_aoa_deg":
                    base_aoa + args.competitor_aoa_offset_deg,
                "minimum_aoa_deg": base_aoa + args.competitor_min_aoa_offset_deg,
                "nominal_pulse_width_s": args.competitor_pw_us * 1e-6,
                "pulse_width_noise_sigma_s": args.competitor_pw_noise_us * 1e-6,
                "pri_s": args.competitor_pri_us * 1e-6}},
        "summary": {
            "real_emitter46_pulses": int(np.sum(e46)),
            "synthetic_competitor_pulses": len(synthetic),
            "e46_correct": int(np.sum(e46 & (raw_code == 1))),
            "e46_wrong_track": int(np.sum(e46 & (raw_code == 2))),
            "e46_ambiguous": int(np.sum(e46 & (raw_code == 3))),
            "competitor_correct": int(np.sum(synthetic_code == 2)),
            "competitor_wrong_track": int(np.sum(synthetic_code == 1)),
            "competitor_ambiguous": int(np.sum(synthetic_code == 3)),
            "background_false_claims": background_claims,
            "confirmations": confirmations,
        },
        "timeline": {"time_s": centers.tolist(),
            "e46_total": e46_total.tolist(),
            "competitor_total": competitor_total.tolist(),
            "e46_correct_rate": rate(e46_correct, e46_total).tolist(),
            "competitor_correct_rate": rate(competitor_correct, competitor_total).tolist(),
            "e46_ambiguous_rate": rate(e46_ambiguous, e46_total).tolist(),
            "competitor_ambiguous_rate": rate(competitor_ambiguous, competitor_total).tolist(),
            "e46_wrong": e46_wrong.tolist(),
            "competitor_wrong": competitor_wrong.tolist(),
            "e46_predicted_aoa": history["E46_TRACK"],
            "competitor_predicted_aoa": history["COMPETITOR_TRACK"],
            "competitor_true_aoa": true_competitor_aoa,
            "e46_reference_aoa": base_aoa},
        "first_e46_hijacks": hijacks[:20],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
