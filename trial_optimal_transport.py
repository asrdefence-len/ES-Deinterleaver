"""Paired, truth-scored experiment: frozen AOA/RF blobs vs OT regrouping.

Run synthetic stress cases:
  python trial_optimal_transport.py --seeds 20
Run an existing Turing recording (set its actual amplitude feature/units):
  python trial_optimal_transport.py --file turing_data/config_101.h5 \
      --amplitude-feature Amplitude --amplitude-scale db --windows 5

This tests a temporal OT feature, NOT the full HACOT/IHACOT algorithm,
not pulse-train extraction, and not persistent emitter tracking.
"""
import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

from compare_aoa_batch import batch_aoa_populations
from compare_hdbscan_all import score
from compare_multiscale_rf import group_window
from testbed.optimal_transport import score_blob_pairs, merge_blob_candidates


FEATURES = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]
CASES = ("shared_scan_hopping", "same_envelope_same_aoa",
         "same_envelope_different_aoa", "sequential_frequency_switch",
         "continuous_same_aoa", "sparse_hopping", "mixed_rf_blob")


def simulate(case, seed):
    """Known synthetic units: seconds, Hz, seconds, degrees, dB power.

    Truth stays separate. Frequencies are separated on purpose to isolate
    regrouping, except mixed_rf_blob, which tests an unrecoverable input mix.
    Each emitter has independent PRI/jitter and missed detections. This is
    a PDW-level test, with no unresolved-IQ-pulse receiver model.
    """
    if case not in CASES:
        raise ValueError("Unknown stress case")
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for emitter in range(2):
        count = 180 if case == "sparse_hopping" else 1800
        # Deliberately distinct pulse timing, even when envelopes coincide.
        pri = 4.0 / count * (1 + 0.043 * emitter)
        toa = np.arange(count) * pri + 0.0007 * emitter
        toa += rng.normal(0, pri * 0.012, count)
        toa = np.maximum(toa, 0)
        keep = (toa < 4) & (rng.random(count) < .85)
        toa = toa[keep]
        n = len(toa)
        same_envelope = case in ("same_envelope_same_aoa", "same_envelope_different_aoa")
        offset = 0 if same_envelope else .22 * emitter
        wrapped = (toa - .18 - offset + .5) % 1 - .5
        power = .003 + np.exp(-.5 * (wrapped / .065)**2)
        if case == "continuous_same_aoa":
            power[:] = 1
        amplitude = -35 + 10 * np.log10(power) + rng.normal(0, .4, n)
        if case == "sequential_frequency_switch":
            channel = np.minimum((toa // (4 / 3)).astype(int), 2)
        else:
            channel = rng.integers(0, 3, n)
        base_rf = 9e9 + emitter * 400e6
        if case == "mixed_rf_blob":
            base_rf = 9e9
        frequency = base_rf + channel * 60e6 + rng.normal(0, 1e5, n)
        bearing = 45 + (12 * emitter if case == "same_envelope_different_aoa" else 0)
        aoa = bearing + rng.normal(0, .2, n)
        pw = 8e-6 + rng.normal(0, .05e-6, n)
        rows.append(np.column_stack((toa, frequency, pw, aoa, amplitude)))
        truth.extend([emitter] * n)
    x = np.vstack(rows)
    order = np.argsort(x[:, 0], kind="stable")
    return x[order], np.asarray(truth)[order]


def frozen_blobs(x, names):
    pops, npop = batch_aoa_populations(x[:, names.index("AoA")])
    return group_window(x, names, pops, npop, 8, .20, 6., 8)


def metrics(labels, truth):
    m = score(labels, truth)
    keep = labels >= 0
    m["coverage_pct"] = 100 * float(keep.mean()) if len(labels) else 0.
    m["ari_grouped"] = float(adjusted_rand_score(truth[keep], labels[keep])) if keep.sum() >= 2 else None
    fragment_counts = [len(np.unique(labels[(truth == emitter) & keep]))
                       for emitter in np.unique(truth)]
    m["mean_fragments_per_emitter"] = float(np.mean(fragment_counts)) if fragment_counts else 0.
    return m


def evaluate_window(x, names, truth, *, case, seed, threshold, aoa_gate,
                    min_pulses, bins, amplitude_feature, amplitude_scale):
    start = perf_counter()
    labels = frozen_blobs(x, names)
    grouping_seconds = perf_counter() - start
    start = perf_counter()
    pairs = score_blob_pairs(
        x[:, names.index("ToA")], labels,
        None if amplitude_scale == "uniform" else x[:, names.index(amplitude_feature)],
        amplitude_scale=amplitude_scale, aoa_deg=x[:, names.index("AoA")],
        min_pulses=min_pulses, bins=bins)
    scoring_seconds = perf_counter() - start
    pair_rows = []
    for p in pairs:
        row = dict(case=case, seed=seed, **asdict(p))
        if truth is not None:
            a = Counter(truth[labels == p.left].tolist())
            b = Counter(truth[labels == p.right].tolist())
            ae, an = a.most_common(1)[0]
            be, bn = b.most_common(1)[0]
            row.update(left_purity=an / p.left_count, right_purity=bn / p.right_count,
                       same_emitter_eval=bool(ae == be),
                       pure_pair_eval=bool(an / p.left_count >= .95 and bn / p.right_count >= .95))
        pair_rows.append(row)
    rows = []
    for method, gate in (("frozen_blobs", None), ("OT_only", None), ("AOA_gated_OT", aoa_gate)):
        start = perf_counter()
        pred = labels.copy() if method == "frozen_blobs" else merge_blob_candidates(
            labels, pairs, max_distance=threshold, aoa_gate_deg=gate)
        merge_seconds = perf_counter() - start
        # Truth enters only here, after all algorithm outputs are fixed.
        m = metrics(pred, truth) if truth is not None else {
            "groups": len(np.unique(pred[pred >= 0])),
            "coverage_pct": 100 * float(np.mean(pred >= 0))}
        rows.append(dict(case=case, seed=seed, method=method, threshold=threshold,
                         pdws=len(x), grouping_seconds=grouping_seconds,
                         scoring_seconds=0. if method == "frozen_blobs" else scoring_seconds,
                         merge_seconds=merge_seconds, **m))
    return rows, pair_rows


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def print_summary(rows):
    print("case | method | groups | purity% | mean fragments/emitter | ARI")
    for case in dict.fromkeys(r["case"] for r in rows):
        for method in ("frozen_blobs", "OT_only", "AOA_gated_OT"):
            rr = [r for r in rows if r["case"] == case and r["method"] == method]
            if not rr:
                continue
            def mean(key):
                vals = [r[key] for r in rr if r.get(key) is not None]
                return f"{np.mean(vals):.3f}" if vals else "n/a"
            print(f"{case} | {method} | {mean('groups')} | {mean('purity_pct')} | "
                  f"{mean('mean_fragments_per_emitter')} | {mean('ari_grouped')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Optional Turing HDF5 recording; otherwise synthetic cases")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=.03)
    parser.add_argument("--aoa-gate", type=float, default=3.)
    parser.add_argument("--min-pulses", type=int, default=100)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--amplitude-feature")
    parser.add_argument("--amplitude-scale", choices=("db", "linear", "uniform"))
    parser.add_argument("--output", default="results/optimal_transport")
    args = parser.parse_args()
    if (min(args.windows, args.window_size, args.seeds) < 1 or args.start_window < 0
            or args.seed_start < 0 or args.bins < 2 or args.min_pulses < 2
            or not np.isfinite(args.threshold) or args.threshold < 0
            or not np.isfinite(args.aoa_gate) or not 0 < args.aoa_gate <= 180):
        parser.error("Invalid trial parameters")
    if args.file and (args.amplitude_scale is None or (
            args.amplitude_scale != "uniform" and not args.amplitude_feature)):
        parser.error("For a recording specify --amplitude-scale and, unless uniform, --amplitude-feature; do not guess units")
    scale = args.amplitude_scale or "db"
    feature = args.amplitude_feature or "Amplitude"
    rows, pairs = [], []
    def run(x, names, truth, case, seed):
        r, p = evaluate_window(x, names, truth, case=case, seed=seed,
            threshold=args.threshold, aoa_gate=args.aoa_gate,
            min_pulses=args.min_pulses, bins=args.bins,
            amplitude_feature=feature, amplitude_scale=scale)
        rows.extend(r)
        pairs.extend(p)

    if args.file:
        import h5py
        with h5py.File(args.file) as h:
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in h["metadata/feature_names"][:]]
            required = {"ToA", "Frequency", "AoA"}
            if scale != "uniform":
                required.add(feature)
            if not required.issubset(names):
                parser.error(f"Missing required features: {sorted(required - set(names))}; available: {names}")
            for wi in range(args.start_window, args.start_window + args.windows):
                start = wi * args.window_size
                end = min(start + args.window_size, len(h["data"]))
                if start >= end:
                    break
                x = np.asarray(h["data"][start:end], float)
                truth = np.asarray(h["labels"][start:end]).reshape(-1) if "labels" in h else None
                if truth is not None and len(truth) != len(x):
                    parser.error("Truth/data length mismatch")
                run(x, names, truth, f"recording_window_{wi}", wi)
    else:
        for case in CASES:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                x, truth = simulate(case, seed)
                run(x, FEATURES, truth, case, seed)
            print(f"Finished {case}: {args.seeds} seeds", flush=True)
    if not rows:
        parser.error("No windows available")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "comparison.csv", rows)
    write_csv(out / "pairs.csv", pairs)
    config = dict(vars(args), effective_amplitude_scale=scale,
                  effective_amplitude_feature=None if scale == "uniform" else feature)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print_summary(rows)
    print(f"Saved {out}. Candidate grouping only; no persistent emitter IDs.")


if __name__ == "__main__":
    main()
