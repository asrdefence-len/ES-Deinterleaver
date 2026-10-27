"""Diagnose contamination introduced by log(PW) grouping, especially emitter 81.

Run: python turing_log_pw_contamination.py turing_data/config_100.h5
Truth is used for diagnostic reporting only, never for clustering.
"""
import argparse
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from turing_baseline import load_sample
from turing_grouping_comparison import groups, robust_scale


def dominant_map(labels, truth):
    result = {}
    for group in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == group)
        counts = Counter(map(int, truth[indices]))
        result[int(group)] = (counts.most_common(1)[0][0], counts, indices)
    return result


def describe(label, group_id, info, data, names):
    leader, counts, indices = info
    print(f"  {label} group {group_id}: n={len(indices)}, dominant={leader}, composition={dict(counts.most_common())}")
    for feature in ("Frequency", "PulseWidth", "AoA", "ToA"):
        vals = data[indices, names.index(feature)]
        print(f"    {feature}: min={vals.min():.6g}, median={np.median(vals):.6g}, max={vals.max():.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--aoa-deg", type=float, default=2.)
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()
    data, truth, names, _ = load_sample(args.file, args.limit)
    if truth is None:
        parser.error("Ground truth is required for evaluation")
    rf = data[:, names.index("Frequency")]
    pw = data[:, names.index("PulseWidth")]
    if np.any(pw <= 0):
        parser.error("PulseWidth must be positive")
    theta = np.deg2rad(data[:, names.index("AoA")])
    rf_scale = robust_scale(rf)
    pw_scale = robust_scale(pw)
    lpw = np.log(pw)
    log_scale = robust_scale(lpw)
    c = groups(data, names, "C", args.aoa_deg, args.min_samples, rf_scale, pw_scale)
    z = np.column_stack((np.sin(theta)/np.deg2rad(args.aoa_deg),
                         np.cos(theta)/np.deg2rad(args.aoa_deg),
                         (rf-np.median(rf))/rf_scale,
                         (lpw-np.median(lpw))/log_scale))
    d = DBSCAN(eps=1., min_samples=args.min_samples).fit_predict(z)
    maps = {"C": dominant_map(c, truth), "D": dominant_map(d, truth)}
    for emitter in (81, 18, 22, 33, 54):
        ids = np.flatnonzero(truth == emitter)
        if not len(ids):
            continue
        print(f"\n=== Emitter {emitter}: {len(ids)} pulses ===")
        for feature in ("Frequency", "PulseWidth", "AoA"):
            vals = data[ids, names.index(feature)]
            print(f"  {feature}: {vals.min():.6g} .. {vals.max():.6g} (median {np.median(vals):.6g})")
        for label, assignments in (("C", c), ("D", d)):
            print(f" {label} assignments:", dict(zip(*np.unique(assignments[ids], return_counts=True))))
            for group_id in sorted(set(map(int, assignments[ids])) - {-1}):
                describe(label, group_id, maps[label][group_id], data, names)
    print("\n=== Additional contamination introduced by D relative to C ===")
    c_wrong = (c >= 0) & np.array([maps["C"][int(g)][0] != int(t) if g >= 0 else False for g,t in zip(c,truth)])
    d_wrong = (d >= 0) & np.array([maps["D"][int(g)][0] != int(t) if g >= 0 else False for g,t in zip(d,truth)])
    new_wrong = d_wrong & ~c_wrong
    fixed_wrong = c_wrong & ~d_wrong
    print(f"C wrong={c_wrong.sum()}, D wrong={d_wrong.sum()}, newly wrong={new_wrong.sum()}, fixed={fixed_wrong.sum()}, net={int(d_wrong.sum())-int(c_wrong.sum())}")
    print("Newly wrong by true emitter:", dict(Counter(map(int, truth[new_wrong])).most_common()))
    print("Newly wrong by D dominant emitter:", dict(Counter(maps["D"][int(d[i])][0] for i in np.flatnonzero(new_wrong)).most_common()))
    print("Fixed by true emitter:", dict(Counter(map(int, truth[fixed_wrong])).most_common()))
    print("\n=== D groups containing newly wrong PDWs ===")
    for group_id, n in Counter(map(int,d[new_wrong])).most_common():
        describe("D", group_id, maps["D"][group_id], data, names)
        print(f"    newly wrong in group={n}")
    print("\nDiagnostic only: no grouping parameters changed.")


if __name__ == "__main__":
    main()
