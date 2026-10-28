"""Truth-assisted diagnostic of batch AOA/RF-PW candidate-group fragmentation.

Run: python -u analyse_candidate_fragmentation.py turing_data/config_101.h5
Operational grouping never uses truth. Truth is used only for diagnostics.
A change in RF/PW is not proof of a radar operating-mode change.
"""
import argparse
import csv
from collections import Counter, defaultdict
import h5py
import numpy as np
from compare_aoa_batch import batch_aoa_populations, rf_pw_groups


def circular_mean(a):
    rad = np.deg2rad(np.asarray(a, float))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def emit_value(x):
    return x.item() if isinstance(x, np.generic) else x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--bin-deg", type=float, default=0.25)
    p.add_argument("--smooth-deg", type=float, default=0.5)
    p.add_argument("--min-peak-count", type=int, default=8)
    p.add_argument("--min-separation-deg", type=float, default=1.5)
    p.add_argument("--prefix", default="turing_fragmentation")
    p.add_argument("--focus-window", type=int, default=0)
    a = p.parse_args()
    window_rows, emitter_rows, group_rows = [], [], []
    with h5py.File(a.file) as f:
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["metadata/feature_names"][:]]
        n = len(f["data"])
        print(f"{a.file}: {n:,} PDWs", flush=True)
        for wi, start in enumerate(range(0, n, a.window_size)):
            stop = min(n, start + a.window_size)
            if stop - start < 8:
                continue
            x = np.asarray(f["data"][start:stop], float)
            truth = np.asarray(f["labels"][start:stop]).reshape(-1)
            aoa = x[:, names.index("AoA")]
            toa = x[:, names.index("ToA")]
            rf = x[:, names.index("Frequency")]
            pw = x[:, names.index("PulseWidth")]
            pops, npop = batch_aoa_populations(
                aoa, a.bin_deg, a.smooth_deg,
                a.min_peak_count, a.min_separation_deg)
            pred = rf_pw_groups(x, names, pops, npop)
            grouped = pred >= 0
            emitter_ids, truth_inverse = np.unique(truth, return_inverse=True)
            groups = np.unique(pred[grouped])
            # One traversal to collect group composition, plus per-emitter memberships.
            emitter_to_groups = defaultdict(Counter)
            group_to_emitters = defaultdict(Counter)
            for i in np.flatnonzero(grouped):
                eid = int(truth_inverse[i])
                gid = int(pred[i])
                emitter_to_groups[eid][gid] += 1
                group_to_emitters[gid][eid] += 1
            dominant_group_for = {}
            for gid, comp in group_to_emitters.items():
                dominant = comp.most_common(1)[0][0]
                dominant_group_for[gid] = dominant
                ii = np.flatnonzero(pred == gid)
                group_rows.append(dict(
                    window=wi, group=gid, size=len(ii),
                    dominant_emitter=str(emit_value(emitter_ids[dominant])),
                    dominant_fraction=comp[dominant] / len(ii),
                    emitter_count=len(comp), aoa_mean_deg=circular_mean(aoa[ii]),
                    rf_median=float(np.median(rf[ii])),
                    pw_median=float(np.median(pw[ii])),
                    toa_start=float(toa[ii].min()), toa_end=float(toa[ii].max())))
            per_emitter = []
            for eid, name in enumerate(emitter_ids):
                ii = np.flatnonzero(truth_inverse == eid)
                cnt = emitter_to_groups[eid]
                assigned = sum(cnt.values())
                largest = max(cnt.values(), default=0)
                dominant_groups = sum(dominant_group_for.get(gid) == eid for gid in cnt)
                # Count groups with >= 10% of this emitter's grouped pulses;
                # diagnostic only, not validated operating-mode classification.
                significant_groups = sum(v >= max(8, 0.10 * assigned) for v in cnt.values())
                r = dict(
                    window=wi, emitter=str(emit_value(name)), pdws=len(ii),
                    grouped=assigned, unassigned=len(ii) - assigned,
                    coverage=assigned / len(ii),
                    groups_touched=len(cnt),
                    groups_dominated=dominant_groups,
                    significant_groups=significant_groups,
                    largest_group_pdws=largest,
                    largest_group_fraction_total=largest / len(ii),
                    largest_group_fraction_grouped=largest / assigned if assigned else 0.,
                    aoa_mean_deg=circular_mean(aoa[ii]),
                    aoa_p05_deg=float(np.percentile(
                        (aoa[ii] - circular_mean(aoa[ii]) + 180) % 360 - 180, 5)),
                    aoa_p95_deg=float(np.percentile(
                        (aoa[ii] - circular_mean(aoa[ii]) + 180) % 360 - 180, 95)),
                    rf_p05=float(np.percentile(rf[ii], 5)),
                    rf_p95=float(np.percentile(rf[ii], 95)),
                    pw_p05=float(np.percentile(pw[ii], 5)),
                    pw_p95=float(np.percentile(pw[ii], 95)))
                emitter_rows.append(r)
                per_emitter.append(r)
            window_rows.append(dict(
                window=wi, start=start, stop=stop, pdws=len(x),
                physical_emitters=len(emitter_ids),
                aoa_populations=npop, candidate_groups=len(groups),
                unassigned=int((~grouped).sum()),
                groups_per_emitter=len(groups) / len(emitter_ids),
                median_groups_touched=float(np.median([r["groups_touched"] for r in per_emitter])),
                median_groups_dominated=float(np.median([r["groups_dominated"] for r in per_emitter])),
                median_largest_group_fraction=float(np.median(
                    [r["largest_group_fraction_total"] for r in per_emitter])),
                mixed_groups=sum(len(v) > 1 for v in group_to_emitters.values())))
            if wi == a.focus_window:
                print(f"\nFOCUS WINDOW {wi}: {len(emitter_ids)} emitters, "
                      f"{len(groups)} groups, {npop} AOA populations")
                print("emitter PDWs grouped groups_touched groups_dominated "
                      "largest_group_fraction")
                for r in sorted(per_emitter, key=lambda r: r["pdws"], reverse=True):
                    print(f"{r['emitter']:>12s} {r['pdws']:6d} {r['grouped']:7d} "
                          f"{r['groups_touched']:14d} {r['groups_dominated']:16d} "
                          f"{r['largest_group_fraction_total']:.3f}")
            if (wi + 1) % 10 == 0 or stop == n:
                print(f"processed {stop:,}/{n:,}", flush=True)
    for suffix, rows in (("windows", window_rows),
                         ("emitters", emitter_rows), ("groups", group_rows)):
        filename = f"{a.prefix}_{suffix}.csv"
        with open(filename, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved", filename)
    print("\nSUMMARY (per 10,000-PDW windows, last window may be partial)")
    print("windows", len(window_rows),
          "median physical emitters", np.median([r["physical_emitters"] for r in window_rows]),
          "median candidate groups", np.median([r["candidate_groups"] for r in window_rows]),
          "median groups/emitter", np.median([r["groups_per_emitter"] for r in window_rows]))
    print("median emitter groups touched", np.median([r["groups_touched"] for r in emitter_rows]),
          "median groups dominated", np.median([r["groups_dominated"] for r in emitter_rows]))
    print("median largest-group recovery fraction",
          np.median([r["largest_group_fraction_total"] for r in emitter_rows]))
    print("IMPORTANT: RF/PW ranges are diagnostic only. This script does NOT infer "
          "mode changes, PRI changes, or distinguish true modes from overfragmentation.")


if __name__ == "__main__":
    main()
