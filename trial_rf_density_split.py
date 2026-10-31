"""Batch before/after experiment for frequency-density steps inside blobs.

  python trial_rf_density_split.py turing_data --output results/rf_density
  python trial_rf_density_split.py turing_data/config_101.h5 --start-window 40 --windows 1

Directories discover .h5, .hdf5 and .hd5 files recursively. Default grouping
matches mixed_rank05_w40_g40.png: AOA + multiscale RF, NOT RF/PW HDBSCAN.
Use --grouping hdbscan to test AOA + RF/PW HDBSCAN instead.
No existing grouping code is changed. Truth is evaluation-only.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter
import warnings

import h5py
import numpy as np
from sklearn.metrics import adjusted_rand_score, completeness_score

from compare_aoa_batch import batch_aoa_populations, rf_pw_groups
from compare_multiscale_rf import group_window
from compare_hdbscan_all import score
from diagnose_rf_histogram_split import fit_density_step


def discover_files(paths):
    files = set()
    for name in paths:
        path = Path(name).expanduser()
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob("*")
                         if p.is_file() and p.suffix.lower() in (".h5", ".hdf5", ".hd5"))
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise ValueError(f"File/directory not found: {name}")
    if not files:
        raise ValueError("No HDF5 files found")
    return sorted(files)


def automatic_step(frequency, *, target_bins=120, min_count=30, bic_gain=10.):
    """Select a per-blob frequency grid, never a universal RF boundary.

    Use rounded 1/2/2.5/5/10 x 10**k bin widths, near span/target_bins.
    This reproduces 25-unit bins for the ~9000–12000 example without fixing
    the boundary or frequency units. A uniform-density null can be wrong
    for a single emitter; the batch audit explicitly measures false splits.
    """
    f = np.asarray(frequency, float)
    if (f.ndim != 1 or not np.all(np.isfinite(f)) or target_bins < 4
            or min_count < 2 or not np.isfinite(bic_gain) or bic_gain < 0):
        raise ValueError("Invalid density-step input/configuration")
    if len(f) < 2 * min_count or np.ptp(f) <= 0:
        return {"accepted": False, "reason": "insufficient_support"}
    nominal = float(np.ptp(f) / target_bins)
    decade = 10. ** np.floor(np.log10(nominal))
    widths = np.array([1., 2., 2.5, 5., 10.]) * decade
    width = float(widths[np.argmin(abs(np.log(widths / nominal)))])
    low = float(np.floor(f.min() / width) * width)
    high = float(np.ceil(f.max() / width) * width)
    result = fit_density_step(f, low, high, width, min_count)
    return dict(result, bin_width=width,
                accepted=bool("cut" in result and result["delta_bic"] > bic_gain))


def split_blobs(frequency, labels, *, target_bins=120, min_count=30, bic_gain=10.):
    """At most one split per original blob; retain every PDW and -1 label."""
    f, lab = np.asarray(frequency, float), np.asarray(labels)
    if f.ndim != 1 or lab.shape != f.shape or not np.issubdtype(lab.dtype, np.integer):
        raise ValueError("Require aligned 1-D frequencies and integer labels")
    out = lab.copy()
    next_label = int(np.max(lab)) + 1 if len(lab) else 0
    diagnostics = []
    for blob in np.unique(lab[lab >= 0]):
        keep = lab == blob
        result = automatic_step(f[keep], target_bins=target_bins,
                                min_count=min_count, bic_gain=bic_gain)
        if not result["accepted"]:
            continue
        cut = result["cut"]
        out[keep & (f >= cut)] = next_label
        diagnostics.append(dict(blob=int(blob), upper_child=next_label,
                                pdws=int(keep.sum()), **result))
        next_label += 1
    return out, diagnostics


def evaluate(labels, truth):
    base = score(labels, truth)
    keep = labels >= 0
    fragment_counts = [len(np.unique(labels[(truth == e) & keep])) for e in np.unique(truth)]
    return dict(base, coverage_pct=100 * float(keep.mean()),
        ari_grouped=float(adjusted_rand_score(truth[keep], labels[keep])) if keep.sum() > 1 else None,
        completeness_grouped=float(completeness_score(truth[keep], labels[keep])) if keep.sum() > 1 else None,
        fragment_excess=sum(max(0, n - 1) for n in fragment_counts),
        mean_fragments_per_emitter=float(np.mean(fragment_counts)))


def evaluate_splits(diagnostics, before, after, truth):
    """Audit only after inference. None of these values can change a split."""
    rows = []
    for diagnostic in diagnostics:
        row = diagnostic.copy()
        if truth is not None:
            parent = before == row["blob"]
            pc = Counter(truth[parent].tolist())
            children = [Counter(truth[parent & (after == child)].tolist())
                        for child in (row["blob"], row["upper_child"])]
            old = sum(pc.values()) - max(pc.values())
            new = sum(sum(c.values()) - max(c.values()) for c in children)
            extra = sum(sum(e in c for c in children) - 1 for e in pc)
            row.update(parent_truth_counts_eval=json.dumps(dict(pc), sort_keys=True),
                child_truth_counts_eval=json.dumps([dict(c) for c in children], sort_keys=True),
                cross_before_eval=old, cross_after_eval=new,
                minority_reduction_eval=old - new, additional_fragments_eval=extra,
                pure_parent_split_eval=len(pc) == 1)
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="HDF5 files and/or directories")
    p.add_argument("--grouping", choices=("multiscale", "hdbscan"), default="multiscale")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--start-window", type=int, default=0)
    p.add_argument("--windows", type=int, default=0, help="Per file; 0 means all remaining windows")
    p.add_argument("--target-bins", type=int, default=120)
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument("--bic-gain", type=float, default=10.)
    p.add_argument("--output", default="results/rf_density")
    a = p.parse_args()
    if (a.window_size < 8 or a.start_window < 0 or a.windows < 0
            or not 4 <= a.target_bins <= 1000 or a.min_count < 2
            or not np.isfinite(a.bic_gain) or a.bic_gain < 0):
        p.error("Invalid experiment parameters")
    try:
        files = discover_files(a.paths)
    except ValueError as exc:
        p.error(str(exc))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, audits, summary = [], [], []
    for path in files:
        first = len(rows)
        print(f"\n{path.name}", flush=True)
        with h5py.File(path) as h:
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in h["metadata/feature_names"][:]]
            required = {"Frequency", "AoA"} | ({"PulseWidth"} if a.grouping == "hdbscan" else set())
            if not required.issubset(names):
                p.error(f"{path}: missing {sorted(required - set(names))}")
            n = len(h["data"])
            limit = (n + a.window_size - 1) // a.window_size
            if a.windows:
                limit = min(limit, a.start_window + a.windows)
            for wi in range(a.start_window, limit):
                start = wi * a.window_size
                stop = min(start + a.window_size, n)
                if stop - start < 8:
                    continue
                x = np.asarray(h["data"][start:stop], float)
                t0 = perf_counter()
                populations, npop = batch_aoa_populations(x[:, names.index("AoA")])
                if a.grouping == "multiscale":
                    before = group_window(x, names, populations, npop, 8, .20, 6., 8)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._hdbscan")
                        before = rf_pw_groups(x, names, populations, npop)
                grouping_seconds = perf_counter() - t0
                t0 = perf_counter()
                after, candidates = split_blobs(x[:, names.index("Frequency")], before,
                    target_bins=a.target_bins, min_count=a.min_count, bic_gain=a.bic_gain)
                split_seconds = perf_counter() - t0
                # Truth is first read after both before/after predictions are fixed.
                truth = np.asarray(h["labels"][start:stop]).reshape(-1) if "labels" in h else None
                if truth is not None and len(truth) != len(x):
                    p.error(f"{path}: truth/data length mismatch")
                for method, labels in (("before", before), ("density_split", after)):
                    m = evaluate(labels, truth) if truth is not None else dict(
                        groups=len(np.unique(labels[labels >= 0])), grouped=int((labels >= 0).sum()),
                        noise=int((labels < 0).sum()))
                    rows.append(dict(file=str(path), window=wi, method=method, pdws=len(x),
                        splits=len(candidates), grouping_seconds=grouping_seconds,
                        split_seconds=split_seconds if method == "density_split" else 0., **m))
                audits.extend(dict(file=str(path), window=wi, **r)
                    for r in evaluate_splits(candidates, before, after, truth))
                if wi == a.start_window or (wi + 1) % 10 == 0 or stop == n:
                    print(f" window {wi}: {stop:,}/{n:,} PDWs; {len(candidates)} split(s)", flush=True)
        for method in ("before", "density_split"):
            rr = [r for r in rows[first:] if r["method"] == method]
            if not rr:
                continue
            total = sum(r["pdws"] for r in rr)
            grouped = sum(r["grouped"] for r in rr)
            row = dict(file=str(path), method=method, windows=len(rr), pdws=total,
                groups=sum(r["groups"] for r in rr), grouped=grouped,
                noise=sum(r["noise"] for r in rr), coverage_pct=100 * grouped / total,
                splits=sum(r["splits"] for r in rr) if method == "density_split" else 0)
            if "cross" in rr[0]:
                cross = sum(r["cross"] for r in rr)
                row.update(cross=cross, purity_pct=100 * (grouped - cross) / grouped if grouped else 0.,
                    fragment_excess=sum(r["fragment_excess"] for r in rr),
                    mean_ari_grouped=float(np.mean([r["ari_grouped"] for r in rr if r["ari_grouped"] is not None])),
                    mean_completeness_grouped=float(np.mean([r["completeness_grouped"] for r in rr
                                                           if r["completeness_grouped"] is not None])))
            summary.append(row)
            print(f" {method}: {row}", flush=True)
        # Checkpoint completed files so long local runs preserve their results.
        write_csv(out / "windows.csv", rows)
        write_csv(out / "splits.csv", audits)
        write_csv(out / "summary.csv", summary)
    if not rows:
        p.error("No available windows")
    config = dict(vars(a), resolved_files=[str(f) for f in files],
        note="One adaptive density-step split per blob; labels are window-local. Truth is evaluation-only.")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nSaved {out}/summary.csv, windows.csv, splits.csv and config.json")
    print("Inspect fragmentation and ARI: purity alone cannot decrease when an existing group is split.")


if __name__ == "__main__":
    main()
