"""Diagnose config 101 candidate-group contamination without changing the grouping baseline.

Truth labels are used exclusively AFTER blind grouping, for diagnostics.
"""
import argparse
from collections import Counter, defaultdict
import h5py
import numpy as np
from candidate_grouping import group_pdws


def fmt(x):
    return f"{x:.5g}"


def describe(x, names, indices):
    parts = []
    for name in ("Frequency", "PulseWidth", "AoA"):
        if name not in names:
            continue
        v = x[indices, names.index(name)]
        q = np.percentile(v, [5, 50, 95])
        parts.append(f"{name} p05/50/95={','.join(map(fmt,q))}")
    return "; ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="turing_data/config_101.h5")
    p.add_argument("--window-size", type=int, default=10000)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--window", type=int, default=None, help="Inspect only this zero-based window")
    args = p.parse_args()
    if args.window_size < 8 or args.top < 1:
        p.error("window-size must be >=8 and top >=1")
    windows, mixed, pair_total = [], [], Counter()
    with h5py.File(args.path) as f:
        names = [v.decode() if isinstance(v,bytes) else str(v) for v in f["metadata/feature_names"][:]]
        n = len(f["data"])
        print(f"Dataset: {args.path}, PDWs={n}, features={names}", flush=True)
        starts = ([args.window * args.window_size] if args.window is not None
                  else range(0,n,args.window_size))
        for start in starts:
            stop = min(start + args.window_size,n)
            if start < 0 or stop-start < 8:
                continue
            x = np.asarray(f["data"][start:stop], dtype=float)
            truth = np.asarray(f["labels"][start:stop]).reshape(-1)
            pred = group_pdws(x,names)
            contamination, detail = 0, []
            for gid in np.unique(pred[pred>=0]):
                ids = np.flatnonzero(pred==gid)
                counts = Counter(truth[ids].tolist())
                major, major_n = counts.most_common(1)[0]
                wrong = len(ids)-major_n
                contamination += wrong
                if wrong:
                    members = counts.most_common()
                    for other, other_n in members[1:]:
                        pair_total[tuple(sorted((str(major),str(other))))] += other_n
                    detail.append(dict(start=start, gid=int(gid), ids=ids, x=x,
                                       names=names, counts=members, wrong=wrong))
            grouped = int(np.sum(pred>=0))
            windows.append((start,stop,contamination,grouped,int(np.sum(pred<0)),len(np.unique(pred[pred>=0]))))
            mixed.extend(detail)
    windows.sort(key=lambda w:w[2],reverse=True)
    mixed.sort(key=lambda d:d["wrong"],reverse=True)
    print("\nWORST WINDOWS (ranked by cross-emitter PDWs)")
    for start,stop,c,g,noise,ng in windows[:args.top]:
        print(f"  {start:>8}:{stop:<8} cross={c:>6} purity={100*(g-c)/g if g else 0:7.3f}% ungrouped={noise:>5} groups={ng}")
    print("\nWORST MIXED GROUPS (ranked by cross-emitter PDWs)")
    for d in mixed[:args.top]:
        print(f"  window={d['start']//args.window_size} group={d['gid']} wrong={d['wrong']} size={len(d['ids'])} emitter_counts={d['counts'][:8]}")
        print("    ALL: "+describe(d["x"],d["names"],d["ids"]))
        for emitter,count in d["counts"][:3]:
            sub=d["ids"][np.asarray([str(v)==str(emitter) for v in np.asarray(d["counts"][:0])])] if False else d["ids"]
            # Reload the specific window to describe each truth-labelled component.
            with h5py.File(args.path) as f:
                start=d["start"]
                labels=np.asarray(f["labels"][start:start+len(d["x"])]).reshape(-1)
            sub=d["ids"][labels[d["ids"]]==emitter]
            print(f"    emitter={emitter} n={count}: "+describe(d["x"],d["names"],sub))
    print("\nTOP DOMINANT/CONTAMINANT PAIRS (descriptive, not inferred tracks)")
    for pair,count in pair_total.most_common(args.top):
        print(f"  {pair}: {count} cross-emitter PDWs")
    print("\nNOTE: This diagnostic uses truth only for scoring. Group IDs are window-local.")


if __name__ == "__main__":
    main()
