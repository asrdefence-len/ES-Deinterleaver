"""Emitter-level diagnostics for the label-blind Turing DBSCAN baseline.

Ground truth is used ONLY after clustering, for evaluation. The per-emitter
best-cluster recall is descriptive (not one-to-one assignment accuracy).
DBSCAN noise (-1) is excluded from all cluster recovery calculations.
"""
import argparse
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from turing_baseline import build_features, load_sample


def diagnose(truth, predicted, top=15):
    truth=np.asarray(truth).reshape(-1)
    predicted=np.asarray(predicted).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Truth and predicted must be nonempty and aligned")
    ids, true_idx=np.unique(truth,return_inverse=True)
    clusters=np.unique(predicted[predicted != -1])
    assigned=predicted != -1
    counts=np.bincount(true_idx,minlength=len(ids))
    # Cross-tab only assigned pulses; -1 is never considered a recovered cluster.
    tab=np.zeros((len(ids),len(clusters)),dtype=np.int64)
    if len(clusters):
        col=np.searchsorted(clusters,predicted[assigned])
        np.add.at(tab,(true_idx[assigned],col),1)
    rows=[]
    for i,label in enumerate(ids):
        row=tab[i]
        best=int(row.max()) if len(row) else 0
        parts=int(np.count_nonzero(row))
        rows.append({"emitter":str(label),"pdws":int(counts[i]),
                     "assigned":int(row.sum()),"unassigned":int(counts[i]-row.sum()),
                     "clusters_touched":parts,
                     "best_cluster":str(clusters[int(row.argmax())]) if best else None,
                     "best_cluster_pdws":best,
                     "best_cluster_recall":float(best/counts[i])})
    cluster_rows=[]
    for j,cluster in enumerate(clusters):
        col=tab[:,j]
        size=int(col.sum())
        dominant=int(col.max())
        cluster_rows.append({"cluster":str(cluster),"pdws":size,
                             "emitters_mixed":int(np.count_nonzero(col)),
                             "dominant_emitter":str(ids[int(col.argmax())]),
                             "purity":float(dominant/size)})
    recalls=np.array([r["best_cluster_recall"] for r in rows])
    # No tuning on ground truth: these scores are evaluation-only.
    return {"pdws":int(len(truth)),"truth_labels":len(ids),
            "predicted_clusters":len(clusters),
            "unassigned_pdws":int((~assigned).sum()),
            "macro_best_cluster_recall":float(recalls.mean()),
            "median_best_cluster_recall":float(np.median(recalls)),
            "emitters_split_across_multiple_clusters":sum(r["clusters_touched"]>1 for r in rows),
            "emitters_without_assigned_pdws":sum(r["assigned"]==0 for r in rows),
            "mixed_clusters":sum(r["emitters_mixed"]>1 for r in cluster_rows),
            "ari_all_pdws":float(adjusted_rand_score(truth,predicted)),
            "lowest_recall_emitters":sorted(rows,key=lambda r:(r["best_cluster_recall"],-r["pdws"]))[:top],
            "largest_emitters":sorted(rows,key=lambda r:-r["pdws"])[:top],
            "most_mixed_clusters":sorted(cluster_rows,key=lambda r:(-r["emitters_mixed"],-r["pdws"]))[:top],
            "lowest_purity_clusters":sorted(cluster_rows,key=lambda r:r["purity"])[:top]}


def run(path,limit=10000,eps=0.25,min_samples=8,top=15):
    data,truth,names,total=load_sample(path,limit)
    if truth is None:
        raise ValueError("Ground-truth labels are required for diagnostics")
    if not np.all(np.isfinite(data)):
        raise ValueError("Non-finite PDW measurements")
    pred=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(build_features(data,names))
    result=diagnose(truth,pred,top)
    result.update({"file":str(path),"total_file_pdws":total,"eps":eps,
                   "min_samples":min_samples,"top":top})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("--limit",type=int,default=10000)
    p.add_argument("--eps",type=float,default=0.25)
    p.add_argument("--min-samples",type=int,default=8)
    p.add_argument("--top",type=int,default=15)
    p.add_argument("--output",default=None,help="Optional JSON output file")
    a=p.parse_args()
    if a.eps<=0 or a.min_samples<1 or a.top<1:
        p.error("eps, min-samples and top must be positive")
    report=json.dumps(run(a.file,a.limit,a.eps,a.min_samples,a.top),indent=2)
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(report+"\n",encoding="utf-8")
        print(f"Saved {a.output}")
    else:
        print(report)


if __name__=="__main__":
    main()
