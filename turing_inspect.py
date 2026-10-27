"""Inspect a Turing Synthetic Radar Dataset HDF5 pulse train.

Read-only; requires h5py and numpy. Does not load the entire dataset into RAM.
No emitter ground truth is used for inference.
"""
import argparse
import json
from pathlib import Path


def inspect_file(path,limit=10000):
    import h5py
    import numpy as np
    if limit<1:
        raise ValueError("limit must be positive")
    with h5py.File(path,"r") as f:
        if "data" not in f:
            raise ValueError("Expected HDF5 dataset 'data'")
        data=f["data"]
        if data.ndim!=2:
            raise ValueError("Expected 2D PDW array")
        n,features=data.shape
        labels=f.get("labels")
        if labels is not None and labels.shape[0]!=n:
            raise ValueError("Labels and PDW count differ")
        sample=np.asarray(data[:min(limit,n)])
        sample_labels=np.asarray(labels[:min(limit,n)]).reshape(-1) if labels is not None else None
        names=None
        if "metadata" in f and "feature_names" in f["metadata"]:
            raw=f["metadata"]["feature_names"][:]
            names=[x.decode("utf-8") if isinstance(x,bytes) else str(x) for x in raw]
        if names is not None and len(names)!=features:
            raise ValueError("Metadata feature_names length does not match data columns")
        result=dict(file=str(path),total_pdws=int(n),columns=int(features),
                    feature_names=names,labels_present=labels is not None,
                    sample_pdws=len(sample),sample_unique_emitters=(
                        len(np.unique(sample_labels)) if sample_labels is not None else None),
                    sample_min=[float(x) for x in np.min(sample,axis=0)] if len(sample) else [],
                    sample_max=[float(x) for x in np.max(sample,axis=0)] if len(sample) else [])
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file",type=Path)
    p.add_argument("--limit",type=int,default=10000)
    a=p.parse_args()
    print(json.dumps(inspect_file(a.file,a.limit),indent=2))


if __name__=="__main__":
    main()
