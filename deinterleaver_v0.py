"""S2B deinterleaver V0: conservative fixed-PRI pulse-train extraction.

Uses frozen AOA/RF grouping; truth labels are evaluation-only.
V0 is NOT jitter/stagger capable and does not associate across blobs.
Run: python -u deinterleaver_v0.py turing_data/config_101.h5 --window 0 --blob 11
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import h5py
import numpy as np
from scipy.signal import find_peaks
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window


@dataclass
class PulseTrainHypothesis:
    id: str
    pdw_indices: np.ndarray
    timing_model: str
    period: float
    phase: float
    rms_residual: float
    occupied_slots: int
    span_slots: int


def candidates(t, lo, hi, bin_width, max_seeds=24):
    """Histogram positive pairwise lags in a bounded PRI interval."""
    s = np.sort(t)
    lag = []
    for i in range(len(s)):
        j0 = np.searchsorted(s, s[i]+lo, side="left")
        j1 = np.searchsorted(s, s[i]+hi, side="right")
        if j1 > j0:
            lag.extend(s[j0:j1]-s[i])
    if not lag:
        return []
    bins = max(2, int(np.ceil((hi-lo)/bin_width)))
    hist, edges = np.histogram(lag, bins=bins, range=(lo,hi))
    peaks, _ = find_peaks(hist, distance=max(1,int(.5/bin_width)))
    peaks = sorted(set(peaks.tolist()+[int(np.argmax(hist))]),
                   key=lambda k: -hist[k])[:max_seeds]
    return [float((edges[k]+edges[k+1])/2) for k in peaks]


def fit(t, period, tol):
    """Fit a phase gate and keep no more than one pulse per PRI slot."""
    phase = np.mod(t, period)
    order = np.argsort(phase)
    sorted_phase = phase[order]
    extended = np.concatenate((sorted_phase, sorted_phase+period))
    right = np.searchsorted(extended, extended+2*tol, side="right")
    start = int(np.argmax(right[:len(t)]-np.arange(len(t))))
    center = (extended[start]+tol) % period
    residual = (t-center+period/2) % period-period/2
    gated = np.flatnonzero(np.abs(residual)<=tol)
    slots = np.rint((t[gated]-center)/period).astype(np.int64)
    chosen = {}
    for idx, slot in zip(gated, slots):
        error = abs(residual[idx])
        if slot not in chosen or error < chosen[slot][0]:
            chosen[slot] = (error, int(idx))
    idx = np.array(sorted((v[1] for v in chosen.values()), key=lambda i:t[i]), dtype=int)
    rms = float(np.sqrt(np.mean(residual[idx]**2))) if len(idx) else float("inf")
    span = int(np.ptp(np.rint((t[idx]-center)/period).astype(np.int64)))+1 if len(idx) else 0
    return idx, float(center), rms, span


def discover(t, lo, hi, tol, min_pulses, max_tracks, bin_width):
    """Iteratively discover candidate fixed-PRI trains; leave ambiguity unassigned."""
    available = np.ones(len(t), dtype=bool)
    tracks = []
    for iteration in range(max_tracks):
        active = np.flatnonzero(available)
        if len(active) < min_pulses:
            break
        origin = float(t[active].min())
        relative = t[active]-origin
        seeds = candidates(relative, lo, hi, bin_width)
        if not seeds:
            break
        best = None
        for seed in seeds:
            for period in np.arange(max(lo,seed-2*bin_width),
                                    min(hi,seed+2*bin_width)+bin_width/10,
                                    bin_width/4):
                idx, phase, rms, span = fit(relative, period, tol)
                # Reject sparse phase coincidences across an enormous interval.
                if len(idx) < min_pulses or span > 12*len(idx):
                    continue
                rank = (len(idx), -rms)
                if best is None or rank > best[0]:
                    best = (rank, idx, float(period), phase, rms, span, origin)
        if best is None:
            break
        _, idx, period, phase, rms, span, origin = best
        selected = active[idx]
        tracks.append(PulseTrainHypothesis(
            id=f"T{iteration+1}", pdw_indices=selected,
            timing_model="FIXED_CANDIDATE", period=period,
            phase=phase+origin, rms_residual=rms,
            occupied_slots=len(idx), span_slots=span))
        available[selected] = False
    return tracks, np.flatnonzero(available)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--blob", type=int, default=11)
    ap.add_argument("--pri-min", type=float, default=20.)
    ap.add_argument("--pri-max", type=float, default=200.)
    ap.add_argument("--bin-width", type=float, default=.1)
    ap.add_argument("--tolerance", type=float, default=2.)
    ap.add_argument("--min-pulses", type=int, default=20)
    ap.add_argument("--max-tracks", type=int, default=4)
    a = ap.parse_args()
    if not (0<a.pri_min<a.pri_max and a.bin_width>0 and
            0<a.tolerance<a.pri_min/4 and a.min_pulses>=3 and a.max_tracks>=1):
        ap.error("Invalid timing parameters")
    with h5py.File(a.file) as h:
        start = a.window*10000
        stop = min(start+10000, len(h["data"]))
        if start<0 or stop<=start:
            ap.error("Window outside dataset")
        names = [v.decode() if isinstance(v,bytes) else str(v)
                 for v in h["metadata/feature_names"][:]]
        x = np.asarray(h["data"][start:stop], float)
        truth = np.asarray(h["labels"][start:stop]).reshape(-1)
    pops,npop = batch_aoa_populations(x[:,names.index("AoA")])
    blobs = group_window(x,names,pops,npop,8,.20,6.,8)
    original = np.flatnonzero(blobs==a.blob)
    if not len(original):
        ap.error("Blob not found")
    t = x[original,names.index("ToA")]
    tracks, unexplained = discover(t,a.pri_min,a.pri_max,a.tolerance,
                                   a.min_pulses,a.max_tracks,a.bin_width)
    print(f"V0 window={a.window} blob={a.blob} PDWs={len(original)} "
          f"candidate_trains={len(tracks)} unexplained={len(unexplained)}")
    for tr in tracks:
        actual = original[tr.pdw_indices]
        composition = Counter(truth[actual].tolist())
        print(f" {tr.id} {tr.timing_model}: period={tr.period:.4f} "
              f"matched={len(actual)} slots={tr.occupied_slots}/{tr.span_slots} "
              f"rms={tr.rms_residual:.3f} truth_eval={composition}")
    print(f"Unexplained truth_eval={Counter(truth[original[unexplained]].tolist())}")
    print("Caution: V0 is exploratory fixed-PRI only; harmonics, coincidental "
          "phase matches, jitter and stagger are not yet reliably resolved.")


if __name__=="__main__":
    main()
