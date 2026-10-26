"""Exploratory TOA-difference histogram, NOT a completed deinterleaver.

All-pair differences create harmonics and cross-emitter peaks. Peak locations
are candidates, not identified emitter PRIs.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PRIPeak:
    interval_s: float
    count: int


def difference_histogram(pdws, min_interval_s=0.0005, max_interval_s=0.003,
                         bin_width_s=0.000005):
    if not 0 < min_interval_s < max_interval_s or bin_width_s <= 0:
        raise ValueError("Invalid histogram interval or bin width")
    times = sorted(p.toa_s for p in pdws)
    bins = Counter()
    for i, t in enumerate(times):
        for later in times[i + 1:]:
            delta = later - t
            if delta > max_interval_s:
                break
            if delta < min_interval_s:
                continue
            index = int((delta - min_interval_s) / bin_width_s)
            bins[index] += 1
    return [(min_interval_s + (i + 0.5) * bin_width_s, bins[i])
            for i in range(int((max_interval_s - min_interval_s) / bin_width_s) + 1)]


def candidate_peaks(histogram, top_n=12, min_spacing_s=0.000015):
    """Local maxima ranked by count with minimum separation."""
    maxima = []
    for i in range(1, len(histogram) - 1):
        if histogram[i][1] > histogram[i - 1][1] and histogram[i][1] >= histogram[i + 1][1]:
            maxima.append(PRIPeak(*histogram[i]))
    maxima.sort(key=lambda peak: peak.count, reverse=True)
    selected = []
    for peak in maxima:
        if all(abs(peak.interval_s - p.interval_s) >= min_spacing_s for p in selected):
            selected.append(peak)
        if len(selected) == top_n:
            break
    return selected
