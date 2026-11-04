"""Truth-separated evaluation for sequential association decisions."""
from __future__ import annotations

from collections import Counter, defaultdict

from .model import AssociationDecision


def evaluate(decisions: list[AssociationDecision], truth_labels: list[int | str | None]) -> dict:
    if len(decisions) != len(truth_labels):
        raise ValueError("A truth entry is required for every decision")
    total_by_truth: Counter[str] = Counter()
    tracks_by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    truth_by_track: dict[str, Counter[str]] = defaultdict(Counter)
    for decision, raw_truth in zip(decisions, truth_labels):
        if raw_truth is None:
            continue
        truth = str(raw_truth)
        total_by_truth[truth] += 1
        if decision.track_id is not None:
            tracks_by_truth[truth][decision.track_id] += 1
            truth_by_track[decision.track_id][truth] += 1

    truth_summary = {}
    for truth, total in sorted(total_by_truth.items(), key=lambda item: item[0]):
        fragments = dict(tracks_by_truth[truth].most_common())
        associated = sum(fragments.values())
        truth_summary[truth] = {
            "total_pdws": total,
            "associated_pdws": associated,
            "coverage": associated / total if total else None,
            "primary_track": next(iter(fragments), None),
            "fragments": fragments,
        }
    track_summary = {}
    correct = 0
    labelled_associations = 0
    for track_id, counts in sorted(truth_by_track.items()):
        total = sum(counts.values())
        dominant, dominant_count = counts.most_common(1)[0]
        correct += dominant_count
        labelled_associations += total
        track_summary[track_id] = {
            "labelled_pdws": total,
            "dominant_truth": dominant,
            "purity": dominant_count / total,
            "truth_counts": dict(counts.most_common()),
        }
    outcomes = Counter(decision.outcome for decision in decisions)
    return {
        "processed_pdws": len(decisions),
        "outcomes": dict(outcomes),
        "labelled_association_purity": correct / labelled_associations if labelled_associations else None,
        "truth_summary": truth_summary,
        "track_summary": track_summary,
    }
