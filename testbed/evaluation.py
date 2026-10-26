"""Truth scoring, with track labels treated as arbitrary."""
from collections import Counter, defaultdict

def evaluate(truth, assignments):
    if set(truth) != set(assignments):
        raise ValueError("Every observed PDW must have exactly one assignment")
    by_track = defaultdict(Counter)
    by_emitter = defaultdict(Counter)
    for pdw_id, emitter in truth.items():
        track = assignments[pdw_id]
        by_track[track][emitter] += 1
        by_emitter[emitter][track] += 1
    total = len(truth)
    # Majority-label purity is not a global one-to-one matching score.
    purity = sum(max(c.values()) for c in by_track.values()) / total if total else 0.0
    fragmentation = {e: len(c) for e, c in by_emitter.items()}
    completeness = {e: max(c.values()) / sum(c.values()) for e, c in by_emitter.items()}
    return {
        "observed_pdws": total,
        "true_emitters_observed": len(by_emitter),
        "estimated_tracks": len(by_track),
        "track_purity": purity,
        "emitter_completeness": completeness,
        "tracks_per_emitter": fragmentation,
        "spurious_track_excess": max(0, len(by_track)-len(by_emitter)),
        "note": "Track purity can be inflated by fragmentation; inspect completeness and tracks_per_emitter together."
    }
