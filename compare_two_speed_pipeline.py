"""End-to-end *baseline* evaluation: PDWs -> streaming -> windows -> emitter IDs.

No ground truth is supplied to either processor. Ground truth is used ONLY
for scoring after processing. Disjoint 50-ms reporting windows avoid counting
the same PDW more than once. Observations are ordered by window then train.
The upstream fixed-PRI estimator is deliberately NOT adapted to mode changes.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from testbed.generator import generate
from testbed.streaming import StreamingDeinterleaver
from testbed.pulse_train_observation import PulseTrainObservation
from testbed.emitter_track_manager import EmitterTrackManager


def run(scenario_path="scenarios/pri_mode_change.json", window_s=0.05):
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    scenario = json.loads(Path(scenario_path).read_text())
    pdws, truth = generate(scenario)
    upstream = StreamingDeinterleaver(bootstrap_pdws=600)
    downstream = EmitterTrackManager()
    by_id = {p.pdw_id: p for p in pdws}
    assignment_to_emitter = {}
    observations = []
    upstream_unassigned = set()
    pending_windows = defaultdict(list)

    def emit_window(window_index):
        grouped = defaultdict(list)
        for p in pending_windows.pop(window_index, []):
            train_id = upstream.assignments.get(p.pdw_id)
            if train_id is None:
                upstream_unassigned.add(p.pdw_id)
            else:
                grouped[train_id].append(p)
        for train_id, members in sorted(grouped.items()):
            members.sort(key=lambda p: p.toa_s)
            track = upstream.tracks[train_id]
            obs = PulseTrainObservation(
                schema_version=1,
                train_id=f"{train_id}:w{window_index}",
                start_toa_s=members[0].toa_s,
                end_toa_s=members[-1].toa_s,
                pri_s=track.pri_s, pri_sigma_s=None,
                frequency_hz=median(p.frequency_hz for p in members),
                frequency_sigma_hz=None,
                aoa_deg=median(p.aoa_deg for p in members),
                aoa_sigma_deg=None,
                pulse_width_s=median(p.pulse_width_s for p in members),
                pulse_width_sigma_s=None,
                pulse_count=len(members),
                pdw_ids=tuple(p.pdw_id for p in members),
            )
            observations.append(obs)
            emitter_id = downstream.process(obs)
            if emitter_id is not None:
                assignment_to_emitter.update((p.pdw_id, emitter_id) for p in members)

    # The 600-PDW bootstrap may span several reporting windows. We do not
    # emit any window until bootstrap has resolved its buffered assignments.
    # Later, emit windows only after all their PDWs have been processed.
    last_flushed = -1
    for offset in range(0, len(pdws), 100):
        chunk = pdws[offset:offset + 100]
        upstream.process_chunk(chunk)
        for p in chunk:
            pending_windows[int(p.toa_s / window_s)].append(p)
        if upstream.bootstrapped:
            complete_through = int(chunk[-1].toa_s / window_s) - 1
            for index in range(last_flushed + 1, complete_through + 1):
                emit_window(index)
                last_flushed = index
    for index in sorted(pending_windows):
        emit_window(index)

    # Fixed identity mapping from early window, not remapped after PRI change.
    early = [p for p in pdws if p.toa_s < 0.35]
    mapping = {}
    for emitter_id in downstream.tracks:
        votes = Counter(truth[p.pdw_id] for p in early
                        if assignment_to_emitter.get(p.pdw_id) == emitter_id)
        if votes:
            mapping[emitter_id] = votes.most_common(1)[0][0]

    windows = []
    for start, end, label in [(0, 0.4, "before"), (0.4, 0.6, "high_prf"),
                              (0.6, 1.0, "after")]:
        for emitter in ("A", "B", "C"):
            members = [p for p in pdws if start <= p.toa_s < end
                       and truth[p.pdw_id] == emitter]
            upstream_count = sum(p.pdw_id in upstream.assignments for p in members)
            downstream_count = sum(p.pdw_id in assignment_to_emitter for p in members)
            correct = sum(mapping.get(assignment_to_emitter.get(p.pdw_id)) == emitter
                          for p in members)
            windows.append(dict(period=label, truth_emitter=emitter,
                                observed=len(members), upstream_assigned=upstream_count,
                                downstream_assigned=downstream_count,
                                correct_identity=correct,
                                upstream_coverage=upstream_count / len(members) if members else 0,
                                identity_recall=correct / len(members) if members else 0))
    return dict(pdw_count=len(pdws), observation_count=len(observations),
                upstream_tracks=len(upstream.tracks),
                downstream_tracks=len(downstream.tracks),
                pending_observations=len(downstream.pending),
                emitter_mapping_early=mapping, windows=windows,
                upstream_unassigned=len(upstream_unassigned),
                downstream_assignments=assignment_to_emitter)


def main():
    results = run()
    print("PDWs:", results["pdw_count"],
          "| Pulse-train window observations:", results["observation_count"])
    print("Upstream trains:", results["upstream_tracks"],
          "| Downstream emitter IDs:", results["downstream_tracks"],
          "| Ambiguous observations:", results["pending_observations"])
    print("Early-window scoring map (truth used ONLY for evaluation):",
          results["emitter_mapping_early"])
    print("period      radar  PDWs  upstream coverage  downstream identity recall")
    for row in results["windows"]:
        print(f'{row["period"]:11} {row["truth_emitter"]:5} '
              f'{row["observed"]:4}  {row["upstream_coverage"]:8.1%} '
              f'{row["identity_recall"]:22.1%}')
    print("NOTE: upstream still uses fixed PRI; this experiment does not "
          "implement PRI-change discovery or MHT.")


if __name__ == "__main__":
    main()
