"""Plot E46-only confidence, identity fragments, PRI inputs and pulse coverage."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result')
    parser.add_argument('--output')
    args = parser.parse_args()
    source = Path(args.result)
    run = json.loads(source.read_text())
    if set(run['families']) != {'E46'}:
        parser.error('This plot requires the E46-only scenario')
    output = Path(args.output) if args.output else source.with_name('e46_overview.png')
    primary = run['track_ids']['E46']
    rows = run['monitor']
    events = run['events']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panel = axes[0, 0]
    t = [r['time_from_start_s'] for r in rows]
    confidence = [100*r['tracks']['E46']['track_confidence'] if 'E46' in r['tracks'] else np.nan for r in rows]
    panel.plot(t, confidence, lw=2, color='tab:blue', label=f'Primary {primary} existence')
    panel.axhline(80, ls='--', color='tab:green', lw=.8, label='Confirmed threshold')
    panel.axhline(30, ls='--', color='tab:red', lw=.8, label='Lost threshold')
    panel.set(title='E46 primary-track confidence', ylabel='Existence probability (%)', ylim=(-3, 105))
    present = [time for time, value in zip(t, confidence) if np.isfinite(value)]
    if present:
        panel.text(.02, .05, f'Last primary state at {max(present):.2f}s; blanks mean absent',
                   transform=panel.transAxes, fontsize=9)
    panel.legend(fontsize=8, loc='center right')

    tracks = list(dict.fromkeys(e['canonical_track_id'] for e in events if e['canonical_track_id']))
    ranks = {track: index+1 for index, track in enumerate(tracks)}
    panel = axes[0, 1]
    for track in tracks:
        selected = [e for e in events if e['canonical_track_id'] == track]
        panel.scatter([e['time_from_start_s'] for e in selected], [ranks[track]]*len(selected),
                      color='tab:orange' if track == primary else 'tab:blue', s=10, alpha=.8)
    panel.set_yticks(list(ranks.values()), labels=tracks, fontsize=6)
    panel.set(title=f'All {len(tracks)} assigned identities (orange = original primary)', ylabel='Canonical track ID')

    panel = axes[1, 0]
    hist_events = [e for e in events if e['measurement']['pri_histogram'] is not None]
    histograms = [e['measurement']['pri_histogram'] for e in hist_events]
    panel.errorbar([e['time_from_start_s'] for e in hist_events],
                   [h['mean_s']*1e6 for h in histograms],
                   yerr=[h['spread_s']*1e6 for h in histograms], fmt='.', ms=3,
                   color='tab:blue', ecolor='lightsteelblue', elinewidth=.5,
                   label='Selected-gap mean ± SD')
    for index, color in [(0, 'tab:green'), (1, 'tab:red')]:
        pairs = [(e, h) for e,h in zip(hist_events,histograms) if len(h['peaks_s']) > index]
        panel.scatter([e['time_from_start_s'] for e,h in pairs],
                      [h['peaks_s'][index]*1e6 for e,h in pairs], marker='_', s=25,
                      color=color, alpha=.7, label=f'Hump {index+1}')
    panel.set(title=f'Actual PRI estimator inputs ({len(hist_events)}/{len(events)} blobs)',
              ylabel='Selected observed gap (µs)')
    panel.legend(fontsize=8)

    # Evaluation and measurement coverage share real window end times, not a
    # uniform sample index. No truth/configuration is used to drive inference.
    window_rows = {r['window']:r for r in rows if r['window'] is not None}
    times, truth, selected = [], [], []
    for row in run['evaluation']:
        if row['window'] not in window_rows:
            continue
        times.append(window_rows[row['window']]['time_from_start_s'])
        truth.append(row['E46']['truth_pulses'])
        selected.append(row['E46']['selected_truth_pulses'])
    panel = axes[1, 1]
    panel.step(times, truth, where='pre', color='black', lw=1.2, label='E46 truth pulses available (evaluation)')
    panel.step(times, selected, where='pre', color='tab:blue', lw=1.1, label='E46 pulses selected for tracking')
    panel.set(title='Pulse coverage and gaps', ylabel='Pulses per processing window')
    panel.legend(fontsize=8)
    for panel in axes.flat:
        panel.set_xlim(0,run['duration_s'])
        panel.set_xlabel('Time from Config 101 start (s)')
        panel.grid(alpha=.2)
    retained = run['canonical_primary_retention']['E46']
    revision = run['run_manifest']['git_revision'][:7]
    fig.suptitle(f'E46-only v1 | AOA + HDBSCAN → gap-histogram PRI → Kalman + basic management | {revision}\n'
                 f'Original primary retains {retained["retained"]}/{retained["total"]} blob observations; nominal confidence remains uncalibrated', fontsize=13)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
