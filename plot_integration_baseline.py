"""Plot all identity fragments and the actual histogram PRI input summaries."""
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
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    source = Path(args.result)
    result = json.loads(source.read_text())
    output = Path(args.output_dir) if args.output_dir else source.parent
    output.mkdir(parents=True, exist_ok=True)
    families = list(result['families'])
    fig, axes = plt.subplots(2, len(families), figsize=(14, 10), squeeze=False,
                             constrained_layout=True, gridspec_kw={'height_ratios': [3, 1]})
    pri_fig, pri_axes = plt.subplots(2, len(families), figsize=(14, 8), squeeze=False,
                                     constrained_layout=True, gridspec_kw={'height_ratios': [3, 1]})
    for col, family in enumerate(families):
        truth_id = str(result['families'][family]['truth_id'])
        # Include each labelled pulse's assigned canonical track, as in the
        # benchmark record. Truth only selects the post-inference visualization.
        events = sorted([e for e in result['events'] if e['truth_counts_eval'].get(truth_id, 0) > 0],
                        key=lambda e: e['time_from_start_s'])
        tracks = list(dict.fromkeys(e['canonical_track_id'] for e in events if e['canonical_track_id']))
        rank = {track: i + 1 for i, track in enumerate(tracks)}
        primary = result['track_ids'][family]
        panel = axes[0, col]
        for track in tracks:
            selected = [e for e in events if e['canonical_track_id'] == track]
            panel.scatter([e['time_from_start_s'] for e in selected], [rank[track]] * len(selected),
                s=[8 + 2 * np.sqrt(e['truth_counts_eval'][truth_id]) for e in selected],
                color='tab:orange' if track == primary else 'tab:blue', alpha=.8)
        unassigned = [e for e in events if e['canonical_track_id'] is None]
        if unassigned:
            panel.scatter([e['time_from_start_s'] for e in unassigned], [0]*len(unassigned),
                          marker='x', color='tab:red', label='Unassigned')
        panel.set_yticks(list(rank.values()), labels=tracks, fontsize=7)
        panel.set(title=f'{family}: {len(tracks)} canonical track fragments', ylabel='Assigned track ID')
        panel.text(.02, .98, f'Orange: original primary {primary}', transform=panel.transAxes,
                   va='top', fontsize=9, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        seen, times, counts = set(), [0.], [0]
        for event in events:
            if event['canonical_track_id']:
                seen.add(event['canonical_track_id'])
            times.append(event['time_from_start_s'])
            counts.append(len(seen))
        times.append(result['duration_s']);counts.append(len(seen))
        axes[1, col].step(times, counts, where='post', color='tab:blue')
        axes[1, col].set(ylabel='Distinct track IDs observed', title='Cumulative distinct identities')

        source_events = [e for e in result['events'] if e['source_family'] == family]
        with_pri = [e for e in source_events if e['measurement'].get('pri_histogram') is not None]
        no_pri = [e for e in source_events if e['measurement'].get('pri_histogram') is None]
        t = [e['time_from_start_s'] for e in with_pri]
        histogram = [e['measurement']['pri_histogram'] for e in with_pri]
        pri_axes[0, col].errorbar(t, [h['mean_s'] * 1e6 for h in histogram],
            yerr=[h['spread_s'] * 1e6 for h in histogram], fmt='o', ms=3,
            color='tab:blue', ecolor='lightsteelblue', elinewidth=.6, alpha=.7,
            label='Selected-gap mean ± SD')
        for index, color in [(0, 'tab:green'), (1, 'tab:red')]:
            valid = [(e, h) for e, h in zip(with_pri, histogram) if len(h['peaks_s']) > index]
            pri_axes[0, col].scatter([e['time_from_start_s'] for e, _ in valid],
                [h['peaks_s'][index] * 1e6 for _, h in valid], marker='_', color=color,
                s=35, alpha=.8, label=f'Hump {index+1} centre')
        pri_axes[0, col].set(title=f'{family}: PRI histogram inputs ({len(with_pri)}/{len(source_events)} blobs)',
                                  ylabel='Selected observed gap (µs)')
        pri_axes[0, col].legend(fontsize=8, loc='upper right')
        pri_axes[1, col].scatter(t, [100*h['used_gaps']/max(1,e['pulse_count']-1)
            for e,h in zip(with_pri,histogram)], s=12, color='tab:blue', alpha=.7)
        pri_axes[1, col].scatter([e['time_from_start_s'] for e in no_pri], [0]*len(no_pri),
                               marker='x', s=15, color='tab:red', label='PRI unavailable')
        pri_axes[1, col].set(ylabel='Selected gaps (%)', ylim=(-5, 105),
                           title='Fraction of successive gaps used by the estimator')
        pri_axes[1, col].legend(fontsize=8)
    for axes_array in [axes, pri_axes]:
        for panel in axes_array.flat:
            panel.set_xlim(0, result['duration_s'])
            panel.set_xlabel('Time from Config 101 start (s)')
            panel.grid(alpha=.2)
    revision = result['run_manifest']['git_revision'][:7]
    fig.suptitle(f'AOA + HDBSCAN → histogram PRI → Kalman tracker | code {revision}\n'
                 'One source emitter splits across many identities; truth labels are used for this evaluation plot', fontsize=14)
    pri_fig.suptitle(f'Mean, spread and hump information actually supplied to the tracker | code {revision}\n'
                     'Observed gaps within individual RF/PW blobs; underlying transmitter PRI is not shown', fontsize=14)
    for figure, name in [(fig, 'identity_fragmentation.png'), (pri_fig, 'pri_histogram_inputs.png')]:
        path = output / name
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(path)


if __name__ == '__main__':
    main()
