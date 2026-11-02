"""Audit PRI candidate diagnostics without changing tracker or association settings."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('run_result')
    parser.add_argument('--family', default='E38')
    parser.add_argument('--output', default='results/consistency/e38_pri_audit')
    args = parser.parse_args()
    run = json.loads(Path(args.run_result).read_text())
    track_id = run['track_ids'][args.family]
    truth_id = run['families'][args.family]['truth_id']
    events = {e['observation_id']: e for e in run['events']}
    records = []
    for line in Path(args.log).read_text().splitlines():
        row = json.loads(line)
        event = events.get(row['observation_id'])
        if (row['prior_hypothesis_rank'] != 0 or row['track_id'] != track_id
                or not row['timing'] or not event or event['source_family'] != args.family):
            continue
        timing, axes, measurement = row['timing'], row['axes'], event['measurement']
        oh, th = timing['observed_harmonic'], timing['track_harmonic']
        observed = measurement['pri_s']
        predicted = (observed / oh - timing['innovation_s']) * th
        truth = {int(k): v for k, v in event['truth_counts_eval'].items()}
        records.append({
            'observation_id': row['observation_id'], 'window': event['window'],
            'time_s': event['time_from_start_s'], 'outcome': row['outcome'],
            'pulse_count': event['pulse_count'], 'observed_pri_us': observed * 1e6,
            'predicted_pri_us': predicted * 1e6,
            'observed_harmonic': oh, 'track_harmonic': th,
            'pri_sigma_us': measurement['pri_sigma_s'] * 1e6,
            'timing_rms_us': measurement['timing_rms_residual_s'] * 1e6,
            'innovation_base_us': timing['innovation_s'] * 1e6,
            'innovation_sigma_base_us': np.sqrt(axes['pri_base_s']['S']) * 1e6,
            'aoa_nis': axes['aoa_deg']['innovation'] ** 2 / axes['aoa_deg']['S'],
            'pw_nis': axes['pw_s']['innovation'] ** 2 / axes['pw_s']['S'],
            'pri_nis': timing['pri_z'] ** 2, 'joint_nis': row['nis'],
            'gate_threshold': row['gate_threshold'],
            'truth_purity_eval': truth.get(truth_id, 0) / sum(truth.values()) if truth else None})
    if not records:
        parser.error('No matching PRI candidates')
    report = {'family': args.family, 'track_id': track_id,
              'sample': 'Incoming rank-zero, same-family PRI candidates, before gate; seed excluded',
              'limitations': 'Generated update branches can be pruned; rank zero may change lineage. Truth purity is evaluation only. Fixed-train fitting uncertainty excludes model/harmonic ambiguity. No Q/R or gate changes.',
              'by_outcome': {}}
    for outcome in sorted({r['outcome'] for r in records}):
        subset = [r for r in records if r['outcome'] == outcome]
        report['by_outcome'][outcome] = {
            'count': len(subset),
            'pulse_count_min_median_max': list(map(float, np.percentile([r['pulse_count'] for r in subset], [0, 50, 100]))),
            'observed_pri_min_median_max_us': list(map(float, np.percentile([r['observed_pri_us'] for r in subset], [0, 50, 100]))),
            'fraction_pri_sigma_at_export_floor': float(np.mean([np.isclose(r['pri_sigma_us'], .05) for r in subset])),
            'mean_pri_nis': float(np.mean([r['pri_nis'] for r in subset])),
            'fraction_joint_nis_from_pri': sum(r['pri_nis'] for r in subset) / sum(r['joint_nis'] for r in subset),
            'pure_truth_summaries_eval': sum(r['truth_purity_eval'] == 1 for r in subset),
            'harmonic_pairs_observed_track': dict(Counter(f"{r['observed_harmonic']}:{r['track_harmonic']}" for r in subset))}
    report['records'] = records
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for outcome, color, marker in [('accepted', 'tab:blue', 'o'), ('rejected', 'tab:red', 'x')]:
        subset = [r for r in records if r['outcome'] == outcome]
        for panel, x, y in [(ax[0, 0], 'time_s', 'observed_pri_us'),
                            (ax[0, 1], 'pulse_count', 'observed_pri_us'),
                            (ax[1, 0], 'time_s', 'pri_nis'),
                            (ax[1, 1], 'timing_rms_us', 'pri_sigma_us')]:
            panel.scatter([r[x] for r in subset], [r[y] for r in subset],
                          color=color, marker=marker, s=28, label=f'{outcome} branch')
    ax[0, 0].plot([r['time_s'] for r in records], [r['predicted_pri_us'] for r in records],
                  '.', color='black', ms=3, label='Incoming track PRI')
    ax[0, 0].set(xlabel='Recording time (s)', ylabel='PRI (µs)', title='Train estimates and incoming track prediction')
    ax[0, 1].set(xlabel='Pulses in summary', ylabel='Reported train PRI (µs)', title='Sparse summaries produce longer candidate periods')
    ax[1, 0].set(xlabel='Recording time (s)', ylabel='PRI contribution to NIS', title='PRI contribution before gating', yscale='log')
    ax[1, 1].set(xlabel='Train timing RMS residual (µs)', ylabel='Exported PRI sigma (µs)', title='Conditional fit uncertainty versus residual size')
    for panel in ax.flat:
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
    fig.suptitle(f'{args.family} PRI audit — incoming rank-zero candidates; truth used only for evaluation')
    fig.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
