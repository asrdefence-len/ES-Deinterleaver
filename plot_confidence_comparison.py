"""Restore recorded primary-track existence plots without hiding missing tracks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('earlier')
    parser.add_argument('current')
    parser.add_argument('--output',default='results/integrated_baseline/confidence_comparison.png')
    args=parser.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for panel,path,label in zip(axes,[args.earlier,args.current],
            ['Earlier saved run: multiscale RF + fixed-train PRI',
             'Current experimental integration: HDBSCAN + gap-histogram PRI']):
        run=json.loads(Path(path).read_text())
        rows=run['monitor']
        for name,color in [('E46','tab:blue'),('E38','tab:orange')]:
            times=[r['time_from_start_s'] for r in rows]
            values=[100*r['tracks'][name]['track_confidence'] if name in r['tracks'] else np.nan for r in rows]
            panel.plot(times,values,color=color,lw=2,label=f'{name} primary {run["track_ids"][name]}')
            present=[t for t,v in zip(times,values) if np.isfinite(v)]
            if present:
                panel.annotate(f'{name}: last present {max(present):.2f}s',
                    xy=(.03,.08 if name=='E46' else .02),xycoords='axes fraction',
                    color=color,fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        panel.axhline(80,color='tab:green',ls='--',lw=.9,label='Confirmed threshold')
        panel.axhline(30,color='tab:red',ls='--',lw=.9,label='Lost threshold')
        panel.set(title=label,xlabel='Time from Config 101 start (s)',ylabel='Existence probability (%)',
                  xlim=(0,run['duration_s']),ylim=(-2,105))
        panel.grid(alpha=.2)
        panel.legend(loc='center right',fontsize=8)
    fig.suptitle('Recorded primary-track confidence — different processing paths\n'
                 'Missing lines mean the primary track is absent; these nominal probabilities are not calibrated.',fontsize=13)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)
    print(output)


if __name__=='__main__':
    main()
