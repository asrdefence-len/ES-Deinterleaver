"""Measure fixed-PRI discovery responses to uniform random timing in a frozen blob."""
import argparse
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import h5py,numpy as np
from compare_aoa_batch import batch_aoa_populations
from compare_multiscale_rf import group_window
from testbed.pulse_association_pipeline import FixedPRIPulseAssociation,PulseMeasurement
from testbed.blob_observation import summarize_blob_trains
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('file')
parser.add_argument('--window',type=int,default=10)
parser.add_argument('--blob',type=int,default=7)
parser.add_argument('--repetitions',type=int,default=20)
parser.add_argument('--output',default='results/consistency/e38_pri_null_control.json')
args=parser.parse_args()
if args.repetitions < 1 or args.window < 0:
 parser.error('Require positive repetitions and nonnegative window')
p=args.file
with h5py.File(p) as h:
 names=[v.decode() if isinstance(v,bytes) else str(v) for v in h['metadata/feature_names'][:]]
 data=np.asarray(h['data'][args.window*10000:(args.window+1)*10000],float)
pop,n=batch_aoa_populations(data[:,names.index('AoA')]);labels=group_window(data,names,pop,n,8,.20,6.,8)
x=data[labels==args.blob].copy()
if len(x)<6:
 parser.error('Selected blob has fewer than six pulses')
rng=np.random.default_rng(10138)
a=FixedPRIPulseAssociation(tolerance_s=5. * 1e-6,min_pulses=6)
results=[]
for iteration in range(args.repetitions+1):
 sample=x.copy()
 if iteration:
  col=names.index('ToA');sample[:,col]=np.sort(rng.uniform(x[:,col].min(),x[:,col].max(),len(x)))
 pdws=tuple(PulseMeasurement(i,float(row[names.index('ToA')]*1e-6),float(row[names.index('Frequency')]*1e6),float(row[names.index('PulseWidth')]*1e-6),float(row[names.index('AoA')])) for i,row in enumerate(sample))
 trains,unexplained,_=a.process(pdws,np.zeros(len(sample),int),f'N{iteration}')
 summaries=summarize_blob_trains(sample,names,np.zeros(len(sample),int),np.arange(len(sample)),f'N{iteration}',trains)
 results.append({'kind':'measured' if iteration==0 else 'uniform_random_timing','iteration':iteration,'input_pulses':len(sample),'explained_fraction':1-len(unexplained)/len(sample),'summaries':[{'pri_us':o.pri_s*1e6 if o.pri_s else None,'sigma_us':o.pri_sigma_s*1e6 if o.pri_sigma_s else None,'pulse_count':o.pulse_count,'rms_us':o.timing_rms_residual_s*1e6 if o.timing_rms_residual_s else None} for o in summaries]})
output=Path(args.output)
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'seed':10138,'window':args.window,'blob':args.blob,'description':'Same frozen measured blob marks/count/span; replace TOAs with independent uniform times. Fixed-PRI pipeline unchanged. This tests specificity, not the true emitter PRI.','runs':results},indent=2)+'\n')
print(json.dumps(results[0],indent=2))
random=results[1:]
print('null explained min/median/max',np.percentile([r['explained_fraction'] for r in random],[0,50,100]))
print('null leading PRI min/median/max',np.percentile([r['summaries'][0]['pri_us'] for r in random],[0,50,100]))
print('null exported sigma min/median/max',np.percentile([s['sigma_us'] for r in random for s in r['summaries']],[0,50,100]))
