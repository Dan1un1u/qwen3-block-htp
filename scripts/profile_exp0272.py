"""Fixed full-model paired campaign: repeat10 primary, repeat1 auxiliary."""
from full_exp0272 import *
import numpy as np

def profile(phase):
 preflight();assert read(R/'full_sanity_gate.json')['pass_all']
 if phase=='formal':assert read(R/'full-short.json')['speed_pass']
 n=5 if phase=='short' else 10;rows=[]
 orders=[[0,1,2],[2,1,0],[1,2,0],[0,2,1],[2,0,1],[1,0,2]]
 for i in range(n):
  for rep in [1,10]:
   for mode in orders[i%len(orders)]:
    tag=f'full-{phase}/round{i:02d}-r{rep}-fp{mode}';f=R/tag/'validated.json'
    z=read(f) if f.exists() else full(mode,rep,tag)
    gold=read(R.parent/('exp0268/full-m8/validated.json' if mode==0 else 'exp0271/timing-r1-fp1/validated.json'))['selected_codes'];assert z['selected_codes']==gold
    rows.append(dict(round=i,**z))
 rng=np.random.default_rng(272);idx=rng.integers(0,n,(20000,n));perf={}
 for rep in [1,10]:
  for key in ['prefill_ns','decode_ns']:
   for den in [0,1]:
    pair=np.array([[next(z[key] for z in rows if z['round']==i and z['repeat']==rep and z['fp32']==m) for m in [den,2]] for i in range(n)]);ci=np.quantile(pair[idx,1].mean(1)/pair[idx,0].mean(1),[.025,.975]);perf[f'r{rep}_{key}_vs{den}']=dict(control_mean_ns=float(pair[:,0].mean()),candidate_mean_ns=float(pair[:,1].mean()),ratio=float(pair[:,1].mean()/pair[:,0].mean()),ci95=ci.tolist(),gate=bool(ci[1]<=1.1) if den==0 else None,auxiliary_only=rep==1)
 out=dict(phase=phase,rounds=n,rows=rows,performance=perf,speed_pass=all(v['gate'] for k,v in perf.items() if k.startswith('r10_') and k.endswith('vs0')))
 write(R/f'full-{phase}.json',out);print('PHASE_DONE',phase,out['speed_pass'],json.dumps(perf),flush=True)
if __name__=='__main__':
 import sys
 profile(sys.argv[1])
