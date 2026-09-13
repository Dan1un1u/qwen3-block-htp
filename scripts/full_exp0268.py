#!/usr/bin/env python3
import sys,json
import numpy as np
from common_exp0268 import S,R,O,P,OLD,sha,write,preflight
from device_exp0268 import run,read,records,physical

def full_gate():
 preflight();assert read(R/'slice_gate.json')['pass_all'];zs={}
 for m in [0,1,2,3,8]:zs[m]=run(m,1,f'full-m{m}',count=28)
 assert all(zs[m]['selected_codes']==zs[0]['selected_codes'] for m in [1,2,3])
 for m in [0,8]:assert zs[m]['selected_codes']==read(OLD/f'full-m{m}/validated.json')['selected_codes']
 write(R/'full_gate.json',dict(pass_all=True,layers=28,selected_codes={str(m):z['selected_codes'] for m,z in zs.items()},scope='all U8 variants same feedback token IDs and selected logit codes; SP2 matches sealed SP2, not full logit equality or PPL'))

def full_profile(phase):
 preflight();assert read(R/'full_gate.json')['pass_all']
 if phase=='formal':assert read(R/'full-short.json')['pass_all']
 n=5 if phase=='short' else 10;rows=[];arms=[0,1,2,3,8]
 for i in range(n):
  for repeat in [1,10]:
   order=arms[i%5:]+arms[:i%5]
   for m in order:
    tag=f'full-{phase}/round{i:02d}-r{repeat}-m{m}';saved=R/tag/'validated.json';z=read(saved) if saved.exists() else run(m,repeat,tag,count=28);assert z['selected_codes']==read(R/f'full-m{m}/validated.json')['selected_codes'];rows.append(dict(round=i,**z))
 rng=np.random.default_rng(268);idx=rng.integers(0,n,(20000,n));perf={}
 for rep in [1,10]:
  for key in ['prefill_ns','decode_ns']:
   for a,b in [(0,1),(0,2),(0,3),(1,2),(2,3),(0,8),(3,8)]:
    pair=np.array([[next(z[key] for z in rows if z['round']==i and z['repeat']==rep and z['mode']==m) for m in [a,b]] for i in range(n)]);ci=np.quantile(pair[idx,1].mean(1)/pair[idx,0].mean(1),[.025,.975]);perf[f'r{rep}_m{b}_over_m{a}_{key}']=dict(control_mean_ns=float(pair[:,0].mean()),candidate_mean_ns=float(pair[:,1].mean()),ratio=float(pair[:,1].mean()/pair[:,0].mean()),ci95=ci.tolist(),gate=bool(ci[1]<=1.1),auxiliary_only=rep==1)
 write(R/f'full-{phase}.json',dict(pass_all=True,rows=rows,performance=perf,speed_pass=all(perf[f'r10_m3_over_m0_{k}']['gate'] for k in ['prefill_ns','decode_ns'])));print('FULL_PROFILE_DONE',phase,json.dumps(perf),flush=True)
if __name__=='__main__':
 a=sys.argv[1]
 if a=='gate':full_gate()
 else:full_profile(a)
