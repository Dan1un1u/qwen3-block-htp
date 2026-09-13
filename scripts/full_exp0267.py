#!/usr/bin/env python3
"""Conditional consecutive-layer and fixed full-model SP2 campaign."""
import sys,json,statistics
import numpy as np
from export_exp0267 import S,R,O,P,sha,write,preflight
from device_exp0267 import run,read,records,physical
from audit_exp0267 import oracle,unpack_u8_hmx_activation

def slice_gate():
 preflight();assert read(R/'layer-formal.json')['speed_pass'];zs={}
 for m in [0,4,5,8]:
  tag=f'slice-a03-m{m}';saved=R/tag/'validated.json';zs[m]=read(saved) if saved.exists() else run(m,1,tag,count=3,dump=True)
  if m:oracle(tag,2,O/'sp2/layer2')
 assert zs[4]['output_hashes']==zs[5]['output_hashes']==zs[8]['output_hashes']
 for m in [5,8]:
  for f in (R/'slice-a03-m4').glob('*.bin'):
   b=R/f'slice-a03-m{m}'/f.name
   if f.name.endswith('_r3.bin') and not f.name.startswith('step00_'):
    a=np.fromfile(f,'u1');v=np.fromfile(b,'u1');off=983040+8*393216
    assert np.array_equal(a[:off],v[:off]) and np.array_equal(a[off+393216:],v[off+393216:])
    assert np.array_equal(unpack_u8_hmx_activation(a[off:off+393216],6144)[:8],unpack_u8_hmx_activation(v[off:off+393216],6144)[:8])
   else:assert sha(f)==sha(b),f.name
 z=run(8,10,'slice-a03-repeat-m8',count=3);assert z['output_hashes']==zs[8]['output_hashes']*10
 write(R/'slice_gate.json',dict(pass_all=True,layers=3,steps=9,serial_optimized_exact=True,repeat10_exact=True,last_layer_integer_oracle=True))

def full_gate():
 preflight();assert read(R/'slice_gate.json')['pass_all'];zs={}
 for m in [0,4,5,8]:zs[m]=run(m,1,f'full-m{m}',count=28)
 assert zs[4]['selected_codes']==zs[5]['selected_codes']==zs[8]['selected_codes']
 write(R/'full_gate.json',dict(pass_all=True,layers=28,selected_codes={str(m):z['selected_codes'] for m,z in zs.items()},scope='all SP2 variants same feedback token IDs and selected logit codes, not full logit equality or PPL'))

def full_profile(phase):
 preflight();assert read(R/'full_gate.json')['pass_all']
 if phase=='formal':assert read(R/'full-short.json')['pass_all']
 n=5 if phase=='short' else 10;rows=[];arms=[0,4,5,8]
 for i in range(n):
  for repeat in [1,10]:
   order=arms[i%4:]+arms[:i%4]
   for m in order:
    z=run(m,repeat,f'full-{phase}/round{i:02d}-r{repeat}-m{m}',count=28);assert z['selected_codes']==read(R/f'full-m{m}/validated.json')['selected_codes'];rows.append(dict(round=i,**z))
 rng=np.random.default_rng(267);idx=rng.integers(0,n,(20000,n));perf={}
 for rep in [1,10]:
  for key in ['prefill_ns','decode_ns']:
   for a,b in [(0,4),(0,5),(0,8),(4,5),(5,8),(4,8)]:
    pair=np.array([[next(z[key] for z in rows if z['round']==i and z['repeat']==rep and z['mode']==m) for m in [a,b]] for i in range(n)]);ci=np.quantile(pair[idx,1].mean(1)/pair[idx,0].mean(1),[.025,.975]);perf[f'r{rep}_m{b}_over_m{a}_{key}']=dict(control_mean_ns=float(pair[:,0].mean()),candidate_mean_ns=float(pair[:,1].mean()),ratio=float(pair[:,1].mean()/pair[:,0].mean()),ci95=ci.tolist(),gate=bool(ci[1]<=1.1),auxiliary_only=rep==1)
 write(R/f'full-{phase}.json',dict(pass_all=True,rows=rows,performance=perf,speed_pass=all(perf[f'r10_m8_over_m0_{k}']['gate'] for k in ['prefill_ns','decode_ns'])));print('FULL_PROFILE_DONE',phase,json.dumps(perf),flush=True)
if __name__=='__main__':
 a=sys.argv[1]
 if a=='slice':slice_gate()
 elif a=='gate':full_gate()
 else:full_profile(a)
