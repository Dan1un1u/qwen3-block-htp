#!/usr/bin/env python3
"""Conditional EXP0259 seeded slice/fullmodel validation and true E2E."""
import device_exp0258 as prev
from device_exp0259 import R,read,write,records,preflight,CFG,sha,S,M
import numpy as np,json,statistics,argparse
P=R.parent/'exp0258'
prev.R=R;prev.base.R=R;prev.base.REMOTE='/data/local/tmp/qwen3-block-htp/exp0259'
# Parent package/prefix paths remain frozen EXP0258/EXP0257. Only binaries differ.
parent_env=prev.env
parent_physical=prev.physical
def env(count,arm):
 root,e=parent_env(count,arm);e['QBH_R3_OPT']='2' if arm else '0';return root,e
def physical(ps,count,arm):
 parent_physical(ps,count,arm)
 for q in ps:
  assert q['dense_r3_optimization']==2*arm
  assert q['dense_r3_total_parallel_heads']==count*24*arm
  assert q['dense_r3_constant_read_bytes']==count*32768*arm
prev.env=env;prev.physical=physical

def eligible():
 g=read(R/'confirmation_gate.json');assert g['integrity_pass'] and g['fullmodel_escalation_allowed']

def run(arm,rep,tag,count=28,dump=False):
 eligible();z=prev.run(arm,rep,tag,count,dump)
 if count==28:assert z['token_ids']==read(P/f'smoke_a{arm}/validated.json')['token_ids']
 return z

def slice_gate():
 preflight();eligible();checks=[]
 for arm,label in [(0,'slice_off'),(1,'slice_stream'),(1,'slice_stream_repeat')]:
  for step in range(9):
   for suffix in ['output','r3']:
    a=R/f'{label}/step{step:02}_{suffix}.bin';b=P/f'slice_a{arm}/step{step:02}_{suffix}.bin'
    assert a.read_bytes()==b.read_bytes(),str(a);checks.append(str(a.relative_to(R)))
 write(R/'slice_gate.json',dict(pass_all=True,exact_files=len(checks),against_sealed_parent=str(P),native_numerical_failure_unchanged=True))
 print('SLICE_GATE_PASS',len(checks),flush=True)

def profile(phase):
 preflight();eligible();assert read(R/'slice_gate.json')['pass_all']
 for arm in [0,1]:assert read(R/f'smoke_a{arm}/validated.json')['token_ids']==read(P/f'smoke_a{arm}/validated.json')['token_ids']
 if phase=='formal':assert read(R/'full_short_gate.json')['integrity_pass']
 rows=[];n=5 if phase=='short' else 10
 for i in range(n):
  for rep in [1,10]:
   for arm in ([0,1] if i%2==0 else [1,0]):
    tag=f'full_{phase}/round{i+1:02d}_r{rep}_a{arm}';p=R/tag/'validated.json';z=read(p) if p.exists() else run(arm,rep,tag);rows.append(dict(round=i+1,**z))
 rng=np.random.default_rng(259);perf={}
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['repeat']==rep and r['arm']==a) for a in [0,1]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975])
   perf[f'r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),R3_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist(),stable_over10percent=bool(ci[0]>1.1))
 write(R/f'full_{phase}_gate.json',dict(integrity_pass=True,performance=perf,rounds=rows,speed_eligible=all(p['ci95'][1]<=1.1 for p in perf.values()),numerical_eligible=False))
 print('FULL_'+phase.upper()+'_COMPLETE',json.dumps(perf),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=28);p.add_argument('--arm',type=int,default=1);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',default='smoke');p.add_argument('--dump',action='store_true');a=p.parse_args()
 if a.action=='stage':eligible();prev.base.stage(a.count)
 elif a.action=='slice_gate':slice_gate()
 elif a.action in ['short','formal']:profile(a.action)
 else:run(a.arm,a.repeat,a.tag,a.count,a.dump)
