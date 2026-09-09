#!/usr/bin/env python3
"""EXP0259 frozen layer0 exact-R3 implementation ablations."""
from export_exp0257 import S,M,sha,write,preflight
from pathlib import Path
import json,subprocess,shlex,statistics,argparse
import numpy as np
import device_exp0257 as base
import exp0240_device as old
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0259')
base.R=R;base.REMOTE='/data/local/tmp/qwen3-block-htp/exp0259'
ARMS=['off','r3','vector','stream']
CFG={'off':(0,0),'r3':(1,0),'vector':(1,1),'stream':(1,2)}
PARENT='/data/local/tmp/qwen3-block-htp/exp0252-layer0'
def read(p):return json.loads(Path(p).read_text())
def records(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.startswith('{')]
def verify_packages():
 preflight();audit=read(R.parent/'exp0247/export_audit.json');result={}
 for cell in ['control','r3']:
  p=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0247')/cell;m=read(p/'manifest.json');assert sha(p/'manifest.json')==audit['manifests'][cell]['manifest_sha256']
  for n,v in m['files'].items():assert sha(p/n)==v['sha256']
  out=base.adb('shell','cd '+PARENT+'/'+cell+' && sha256sum '+' '.join(shlex.quote(n) for n in m['files']))
  got={line.split(None,1)[1].strip():line.split()[0] for line in out.splitlines()}
  assert all(got[n]==v['sha256'] for n,v in m['files'].items())
  result[cell]=dict(manifest_sha256=sha(p/'manifest.json'),files=m['files'])
 write(R/'package_provenance.json',result)
def physical(ps,arm,dump):
 mode,opt=CFG[arm]
 for p in ps:
  assert p['block_invocation_count']==1 and p['slice_layer_0']['layer_index']==0
  assert p['dense_r3_mode']==mode and p['dense_r3_optimization']==opt and p['wide_score_mode']==4
  assert p['dense_r3_total_calls']==p['dense_r3_total_hmx_calls']==mode
  assert p['dense_r3_total_rows']==mode*24*p['logical_m'] and p['dense_r3_total_refined_values']==0
  assert p['dense_r3_total_parallel_heads']==(24 if opt==2 else 0)
  assert p['dense_r3_constant_read_bytes']==(32768 if opt else 0)
  assert p['vtcm_requested_bytes']==p['vtcm_acquired_bytes']==8388608
  assert p['intermediate_ddr_read_bytes']==p['intermediate_ddr_write_bytes']==p['intermediate_spill_fill_count']==0
  assert p['ledger_unattributed_ticks']==0 and sum(normalized([p])[k] for _,k in LEDGER)==p['invocation_ticks']
  if not dump:assert p['u8_attention_audit_ddr_write_bytes']==0
  assert p['scan_total_kv_length']==64+p['replay_step']
def run(arm,rep,tag,dump=False):
 preflight();root=base.runtime_root(1);mode,opt=CFG[arm]
 e=dict(old.ENV,LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_DENSE_R3=str(mode),QBH_R3_OPT=str(opt),QBH_WIDE_SCORE='4',QBH_REPLAY_DECODE_STEPS='8')
 if dump:
  e.update(QBH_REPLAY_DUMP_DIR=root+'/'+tag.replace('/','_'),QBH_DENSE_R3_AUDIT='1');base.adb('shell','mkdir '+e['QBH_REPLAY_DUMP_DIR'])
 p=base.execute(1,e,PARENT+('/control' if arm=='off' else '/r3'),rep,tag)
 if dump:base.adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',base.win(p))
 ps=[q for q in records(p/'stdout.jsonl') if q.get('record')=='exp0240_profile'];assert len(ps)==rep*9;physical(ps,arm,dump)
 hashes=[q['output_hash'] for q in ps]
 golden=R/f'audit_{"off" if arm=="off" else "r3"}/validated.json'
 if golden.exists():assert hashes==read(golden)['output_hashes']*rep,'output determinism/exactness'
 z=dict(arm=arm,repeat=rep,pass_all=True,profiles=len(ps),output_hashes=hashes[:9],prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode'))
 write(p/'validated.json',z);print('PASS',tag,round(z['prefill_ns']/1000,2),round(z['decode_ns']/1000,2),flush=True);return z

def profile(phase):
 preflight();assert read(R/'numerical_gate.json')['pass_all']
 if phase=='formal':assert read(R/'short_gate.json')['integrity_pass']
 rows=[];n=5 if phase=='short' else 10
 for i in range(n):
  order=ARMS[i%4:]+ARMS[:i%4]
  for rep in [1,10]:
   for arm in order:
    tag=f'{phase}/round{i+1:02d}_r{rep}_{arm}';p=R/tag/'validated.json';z=read(p) if p.exists() else run(arm,rep,tag);rows.append(dict(round=i+1,**z))
 rng=np.random.default_rng(259);perf={}
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   for arm in ARMS[1:]:
    for control in ['off']+(['r3'] if arm!='r3' else []):
     pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [control,arm]] for i in range(n)])
     ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);perf[f'{arm}_vs_{control}_r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist())
 write(R/(phase+'_gate.json'),dict(integrity_pass=True,rows=rows,performance=perf));print(phase.upper()+'_COMPLETE',json.dumps(perf),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--arm',choices=ARMS,default='r3');p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',default='smoke');p.add_argument('--dump',action='store_true');a=p.parse_args()
 if a.action=='stage':base.stage(1)
 elif a.action=='verify':verify_packages()
 elif a.action in ['short','formal']:profile(a.action)
 else:run(a.arm,a.repeat,a.tag,a.dump)
