#!/usr/bin/env python3
from export_exp0261 import S,M,R,O,P,sha,write,preflight,read
import device_exp0257 as base
import exp0240_device as old
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
from pathlib import Path
import numpy as np
import subprocess,json,shlex,statistics,argparse,tarfile
R=R.parent/'exp0262'
base.R=R;base.REMOTE='/data/local/tmp/qwen3-block-htp/exp0262'
REMOTE='/data/local/tmp/qwen3-block-htp/exp0261-r4';PARENT='/data/local/tmp/qwen3-block-htp/exp0252-layer0/r3'
LAYER_FIELDS=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
def records(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.startswith('{')]
def deploy():
 preflight();a=read(R/'export_audit.json');assert a['pass_all'];p=O/'r4';m=read(p/'manifest.json');assert sha(p/'manifest.json')==a['manifest_sha256']
 pm=read(P/'manifest.json');out=base.adb('shell','cd '+PARENT+' && sha256sum '+' '.join(shlex.quote(n) for n in pm['files']))
 got={l.split(None,1)[1].strip():l.split()[0] for l in out.splitlines()};assert all(got[n]==v['sha256'] for n,v in pm['files'].items())
 for n,v in m['files'].items():assert sha(p/n)==v['sha256']
 changed=set(a['changed_files']);lines=['set -e'];archive=O/'payload.tar'
 with tarfile.open(archive,'x') as tar:
  for n in m['files']:
   lines.append('mkdir -p '+shlex.quote(str(Path(n).parent)))
   if n in changed:tar.add(p/n,arcname=n)
   else:lines.append('ln -s '+shlex.quote(PARENT+'/'+n)+' '+shlex.quote(n))
  tar.add(p/'manifest.json',arcname='manifest.json')
 checks=R/'device_files.sha256';checks.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in m['files'].items()))
 links=R/'device_links.sh';links.write_text('\n'.join(lines)+'\n')
 base.adb('shell','mkdir '+REMOTE)
 for f in [archive,checks,links]:base.adb('push',base.win(f),REMOTE+'/'+f.name)
 base.adb('shell',f'cd {REMOTE} && sh device_links.sh && tar -xf payload.tar && sha256sum -c device_files.sha256')
 write(R/'device_package.json',dict(remote=REMOTE,manifest_sha256=a['manifest_sha256'],parent_remote=PARENT,parent_files=len(pm['files']),files=len(m['files'])))
 print('DEPLOY_PASS',flush=True)
def physical(ps,arm,dump):
 for p in ps:
  assert p['block_invocation_count']==1 and p['vtcm_requested_bytes']==p['vtcm_acquired_bytes']==8388608
  assert p['intermediate_ddr_read_bytes']==p['intermediate_ddr_write_bytes']==p['intermediate_spill_fill_count']==p['ledger_unattributed_ticks']==0
  assert sum(normalized([p])[k] for _,k in LEDGER)==p['invocation_ticks']
  l=p['slice_layer_0'];assert l['status']==3 and l['layer_index']==0 and l['layer_unattributed_ticks']==0
  assert sum(l[k] for k in LAYER_FIELDS)==l['layer_ticks']
  assert l['cache_valid_after']==64+p['replay_step'] and l['cache_valid_before']==(0 if p['replay_step']==0 else 63+p['replay_step'])
  assert p['dense_r3_mode']==1 and p['dense_r3_optimization']==2 and p['wide_score_mode']==4
  assert p['dense_r3_total_calls']==p['dense_r3_total_hmx_calls']==1 and p['dense_r3_total_parallel_heads']==24
  assert p['dense_r4_mode']==arm and p['dense_r4_calls']==bool(arm) and p['dense_r4_rows']==bool(arm)*p['logical_m']
  assert p['dense_r4_hmx_calls']==(((9 if p['dense_r4_optimization']==2 and arm==1 else 5) if p['logical_m']==64 else 2) if arm else 0)
  if not dump:assert p['dense_r4_audit_bytes']==p['u8_attention_audit_ddr_write_bytes']==0 and arm!=2

def run(arm,rep,tag,dump=False,opt=2):
 preflight();root=base.runtime_root(1)
 e=dict(old.ENV,LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_DENSE_R3='1',QBH_R3_OPT='2',QBH_WIDE_SCORE='4',QBH_DENSE_R4=str(arm),QBH_R4_OPT=str(opt if arm else 0),QBH_REPLAY_DECODE_STEPS='8')
 if dump:
  e.update(QBH_REPLAY_DUMP_DIR=root+'/'+tag.replace('/','_'),QBH_DENSE_R3_AUDIT='1',QBH_DENSE_R4_AUDIT='1');
  if not (R/tag/'stdout.jsonl').exists():base.adb('shell','mkdir '+e['QBH_REPLAY_DUMP_DIR'])
 p=R/tag
 if not (p/'stdout.jsonl').exists():
  p=base.execute(1,e,PARENT if arm==0 else REMOTE,rep,tag)
  if dump:base.adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',base.win(p))
 else:assert not (p/'validated.json').exists() # retained successful RPC, failed parser only
 ps=[z for z in records(p/'stdout.jsonl') if z.get('record')=='exp0240_profile'];assert len(ps)==rep*9;physical(ps,arm,dump);assert all(z['dense_r4_optimization']==(opt if arm else 0) for z in ps)
 hashes=[z['output_hash'] for z in ps]
 gold=R/f'audit_a{arm}/validated.json'
 if gold.exists():assert hashes==read(gold)['output_hashes']*rep,'determinism failure'
 elif arm==0:
  oldgold=read(R.parent/'exp0259/audit_r3/validated.json');assert hashes==oldgold['output_hashes']*rep,'old R3 exactness'
 z=dict(pass_all=True,arm=arm,repeat=rep,output_hashes=hashes[:9],profiles=len(ps),prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode'))
 write(p/'validated.json',z);print('PASS',tag,round(z['prefill_ns']/1000,2),round(z['decode_ns']/1000,2),flush=True);return z

def profile(phase):
 preflight();assert read(R/'numerical_gate.json')['component_pass'] and read(R/'numerical_gate.json')['full_layer_ideal_R4_gate_pass']
 if phase=='formal':assert read(R/'short_gate.json')['integrity_pass']
 rows=[];n=5 if phase=='short' else 10
 for i in range(n):
  for rep in [1,10]:
   for arm in ([0,1] if i%2==0 else [1,0]):
    tag=f'{phase}/round{i+1:02d}_r{rep}_a{arm}';p=R/tag/'validated.json';z=read(p) if p.exists() else run(arm,rep,tag);rows.append(dict(round=i+1,**z))
 rng=np.random.default_rng(261);perf={}
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [0,1]] for i in range(n)])
   ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);perf[f'r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist(),stable_over10=bool(ci[0]>1.1))
 write(R/(phase+'_gate.json'),dict(integrity_pass=True,rows=rows,performance=perf));print(phase.upper()+'_COMPLETE',json.dumps(perf),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--arm',type=int,default=0);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',default='smoke');p.add_argument('--dump',action='store_true');p.add_argument('--opt',type=int,default=2);a=p.parse_args()
 if a.action=='stage':base.stage(1)
 elif a.action=='deploy':deploy()
 elif a.action in ['short','formal']:profile(a.action)
 else:run(a.arm,a.repeat,a.tag,a.dump,a.opt)
