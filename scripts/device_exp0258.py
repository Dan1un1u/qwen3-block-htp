#!/usr/bin/env python3
"""EXP0258 isolated paired native denseR3 fullmodel diagnostic."""
from export_exp0258 import *
import device_exp0257 as base
import exp0240_device as old
import subprocess,shlex,struct,tarfile,argparse,statistics
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
REMOTE='/data/local/tmp/qwen3-block-htp/exp0258'
base.R=R;base.O=O;base.REMOTE=REMOTE
adb=base.adb;win=base.win
PARENT_REMOTE='/data/local/tmp/qwen3-block-htp/exp0257-package-v2'
CONTROL_SEED='/data/local/tmp/qwen3-block-htp/exp0257-prefix/prefix_kv_u8.bin'
LAYER_FIELDS=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
def records(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.startswith('{')]
def env(count,arm):
 root=base.runtime_root(count);e=dict(old.ENV);e.update(QBH_WIDE_SCORE='4',QBH_DENSE_R3=str(arm),QBH_KV_CACHE_CAPACITY='128',LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_PREFIX_KV='1',QBH_PREFIX_FILE=CONTROL_SEED if arm==0 else REMOTE+'-package/prefix_kv_u8.bin');return root,e
def package(arm):return PARENT_REMOTE if arm==0 else REMOTE+'-package'
def deploy():
 preflight();audit=read(R/'export_audit.json');assert audit['pass_all'];m=read(O/'r3/manifest.json');assert sha(O/'r3/manifest.json')==audit['manifest_sha256']
 check=adb('shell',f'cd {PARENT_REMOTE} && sha256sum -c files.sha256',timeout=600);assert check.count(': OK')==906 and 'FAILED' not in check
 assert read(R.parent/'exp0257/prefix_device.json')['sha256'] in adb('shell','sha256sum '+CONTROL_SEED)
 root=REMOTE+'-package';adb('shell',f'mkdir {root}');lines=['set -e'];changed=set(audit['changed_files'])
 for n in m['files']:
  lines.append('mkdir -p '+shlex.quote(str(Path(n).parent)))
  if n not in changed:lines.append('ln -s '+shlex.quote(PARENT_REMOTE+'/'+n)+' '+shlex.quote(n))
 script=R/'device_links.sh';script.write_text('\n'.join(lines)+'\n');adb('push',win(script),root+'/links.sh');adb('shell',f'cd {root} && sh links.sh')
 checks=R/'files.sha256';checks.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in m['files'].items()));archive=O/'delta.tar'
 with tarfile.open(archive,'x',dereference=True) as tar:
  for n in sorted(changed):tar.add(O/'r3'/n,arcname=n)
  tar.add(O/'r3/manifest.json',arcname='manifest.json');tar.add(O/'prefix/prefix_kv_u8.bin',arcname='prefix_kv_u8.bin');tar.add(checks,arcname='files.sha256')
 adb('push',win(archive),root+'/delta.tar');adb('shell',f'cd {root} && tar -xf delta.tar')
 check=adb('shell',f'cd {root} && sha256sum -c files.sha256',timeout=600);assert check.count(': OK')==906 and 'FAILED' not in check;assert audit['seed_sha256'] in adb('shell','sha256sum '+root+'/prefix_kv_u8.bin')
 write(R/'device_package.json',dict(remote=root,manifest_sha256=audit['manifest_sha256'],verified_files=906,seed_sha256=audit['seed_sha256'],control_remote=PARENT_REMOTE,control_verified_files=906));print('DEPLOY_PASS',flush=True)
def physical(ps,count,arm):
 for q in ps:
  assert q['block_invocation_count']==count and q['vtcm_acquired_bytes']==q['vtcm_requested_bytes']==8388608
  assert q['boundary_ddr_write_bytes']==q['intermediate_ddr_read_bytes']==q['intermediate_ddr_write_bytes']==q['intermediate_spill_fill_count']==q['ledger_unattributed_ticks']==0
  assert q['dense_r3_mode']==arm and q['wide_score_mode']==4 and q['prefix_kv_mode']==1
  assert q['dense_r3_total_calls']==q['dense_r3_total_hmx_calls']==arm*count
  assert q['dense_r3_total_rows']==arm*count*24*q['logical_m'] and q['dense_r3_total_refined_values']==0
  assert q['prefix_group_patch_count']==(count*8 if q['mode']=='prefill' else 0)
  assert sum(normalized([q])[k] for _,k in LEDGER)==q['invocation_ticks']
  step=q.get('generation_step',q.get('replay_step'));assert q['scan_total_kv_length']==64+step
  for j in range(count):
   l=q[f'slice_layer_{j}'];assert l['status']==3 and l['layer_index']==j and l['cache_valid_after']==64+step
   assert l['cache_valid_before']==(0 if step==0 else 63+step)
   assert l['layer_unattributed_ticks']==l['hidden_ddr_read_bytes']==l['hidden_ddr_write_bytes']==0 and sum(l[k] for k in LAYER_FIELDS)==l['layer_ticks']
def run(arm,repeat,tag,count=28,dump=False):
 preflight();root,e=env(count,arm)
 if count==28:
  e.update(QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS='16')
  ids=read(R/'prompts.json')['samples'][0]['token_ids'];row=[0,2,16]+ids+[0]*16;file=R/(tag.replace('/','_')+'.bin');file.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)));remote=root+'/'+file.name;adb('push',win(file),remote);e['QBH_EVAL_FILE']=remote
 if dump:
  e.update(QBH_REPLAY_DUMP_DIR=root+'/'+tag.replace('/','_'),QBH_DENSE_R3_AUDIT='1');adb('shell','mkdir '+e['QBH_REPLAY_DUMP_DIR'])
 path=base.execute(count,e,package(arm),1,tag)
 if dump:adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',win(path))
 rs=records(path/'stdout.jsonl');ps=[x for x in rs if x.get('record')==('generation_profile' if count==28 else 'exp0240_profile')]
 assert len(ps)==(16*repeat if count==28 else 9);physical(ps,count,arm)
 fs=[x for x in rs if x.get('generation_sequence_complete')]
 if count==28:
  assert len(fs)==repeat and all(f['all_steps_pass'] for f in fs)
  golden=R/f'smoke_a{arm}/validated.json'
  if golden.exists():gold=read(golden)['token_ids'];assert all(f['token_ids']==gold for f in fs),'new token determinism failure'
  elif arm==0:assert fs[0]['token_ids']==read(R.parent/'exp0257/seed_full_nr64/validated.json')['token_ids'][:16]
  assert all(f['token_ids']==fs[0]['token_ids'] for f in fs)
 result=dict(pass_all=True,arm=arm,repeat=repeat,layers=count,profiles=len(ps),prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode')*(15 if count==28 else 8),token_ids=fs[0]['token_ids'] if fs else None,numerical_ideal_R3_gate='known_fail_waived_for_diagnostic_speed' if arm else 'parent_bounded_checks_pass')
 write(path/'validated.json',result);print('PASS',tag,round(result['prefill_ns']/1e6,3),round(result['decode_ns']/(15e6 if count==28 else 8e6),3),flush=True);return result

def profile(phase):
 preflight();assert read(R/'slice_gate.json')['pass_all']
 for arm in [0,1]:assert read(R/f'smoke_a{arm}/validated.json')['pass_all']
 if phase=='formal':assert read(R/'short_gate.json')['integrity_pass']
 rows=[]
 for i in range(5 if phase=='short' else 10):
  for rep in [1,10]:
   for arm in ([0,1] if i%2==0 else [1,0]):
    tag=f'{phase}/round{i+1:02d}_r{rep}_a{arm}';p=R/tag/'validated.json';z=read(p) if p.exists() else run(arm,rep,tag);rows.append(dict(round=i+1,**z))
 rng=np.random.default_rng(258);perf={}
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['repeat']==rep and r['arm']==a) for a in [0,1]] for i in range(5 if phase=='short' else 10)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,len(pairs),(10000,len(pairs)))] ,axis=1),[.025,.975]);perf[f'r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),R3_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist(),stable_over10percent=bool(ci[0]>1.1))
 write(R/(phase+'_gate.json'),dict(integrity_pass=True,performance=perf,rounds=rows,numerical_eligible=False,speed_eligible=not any(p['stable_over10percent'] for p in perf.values()),scope='userapproved performance-only exception'))
 print(phase.upper()+'_COMPLETE',json.dumps(perf),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--arm',type=int,default=0);p.add_argument('--count',type=int,default=28);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',default='smoke');p.add_argument('--dump',action='store_true');a=p.parse_args()
 if a.action=='deploy':deploy()
 elif a.action=='stage':base.stage(a.count)
 elif a.action in ['short','formal']:profile(a.action)
 else:run(a.arm,a.repeat,a.tag,a.count,a.dump)
