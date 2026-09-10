#!/usr/bin/env python3
"""EXP0265 native slice and complete feedback-generation measurements."""
from export_exp0265 import *
import device_exp0257 as base
import device_exp0258 as legacy
import exp0240_device as old
import subprocess,shlex,struct,tarfile,argparse,statistics
base.R=R;base.REMOTE='/data/local/tmp/qwen3-block-htp/exp0265'
REMOTE=base.REMOTE+'-r4';CONTROL='/data/local/tmp/qwen3-block-htp/exp0258-package'
SEED=CONTROL+'/prefix_kv_u8.bin'
def records(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.startswith('{')]
def deploy():
 preflight();a=read(R/'export_audit.json');assert a['pass_all'];m=read(O/'r4/manifest.json');assert sha(O/'r4/manifest.json')==a['manifest_sha256'];pm=read(P/'manifest.json')
 raw=base.adb('shell','cd '+CONTROL+' && sha256sum '+' '.join(shlex.quote(n) for n in pm['files']),timeout=600);got={l.split(None,1)[1].strip():l.split()[0] for l in raw.splitlines()};assert all(got[n]==v['sha256'] for n,v in pm['files'].items())
 assert a['frozen_prefix_sha256'] in base.adb('shell','sha256sum '+SEED)
 for n,v in m['files'].items():assert sha(O/'r4'/n)==v['sha256']
 changed=set(a['changed_files']);lines=['set -e'];archive=O/'delta.tar'
 with tarfile.open(archive,'x',dereference=True) as tar:
  for n in m['files']:
   lines.append('mkdir -p '+shlex.quote(str(Path(n).parent)))
   if n in changed:tar.add(O/'r4'/n,arcname=n)
   else:lines.append('ln -s '+shlex.quote(CONTROL+'/'+n)+' '+shlex.quote(n))
  tar.add(O/'r4/manifest.json',arcname='manifest.json')
 checks=R/'device_files.sha256';checks.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in m['files'].items()));script=R/'device_links.sh';script.write_text('\n'.join(lines)+'\n')
 base.adb('shell','mkdir '+REMOTE)
 for p in [archive,checks,script]:base.adb('push',base.win(p),REMOTE+'/'+p.name,timeout=600)
 raw=base.adb('shell',f'cd {REMOTE} && sh device_links.sh && tar -xf delta.tar && sha256sum -c device_files.sha256',timeout=600);assert raw.count(': OK')==len(m['files']) and 'FAILED' not in raw
 write(R/'device_package.json',dict(pass_all=True,remote=REMOTE,manifest_sha256=a['manifest_sha256'],verified_files=len(m['files']),control_remote=CONTROL,control_manifest_sha256=a['parent_manifest_sha256'],control_files_verified=len(pm['files']),prefix_sha256=a['frozen_prefix_sha256']));print('DEPLOY_PASS',flush=True)
def physical(ps,count,arm,opt,dump):
 legacy.physical(ps,count,1)
 for q in ps:
  assert q['dense_r3_optimization']==2 and q['dense_r3_total_parallel_heads']==count*24
  assert q['dense_r4_mode']==arm and q['dense_r4_optimization']==(opt if arm else 0)
  assert q['dense_r4_calls']==count*arm and q['dense_r4_hmx_calls']==count*arm*(9 if q['logical_m']==64 else 2)
  assert q['dense_r4_rows']==count*arm*q['logical_m'] and q['dense_r4_pipeline_batches']==(8*count*arm if q['logical_m']==64 else 0)
  if arm and q['logical_m']==64:assert q['dense_r4_parallel_dispatches']==9*count and q['dense_r4_parallel_prepare_tiles']==192*count and q['dense_r4_parallel_finish_groups']==96*count
  if not dump:assert q['dense_r4_audit_bytes']==q['u8_attention_audit_ddr_write_bytes']==0

def run(arm,repeat,tag,count=28,opt=6,dump=False):
 preflight();root=base.runtime_root(count);e=dict(old.ENV,LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_WIDE_SCORE='4',QBH_DENSE_R3='1',QBH_R3_OPT='2',QBH_DENSE_R4=str(arm),QBH_R4_OPT=str(opt if arm else 0),QBH_PREFIX_KV='1',QBH_PREFIX_FILE=SEED,QBH_KV_CACHE_CAPACITY='128')
 if count==28:
  assert read(R/'slice_gate.json')['pass_all'];e.update(QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS='16');ids=read(R/'prompts.json')['samples'][0]['token_ids'];assert len(ids)==64
  row=[0,2,16]+ids+[0]*16;f=R/(tag.replace('/','_')+'_eval.bin');f.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)));remote=root+'/'+f.name;base.adb('push',base.win(f),remote);e['QBH_EVAL_FILE']=remote
 if dump:
  e.update(QBH_REPLAY_DUMP_DIR=root+'/'+tag.replace('/','_'),QBH_DENSE_R3_AUDIT='1',QBH_DENSE_R4_AUDIT='1');base.adb('shell','mkdir '+e['QBH_REPLAY_DUMP_DIR'])
 p=R/tag
 if (p/'stdout.jsonl').exists():assert not (p/'validated.json').exists()
 else:
  p=base.execute(count,e,CONTROL if arm==0 else REMOTE,1,tag)
  if dump and count!=28:base.adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',base.win(p))
 rs=records(p/'stdout.jsonl');ps=[x for x in rs if x.get('record')==('generation_profile' if count==28 else 'exp0240_profile')];assert len(ps)==(16*repeat if count==28 else 9);physical(ps,count,arm,opt,dump)
 fs=[x for x in rs if x.get('generation_sequence_complete')];codes=[(q['selected_token_id'],q['selected_logit_half_bits']) for q in rs if 'selected_logit_half_bits' in q]
 if count==28:
  assert len(fs)==repeat and all(f['all_steps_pass'] for f in fs);assert all(f['token_ids']==fs[0]['token_ids'] for f in fs);assert len(codes)==16*repeat and all(v==codes[i%16] for i,v in enumerate(codes))
  gold=R/('full_control' if arm==0 else 'full_previous')/'validated.json'
  if gold.exists():g=read(gold);assert fs[0]['token_ids']==g['token_ids'] and codes==[tuple(x) for x in g['selected_codes']]*repeat
  elif arm==0:
   g=read(R.parent/'exp0259/smoke_a1/validated.json');assert fs[0]['token_ids']==g['token_ids'];qs=records(R.parent/'exp0259/smoke_a1/stdout.jsonl');assert codes==[(q['selected_token_id'],q['selected_logit_half_bits']) for q in qs if 'selected_logit_half_bits' in q]*repeat
 z=dict(pass_all=True,arm=arm,opt=opt if arm else 0,repeat=repeat,layers=count,profiles=len(ps),prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode'),token_ids=fs[0]['token_ids'] if fs else None,selected_codes=codes[:16] if codes else None,output_hashes=[q['output_hash'] for q in ps] if count!=28 else None)
 write(p/'validated.json',z);print('PASS',tag,round(z['prefill_ns']/1000,2),round(z['decode_ns']/1000,2),flush=True);return z

def profile(phase):
 preflight();assert read(R/'full_gate.json')['pass_all']
 if phase=='formal':assert read(R/'short_gate.json')['integrity_pass']
 n=5 if phase=='short' else 10;rows=[]
 for i in range(n):
  for rep in [1,10]:
   for arm in ([0,1] if i%2==0 else [1,0]):
    tag=f'{phase}/round{i+1:02d}_r{rep}_a{arm}';p=R/tag/'validated.json';z=read(p) if p.exists() else run(arm,rep,tag);rows.append(dict(round=i+1,**z))
 rng=np.random.default_rng(265);perf={}
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [0,1]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);perf[f'r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist(),auxiliary_only=rep==1)
 write(R/(phase+'_gate.json'),dict(integrity_pass=True,performance=perf,rows=rows,repeat10_speed_eligible=all(v['ci95'][1]<=1.1 for k,v in perf.items() if k.startswith('r10_')),repeat1_auxiliary=True,protocol='PC079'));print(phase.upper()+'_COMPLETE',json.dumps(perf),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=28);p.add_argument('--arm',type=int,default=1);p.add_argument('--opt',type=int,default=6);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',default='smoke');p.add_argument('--dump',action='store_true');a=p.parse_args()
 if a.action=='stage':base.stage(a.count)
 elif a.action=='deploy':deploy()
 elif a.action in ['short','formal']:profile(a.action)
 else:run(a.arm,a.repeat,a.tag,a.count,a.opt,a.dump)
