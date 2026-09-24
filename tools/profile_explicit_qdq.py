#!/usr/bin/env python3
"""Matched explicit-QDQ diagnostic. Original fused results remain immutable."""
import argparse,hashlib,json,shlex,shutil,struct,subprocess
from pathlib import Path
S=Path(__file__).resolve().parents[1]
ADB='/mnt/c/adb/adb.exe'
SERIAL='3B15C8007Z300000'
COUNTERS=['fp_softmax_dq_ticks','fp_softmax_compute_ticks','fp_softmax_q_ticks','fp_swiglu_dq_ticks','fp_swiglu_compute_ticks','fp_swiglu_q_ticks']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+chr(10))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def adb(*a,check=True):
 return subprocess.run([ADB,'-P','5038','-s',SERIAL,*a],capture_output=True,text=True,check=check,timeout=900)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'ACTIVE_EXPERIMENT=L32-0073' in z
def stage(a):
 preflight();seal=read(S/'build/llama-build-seal.json')
 assert seal['fp_islands'] and seal['model_size']==a.model
 assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 name=f'binaries-{a.layers}-{a.version}'
 d=R/name;d.mkdir(parents=True,exist_ok=False);remote=REMOTE+'/'+name
 adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h
  shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli')
 save(d/'seal.json',seal)
def execute(a,tag,arm,repeat=1,audit=False):
 from run_llama32_frontend import records
 from report_llama32_pipeline_profile import MODULES
 preflight()
 d=R/tag
 if (d/'validated.json').exists():return read(d/'validated.json')
 assert not (d/'exit.json').exists(),'Retain failure; use a fresh tag'
 d.mkdir(parents=True,exist_ok=True)
 base=read(OLD/'phases-formal-00/protocol.json')['command']
 prefix,args=base.split(' ./qwen3_block_cli ',1)
 env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]))
 argv=shlex.split(args);binary=REMOTE+f'/binaries-{a.layers}-{a.version}'
 for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
 env.update(LD_LIBRARY_PATH=binary,DSP_LIBRARY_PATH=binary,ADSP_LIBRARY_PATH=binary,QBH_FP_QDQ_SPLIT=str(int(arm=='split')))
 raw=(OLD/'phases-formal-00/trajectory.bin').read_bytes();stored=struct.unpack('<'+'I'*(len(raw)//4),raw);fixture=dict(prompt=list(stored[7:71]),fixed=list(stored[71:73]));words=[0x51424556,2,repeat,69]
 for j in range(repeat):words += [j,3,2]+fixture['prompt']+fixture['fixed'][:2]
 ef=d/'trajectory.bin';ef.write_bytes(struct.pack('<'+'I'*len(words),*words));er=binary+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
 if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
 cmd='cd '+binary+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
 save(d/'protocol.json',dict(command=cmd,layers=a.layers,repeat=repeat,arm=arm,audit=audit,seal_sha256=sha(R/f'binaries-{a.layers}-{a.version}/seal.json'),old_protocol_sha256=sha(OLD/'phases-formal-00/protocol.json'),reference_sha256=sha(OLD/'reference/summary.json')))
 adb('push',win(ef),er)
 if audit:adb('shell','mkdir -p '+env['QBH_GENERATION_AUDIT_DIR'])
 z=adb('shell',cmd,check=False)
 (d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode))
 if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',win(d/'audit'),check=False)
 assert z.returncode==0,(tag,z.stderr[-2000:],z.stdout[-1500:])
 rr=records((d/'stdout.txt').read_text());pp=[x for x in rr if x.get('record')=='generation_profile']
 ss=[x for x in rr if 'selected_logit_half_bits' in x and 'generation_step' in x]
 assert len(pp)==len(ss)==2*repeat,(tag,len(pp),len(ss))
 gold=read(OLD/'reference/summary.json');checks=[]
 save(d/'records.json',rr)
 for j,(x,t) in enumerate(zip(pp,ss)):
  step=j%2;g=gold['heads'][str(a.layers)][step]
  assert x['dsp_status']==3 and x['numerical_status']==1
  assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']<=8388608
  assert x['intermediate_spill_fill_count']==x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==0
  total=sum(sum(x[k] for k in f) for _,f in MODULES)-x['generation_final_norm_ticks'];assert total==x['invocation_ticks']
  for i in range(a.layers):
   layer=x[f'slice_layer_{i}'];assert layer['status']==3 and layer['layer_unattributed_ticks']==0
   if audit:
    got=int(layer['output_hash'],16) if isinstance(layer['output_hash'],str) else layer['output_hash']
    checks.append(dict(step=step,layer=i,exact=got==gold['hashes'][i][step],got=got,expected=gold['hashes'][i][step]))
  assert (t['selected_token_id'],t['selected_logit_half_bits'])==(g['token'],g['code']),(tag,j,'head',t,g)
  if arm=='split' and step==0:
   assert all(x[k]>0 for k in COUNTERS)
   assert sum(x[k] for k in COUNTERS[:3])==x['attention_softmax_ticks']
   assert sum(x[k] for k in COUNTERS[3:])<=x['activation_ticks']
  if arm=='fused':assert all(x[k]==0 for k in COUNTERS)
 if audit:
  save(d/'hash-checks.json',checks)
  assert all(x['exact'] for x in checks),[x for x in checks if not x['exact']][:5]
 save(d/'validated.json',dict(pass_checks=True,exact_layers=len(checks),profiles=pp))
 print('PASS',a.model,tag,flush=True)
 return read(d/'validated.json')
def run(a):
 for arm in ['fused','split']:execute(a,'audit-full-'+arm,arm,audit=True)
 for arm in ['fused','split']:execute(a,'warmup-'+arm,arm)
 for kind,n in [('short',5),('formal',10)]:
  for i in range(n):
   for arm in (['fused','split'] if i%2==0 else ['split','fused']):execute(a,f'{kind}-{i:02d}-{arm}',arm,repeat=10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','audit','run']);p.add_argument('--model',choices=['1B','3B'],default='1B');p.add_argument('--layers',type=int,default=16);p.add_argument('--version',default='a01');p.add_argument('--arm',choices=['fused','split'],default='split');p.add_argument('--tag',default='audit');a=p.parse_args()
 R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0073')/a.model;REMOTE='/data/local/tmp/llama32-htp/l32-0073/'+a.model
 OLD=Path('/mnt/d/llm_exp/results/llama32-htp')/('l32-0063' if a.model=='1B' else 'l32-0064')
 if a.action=='stage':stage(a)
 elif a.action=='audit':execute(a,a.tag,a.arm,audit=True)
 else:run(a)
