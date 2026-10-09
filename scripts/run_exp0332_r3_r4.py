#!/usr/bin/env python3
"""Reproduce EXP0332 composition, device checks, quality and workflow timing."""
import argparse, hashlib, json, math, os, shlex, shutil, struct, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

S=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0332')
MODELS=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0332')
ROOT=R.parent
REMOTE='/data/local/tmp/qwen3-block-htp/exp0332'
SERIAL='3B15C8007Z300000'
ADB=['/mnt/c/adb/adb.exe','-P','5038','-s',SERIAL]
ARMS={
 'D0':dict(r3=0,r4=5,r4opt=10,rounding=0),
 'D2':dict(r3=0,r4=5,r4opt=10,rounding=2),
 'RD2':dict(r3=1,r4=5,r4opt=10,rounding=2),
 'B2':dict(r3=0,r4=4,r4opt=12,rounding=2),
 'RB2':dict(r3=1,r4=4,r4opt=12,rounding=2),
}
CAP=64*6144;H=2048;F=6144

def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'EXPERIMENT=EXP-0332' in z
 return z

def stamp(label,seconds,**kw):
 R.mkdir(parents=True,exist_ok=True)
 with (R/'workflow-timings.jsonl').open('a') as f:
  f.write(json.dumps(dict(label=label,seconds=seconds,utc=datetime.now(timezone.utc).isoformat(),**kw))+'\n')
 print('TIMING',label,round(seconds,3),flush=True)
def adb(*args,check=True):return subprocess.run([*ADB,*args],text=True,capture_output=True,check=check,timeout=900)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def verify_package(p,pin):
 assert sha(p/'manifest.json')==pin,(p,'manifest')
 j=read(p/'manifest.json')
 for n,v in j['files'].items():assert sha(p/n)==v['sha256'],(p,n)
 return j

def prepare():
 preflight();R.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
 start=time.monotonic()
 base=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0329/R4')
 r3=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0324')
 b=verify_package(base,'2abe3974a237e3f65e17ddc4ce2740f946a3dcca96e1f01fe4d49832a8b9eee0')
 j=verify_package(r3,'f67d96e0f861c9d424c25d255305075924bfaa1b4249f1e22112245249ee64bc')
 # The only changed non-Down model metadata are the two Q/K grids and score config.
 for li in range(28):
  old=(base/f'layer{li}/qparams_u8.bin').read_bytes();new=(r3/f'layer{li}/qparams_u8.bin').read_bytes();assert len(old)==len(new)
  for off in range(0,len(old),48):
   name=old[off:off+32].split(b'\0')[0]
   if name not in [b'q_rope',b'k_rope']:assert old[off:off+48]==new[off:off+48],(li,name)
  for proj in ['q','k','v','o','gate','up']:
   for suffix in ['weight_w4_hmx.bin','weight_w4_scale_f32.bin']:
    n=f'layer{li}/{proj}_{suffix}';assert b['files'][n]['sha256']==j['files'][n]['sha256'],n
 offprefix=ROOT/'exp0318/prefix_kv_u8.bin'
 for name,on in [('off',False),('on',True)]:
  dst=MODELS/name;dst.mkdir(exist_ok=False);links={}
  for n in b['files']:
   use=r3/n if on and n.startswith('layer') and Path(n).name in ['qparams_u8.bin','attention_config_all_groups.bin'] else base/n
   q=dst/n;q.parent.mkdir(parents=True,exist_ok=True);os.link(use,q)
   parent_remote='/data/local/tmp/qwen3-block-htp/exp0324/model' if use.parent.parent==r3 else '/data/local/tmp/qwen3-block-htp/exp0329/model-r4'
   links[n]=parent_remote+'/'+n
  pf=r3/'prefix_kv_u8.bin' if on else offprefix
  shutil.copyfile(pf,dst/'prefix_kv_u8.bin')
  files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dst.rglob('*') if p.is_file()}
  save(dst/'manifest.json',dict(experiment='EXP-0332',r3=on,parent_r4_manifest=sha(base/'manifest.json'),parent_r3_manifest=sha(r3/'manifest.json'),files=files))
  remote=REMOTE+'/models/'+name;dirs=sorted({str(Path(n).parent) for n in files})
  adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs))
  ns=list(links)
  for i in range(0,len(ns),32):adb('shell',' && '.join('ln -s '+shlex.quote(links[n])+' '+shlex.quote(remote+'/'+n) for n in ns[i:i+32]))
  adb('push',win(dst/'prefix_kv_u8.bin'),remote+'/prefix_kv_u8.bin')
  for i in range(0,len(files),32):
   ns=list(files)[i:i+32];ls=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in ns)).stdout.splitlines();assert len(ls)==len(ns)
   for l in ls:
    h,n=l.split(None,1);assert h==files[n.removeprefix(remote+'/')]['sha256'],n
  save(R/('package-'+name+'.json'),dict(package=str(dst),remote=remote,manifest_sha256=sha(dst/'manifest.json'),files_verified=len(files)))
 stamp('model_composition_validation_and_cached_deployment',time.monotonic()-start,model_bytes_pushed=(MODELS/'off/prefix_kv_u8.bin').stat().st_size*2,full_weights_already_on_phone=True)
 print('PACKAGES_PASS',flush=True)

def build(layers,tag,clean=False):
 preflight();start=time.monotonic()
 if clean:
  backup=S/'build/exp0332-preserved-builds'/tag;backup.mkdir(parents=True,exist_ok=False)
  for n in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
   p=S/n;assert p.resolve().parent==S.resolve()
   if p.exists():shutil.move(str(p),str(backup/n))
 env=os.environ.copy();env.update(QBH_QWEN_MODEL_SIZE='1.7B',QBH_FP_ISLANDS='OFF',QBH_PAPER_TRACE='OFF')
 out=R/('build-'+tag+'.log');R.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:z=subprocess.run(['bash',str(S/'scripts/build_qwen3_sp2.sh'),str(layers)],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT)
 stamp('clean_build' if clean else 'incremental_build',time.monotonic()-start,layers=layers,tag=tag,returncode=z.returncode,log=str(out),parallel_jobs=8)
 assert z.returncode==0,out
 stage(tag,layers)

def stage(tag,layers):
 preflight();start=time.monotonic();seal=read(S/'build/qwen3-sp2-build-seal.json')
 assert seal['source_head']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
 d=R/('binaries-'+tag);d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name;adb('shell','mkdir -p '+remote)
 for name,h in seal['files'].items():
  p=Path(name);assert sha(p)==h;shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli')
 save(d/'seal.json',seal);save(R/('runtime-'+tag+'.json'),dict(remote=remote,seal=seal,layers=layers,archive=str(d)))
 stamp('binary_archive_deploy_hash_check',time.monotonic()-start,tag=tag,artifact_bytes=sum(p.stat().st_size for p in d.iterdir() if p.is_file()))

def configuration(arm,tag,build,audit=False,long=False):
 rt=read(R/('runtime-'+build+'.json'));cfg=ARMS[arm];pkg=read(R/('package-'+('on' if cfg['r3'] else 'off')+'.json'))
 p=read(ROOT/'exp0308/1.7B/audit1-INT16/protocol.json')['command'];prefix,args=p.split(' ./qwen3_block_cli ',1)
 env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=pkg['remote']
 for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT','QBH_DENSE_R4_AUDIT','QBH_DENSE_R3_AUDIT']:env.pop(k,None)
 env.update(LD_LIBRARY_PATH=rt['remote'],DSP_LIBRARY_PATH=rt['remote'],ADSP_LIBRARY_PATH=rt['remote'],QBH_SP2='8',QBH_U8_PREFILL_OPT='0',QBH_FP32_RESIDUAL='2',QBH_DENSE_R3=str(cfg['r3']),QBH_R3_OPT='2' if cfg['r3'] else '0',QBH_DENSE_R4=str(cfg['r4']),QBH_R4_OPT=str(cfg['r4opt']),QBH_PROJECTION_ROUNDING=str(cfg['rounding']),QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_PREFIX_KV='1',QBH_PREFIX_FILE=pkg['remote']+'/prefix_kv_u8.bin',QBH_SP2_DOWN_HVX='0')
 if audit:
  remote=REMOTE+'/audit/'+tag;adb('shell','mkdir -p '+remote)
  env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=remote,QBH_DENSE_R4_AUDIT='1')
  if cfg['r3']:env['QBH_DENSE_R3_AUDIT']='1'
 return rt,pkg,env,argv

def execute(arm,tag,build,words,audit=False,quiet=False):
 preflight();d=R/tag;d.mkdir(parents=True,exist_ok=False);rt,pkg,env,argv=configuration(arm,tag,build,audit)
 p=d/'trajectory.bin';p.write_bytes(struct.pack('<'+'I'*len(words),*words));remote=REMOTE+'/inputs/'+tag+'.bin';adb('shell','mkdir -p '+REMOTE+'/inputs');adb('push',win(p),remote);env['QBH_EVAL_FILE']=remote
 if quiet:env['QBH_EVAL_QUIET']='1'
 cmd='cd '+rt['remote']+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+shlex.join(argv)
 save(d/'protocol.json',dict(command=cmd,arm=arm,configuration=ARMS[arm],runtime=rt,package=pkg,audit=audit))
 start=time.monotonic();first_ready=None;first_result=None;records=[]
 with (d/'stderr.txt').open('w') as err,(d/'stdout.txt').open('w') as out:
  proc=subprocess.Popen([*ADB,'shell',cmd],stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
  for line in proc.stdout:
   out.write(line);out.flush()
   try:v=json.loads(line)
   except ValueError:continue
   records.append(v)
   if v.get('record')=='eval_model_ready':first_ready=time.monotonic()-start
   if first_result is None and v.get('record') in ['generation_profile','eval_step']:first_result=time.monotonic()-start
  rc=proc.wait()
 save(d/'exit.json',dict(returncode=rc));save(d/'records.json',records)
 stamp('process_load_and_first_device_result',time.monotonic()-start,arm=arm,tag=tag,model_ready_seconds=first_ready,first_result_seconds=first_result,returncode=rc)
 assert rc==0,(tag,(d/'stderr.txt').read_text()[-3000:])
 for v in records:
  if v.get('record')=='generation_profile':
   assert v['dsp_status']==3 and v['numerical_status']==1 and v['vtcm_peak_plan_bytes']<=8388608
   assert v['intermediate_ddr_read_bytes']==v['intermediate_ddr_write_bytes']==v['intermediate_spill_fill_count']==0
 for v in records:
  if v.get('record')=='eval_step':
   assert v['pass'] and v['vtcm_bytes']<=8388608 and v['intermediate_read']==v['intermediate_write']==v['spill']==0
 if audit:adb('pull',REMOTE+'/audit/'+tag+'/.',win(d/'audit'))
 print('RUN_PASS',tag,len(records),flush=True);return records

def fixed(arm,tag,build,steps=4,audit=True):
 f=read(ROOT/'exp0302/1.7B/a8/fixture.json');w=[0x51424556,2,1,67+steps,0,3,steps]+f['prompt_ids'][:64]+f['fixed'][:steps]
 return execute(arm,tag,build,w,audit)

def quality(arm,build='full'):
 ids=read(Path('/mnt/d/llm_exp/results/quality-down16-20261005/qwen17/ppl/tokens.json'));assert len(ids)==8256
 assert sha(Path('/mnt/d/llm_exp/results/quality-down16-20261005/qwen17/ppl/tokens.json'))=='e08c49458e0f03f12781431ad46d6118f636d1129a495307dca2f16a989a39f0'
 samples=[dict(id=i,start=start,context=ids[start-64:start],targets=ids[start:min(start+43,8256)]) for i,start in enumerate(range(64,8256,43))]
 allrows=[];start_time=time.monotonic()
 for batch in range(0,len(samples),64):
  group=samples[batch:batch+64];w=[0x51424556,2,len(group),110]
  for x in group:w += [x['id'],1,len(x['targets'])]+x['context']+x['targets']+[0]*(43-len(x['targets']))
  tag='quality/'+arm+f'/batch-{batch//64:02d}';rr=execute(arm,tag.replace('/','-'),build,w,False,quiet=True)
  scores=[v for v in rr if v.get('record')=='eval_step'];by={x['id']:x for x in group}
  assert len(scores)==sum(len(x['targets']) for x in group)
  for v in scores:
   x=by[v['sample_id']];i=v['step'];assert v['target_token']==x['targets'][i] and v['pass'] and v['cache_valid']==64+i and v['vocab_count']==151936 and v['nonfinite']==0
   assert math.isfinite(v['nll']) and abs(v['nll']-(v['logsumexp']-v['target_logit']))<2e-5
   allrows.append(dict(corpus_position=x['start']+i,**v))
  print('PPL_PROGRESS',arm,len(allrows),math.exp(math.fsum(v['nll'] for v in allrows)/len(allrows)),flush=True)
 assert [v['corpus_position'] for v in allrows]==list(range(64,8256))
 mean=math.fsum(v['nll'] for v in allrows)/8192
 result=dict(arm=arm,configuration=ARMS[arm],target_count=8192,windows=len(samples),mean_nll=mean,ppl=math.exp(mean),elapsed_seconds=time.monotonic()-start_time,scope='WT2 test subset8192targets64warmup43target resets, tail retained; not full corpus PPL',saturated_entries=sum(v['saturated'] for v in allrows))
 save(R/'quality'/arm/'rows.json',allrows);save(R/'quality'/arm/'result.json',result);print('FINAL_PPL',json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build','fixed','quality']);p.add_argument('--arm',choices=list(ARMS),default='RD2');p.add_argument('--tag',default='one');p.add_argument('--layers',type=int,default=1);p.add_argument('--build-tag');p.add_argument('--clean',action='store_true');p.add_argument('--steps',type=int,default=4);p.add_argument('--no-audit',action='store_true');a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='build':build(a.layers,a.tag,a.clean)
 elif a.action=='fixed':fixed(a.arm,a.tag,a.build_tag or a.tag.split('-')[0],a.steps,not a.no_audit)
 else:quality(a.arm,a.build_tag or 'quality')
