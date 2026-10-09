#!/usr/bin/env python3
"""Frozen EXP0333: full2048 quality and short/long compatible rotation presets."""
import argparse, hashlib, importlib.util, json, math, os, shlex, shutil, struct, subprocess, time
from pathlib import Path
S=Path(__file__).resolve().parents[1]
R=Path(os.environ.get('QBH_RESULTS_ROOT','/mnt/d/llm_exp/results/qwen3-block-htp'))/'exp0333'
PARENTS=R.parent
REMOTE='/data/local/tmp/qwen3-block-htp/exp0333'
SERIAL='3B15C8007Z300000'
ADB=['/mnt/c/adb/adb.exe','-P','5038','-s',SERIAL]
TOKEN_FILE=Path('/mnt/d/llm_exp/results/quality-down16-20261005/qwen17/ppl/tokens.json')
TOKEN_SHA='e08c49458e0f03f12781431ad46d6118f636d1129a495307dca2f16a989a39f0'
ARMS={
 'C0':dict(r3=0,r4=0,r4opt=0,rounding=0),
 'C2':dict(r3=0,r4=0,r4opt=0,rounding=2),
 'R2':dict(r3=1,r4=0,r4opt=0,rounding=2),
 'D0':dict(r3=0,r4=5,r4opt=10,rounding=0),
 'D2':dict(r3=0,r4=5,r4opt=10,rounding=2),
 'RD2':dict(r3=1,r4=5,r4opt=10,rounding=2),
 'B2':dict(r3=0,r4=4,r4opt=12,rounding=2),
 'RB2':dict(r3=1,r4=4,r4opt=12,rounding=2),
}
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def preflight():
 text=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'EXPERIMENT=EXP-0333' in text
 return text
def adb(*args,timeout=1200):return subprocess.run([*ADB,*args],text=True,capture_output=True,check=True,timeout=timeout)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def package_key(arm):return ('rotated' if ARMS[arm]['r4'] else 'original')+('-on' if ARMS[arm]['r3'] else '-off')
def prepare():
 preflight();R.mkdir(exist_ok=False)
 sources={
  'original-off':read(PARENTS/'exp0318/package-64.json'),
  'original-on':read(PARENTS/'exp0324/package.json'),
  'rotated-off':read(PARENTS/'exp0332/package-off.json'),
  'rotated-on':read(PARENTS/'exp0332/package-on.json'),
 }
 for key,meta in sources.items():
  p=Path(meta['package']);assert sha(p/'manifest.json')==meta['manifest_sha256'];j=read(p/'manifest.json')
  for n,v in j['files'].items():assert sha(p/n)==v['sha256'],(p,n)
  # Verify cached actual payloads on the phone, not merely local manifests.
  names=list(j['files'])
  for i in range(0,len(names),32):
   ns=names[i:i+32];ls=adb('shell','sha256sum '+' '.join(shlex.quote(meta['remote']+'/'+n) for n in ns)).stdout.splitlines()
   assert len(ls)==len(ns)
   for line in ls:
    h,n=line.split(None,1);assert h==j['files'][n.removeprefix(meta['remote']+'/')]['sha256']
  prefix=Path(meta['package'])/'prefix_kv_u8.bin' if key!='original-off' else PARENTS/'exp0318/prefix_kv_u8.bin'
  remote_prefix=meta['remote']+'/prefix_kv_u8.bin' if key!='original-off' else '/data/local/tmp/qwen3-block-htp/exp0257-prefix/prefix_kv_u8.bin'
  assert adb('shell','sha256sum '+remote_prefix).stdout.split()[0]==sha(prefix)
  save(R/('package-'+key+'.json'),dict(**meta,prefix=str(prefix),prefix_remote=remote_prefix,prefix_sha256=sha(prefix),verified_files=len(names)))
  print('PACKAGE_VERIFIED',key,len(names),flush=True)
 assert sha(TOKEN_FILE)==TOKEN_SHA
 save(R/'freeze.json',dict(arms=ARMS,target_tokens=8192,token_sha256=TOKEN_SHA,window_total_tokens=2048,warmup_tokens=64,targets_per_full_window=1984,window_counts=[1984]*4+[256],context_range=[64,2047],scope='WikiText2 raw test subset,not fullcorpus',long_opt=2555807,kv_capacity=2112,performance_development_prompt_tokens=64))
 # Matched historical teacher and R3 controls must stay immutable.
 pins={320:'12c97c9adb317a4c956bbd0115a8643159d27dc10fb65829c2ce66b2c4e1d657',326:'0634f0639783b824f513e447be4f2c4a8f953688aec510351c1f7131e6d4a755',332:'2202021c0d7d983af42f3e54e1fa2fea53301dfa57b901e7a8282d1df1215ca1'}
 checked={}
 for n,h in pins.items():
  root=PARENTS/f'exp{n:04d}';assert sha(root/'EVIDENCE_SHA256.json')==h;ledger=read(root/'EVIDENCE_SHA256.json')
  wanted=['rows-1984.json','result-1984.json','provenance.json'] if n==320 else (['quality/rows.json','quality/result.json'] if n==326 else ['comparison.json']+[f'quality/{a}/rows.json' for a in ('D0','D2','RD2','B2','RB2')])
  for f in wanted:assert sha(root/f)==ledger[f],(n,f)
  checked[str(n)]=dict(ledger_sha256=h,checked_files=wanted)
 save(R/'parent-evidence-verified.json',checked)

def legacy():
 spec=importlib.util.spec_from_file_location('exp0332_build',S/'scripts/run_exp0332_r3_r4.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.R=R;m.REMOTE=REMOTE;m.preflight=preflight
 return m

def build(layers,tag):legacy().build(layers,tag,False)

def configuration(arm,build):
 rt=read(R/('runtime-'+build+'.json'));p=read(R/('package-'+package_key(arm)+'.json'))
 cfg=ARMS[arm];pr=read(PARENTS/'exp0318/runs/boundary-64-a3/protocol.json');words=shlex.split(pr['command']);i=words.index('./qwen3_block_cli')
 env=dict(v.split('=',1) for v in words[3:i]);argv=words[i+1:]
 for k in ['QBH_EVAL_FILE','QBH_GENERATION_BOUNDARY_AUDIT','QBH_GENERATION_AUDIT_DIR','QBH_DENSE_R3_AUDIT','QBH_DENSE_R4_AUDIT','QBH_LONG_CACHE_GUARD']:env.pop(k,None)
 env.update(LD_LIBRARY_PATH=rt['remote'],DSP_LIBRARY_PATH=rt['remote'],ADSP_LIBRARY_PATH=rt['remote'],QBH_DENSE_R3=str(cfg['r3']),QBH_R3_OPT='2' if cfg['r3'] else '0',QBH_DENSE_R4=str(cfg['r4']),QBH_R4_OPT=str(cfg['r4opt']),QBH_PROJECTION_ROUNDING=str(cfg['rounding']),QBH_LONG_OPT='2555807',QBH_KV_CACHE_CAPACITY='2112',QBH_PREFIX_FILE=p['prefix_remote'])
 return rt,p,env,argv

def window(arm,start,count,tag,build='full',prompt_length=64,audit=False,length_override=None):
 preflight();ids=read(TOKEN_FILE);assert sha(TOKEN_FILE)==TOKEN_SHA and len(ids)==8256
 d=R/'runs'/tag;d.mkdir(parents=True,exist_ok=False);rt,p,env,argv=configuration(arm,build)
 prompt=ids[start-prompt_length:start];targets=ids[start:start+count];assert len(prompt)==prompt_length and len(targets)==count
 for n,w in [('long_prompt_u32.bin',prompt),('targets.bin',targets)]:
  (d/n).write_bytes(struct.pack('<'+'I'*len(w),*w))
 remote=REMOTE+'/inputs/'+tag;adb('shell','mkdir -p '+shlex.quote(remote))
 # Share immutable package through symlinks and override only the corpus inputs.
 files=read(Path(p['package'])/'manifest.json')['files'];names=[n for n in files if n not in ('long_prompt_u32.bin','targets.bin')]
 dirs=sorted({str(Path(n).parent) for n in names});adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs))
 for i in range(0,len(names),40):adb('shell',' && '.join('ln -s '+shlex.quote(p['remote']+'/'+n)+' '+shlex.quote(remote+'/'+n) for n in names[i:i+40]))
 # Rotated EXP0332 packages deliberately omit unrelated long-input files.
 for n in ['long_fixed_u32.bin','long_rope_cos_f16.bin','long_rope_sin_f16.bin']:
  if n not in files:adb('shell','ln -s '+shlex.quote('/data/local/tmp/qwen3-block-htp/exp0318/models/64/'+n)+' '+shlex.quote(remote+'/'+n))
 for n in ['long_prompt_u32.bin','targets.bin']:
  adb('push',win(d/n),remote+'/'+n);assert adb('shell','sha256sum '+remote+'/'+n).stdout.split()[0]==sha(d/n)
 env.update(QBH_LONG_PREFILL_TOKENS=str(prompt_length),QBH_LONG_TARGET_FILE=remote+'/targets.bin',QBH_LONG_DECODE_STEPS=str(count-1),QBH_LONG_REPEATS='1')
 if audit:
  dump=REMOTE+'/audit/'+tag;adb('shell','mkdir -p '+dump)
  env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=dump,QBH_LONG_CACHE_GUARD='1',QBH_LONG_AUDIT_FROM_POSITION=str(prompt_length))
  if ARMS[arm]['r3']:env['QBH_DENSE_R3_AUDIT']='1'
  if ARMS[arm]['r4']:env['QBH_DENSE_R4_AUDIT']='1'
 argv[0]=remote
 cmd='cd '+rt['remote']+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+shlex.join(argv)
 save(d/'protocol.json',dict(command=cmd,arm=arm,configuration=ARMS[arm],runtime=rt,package=p,start=start,count=count,prompt=prompt,targets=targets,audit=audit,prompt_length=prompt_length))
 start_time=time.monotonic();records=[]
 with (d/'stdout.txt').open('w') as out,(d/'stderr.txt').open('w') as err:
  proc=subprocess.Popen([*ADB,'shell',cmd],stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
  for line in proc.stdout:
   out.write(line)
   try:v=json.loads(line)
   except ValueError:continue
   records.append(v)
   if v.get('record')=='long_eval' and v['target_index']%256==0:print('PROGRESS',tag,v['target_index'],v['cache_valid'],flush=True)
  rc=proc.wait()
 save(d/'exit.json',dict(returncode=rc,elapsed_seconds=time.monotonic()-start_time));save(d/'records.json',records)
 assert rc==0,(tag,(d/'stderr.txt').read_text()[-2500:])
 assert any(v.get('record')=='long_complete' for v in records)
 rows=[v for v in records if v.get('record')=='long_eval'];assert len(rows)==count
 for i,v in enumerate(rows):
  assert v['target_index']==i and v['target_token']==targets[i] and v['cache_valid']==prompt_length+i and v['vocab_count']==151936
  assert v['saturated']==0 and math.isfinite(v['nll']) and abs(v['nll']-(v['logsumexp']-v['target_logit']))<2e-5
  v['corpus_position']=start+i
 for v in records:
  if v.get('record')=='long_profile':
   assert v['dsp_status']==3 and v['vtcm_acquired_bytes']==v['vtcm_requested_bytes']==8388608
   assert v['vtcm_peak_plan_bytes']<=8388608 and v['intermediate_ddr_read_bytes']==v['intermediate_ddr_write_bytes']==v['intermediate_spill_fill_count']==0
 save(d/'scores.json',rows)
 if audit:adb('pull',dump+'/.',win(d/'audit'))
 print('WINDOW_PASS',tag,len(rows),math.exp(math.fsum(v['nll'] for v in rows)/count),flush=True)
 return rows

def quality(arm):
 rows=[]
 for i,start in enumerate(range(64,8256,1984)):
  tag='ppl-'+arm+f'-{i}'
  if (R/'runs'/tag/'scores.json').exists():
   assert read(R/'runs'/tag/'exit.json')['returncode']==0
   got=read(R/'runs'/tag/'scores.json')
  else:got=window(arm,start,min(1984,8256-start),tag)
  rows+=got
 assert [v['corpus_position'] for v in rows]==list(range(64,8256))
 teacher=read(PARENTS/'exp0320/rows-1984.json')
 for a,b in zip(rows,teacher):assert a['corpus_position']==b['position'] and a['target_token']==b['target'] and a['cache_valid']==b['context']
 mean=math.fsum(v['nll'] for v in rows)/8192
 save(R/'quality'/arm/'rows.json',rows);result=dict(arm=arm,configuration=ARMS[arm],ppl=math.exp(mean),mean_nll=mean,targets=8192,windows=5,scope='WT2 rawtest subset;full2048 including64warmup,1984targets/window,256targettail;context64..2047',saturation=sum(v['saturated'] for v in rows))
 save(R/'quality'/arm/'result.json',result);print('FINAL_PPL',json.dumps(result),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','build','window','quality']);a.add_argument('--arm',choices=list(ARMS),default='RD2');a.add_argument('--layers',type=int,default=28);a.add_argument('--tag',default='full');a.add_argument('--build-tag',default='full');a.add_argument('--start',type=int,default=64);a.add_argument('--count',type=int,default=43);a.add_argument('--prompt',type=int,default=64);a.add_argument('--audit',action='store_true');v=a.parse_args()
 if v.action=='prepare':prepare()
 elif v.action=='build':build(v.layers,v.tag)
 elif v.action=='window':window(v.arm,v.start,v.count,v.tag,v.build_tag,v.prompt,v.audit)
 else:quality(v.arm)
