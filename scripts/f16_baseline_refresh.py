#!/usr/bin/env python3
"""Bounded W16A16 fairness refresh: frozen inputs, exact paired audit and E2E."""
import argparse,hashlib,json,os,shlex,shutil,statistics,struct,subprocess
from pathlib import Path
S=Path(__file__).resolve().parents[1]
LLAMA=S.name=='llama32-htp'; EXP='L32-0039' if LLAMA else 'EXP-0283';COUNT=16 if LLAMA else 28
R=Path('/mnt/d/llm_exp/results')/S.name/EXP.lower().replace('exp-','exp')
P=Path('/mnt/d/llm_exp/models')/S.name/('l32-0001/frontend-a02' if LLAMA else 'exp0217/f16f16_greedy16')
REMOTE_PACKAGE='/data/local/tmp/llama32-htp/l32-0001/device-frontend-a03/package' if LLAMA else '/data/local/tmp/qwen3-block-htp/exp0218-f16f16/block_package_layer14_m64'
MANIFEST='c5df4b14a06af3274063a228f1fe9fb5480064b0386d0af5c90e24476ce0d5d9' if LLAMA else '0f8a359b559f252cb13329f57d4cabd56f17ca9bfb64bc5643f4d6014795070f'
ADB='/mnt/c/adb/adb.exe'
ARGS='2 32 hvx on off fused gate8_interleaved control hvx crouton_native_batch8 4 64 '+('parallel_qk_norm_rope' if LLAMA else 'gqa_qkv_overlap')+' 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0'
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def preflight():
 out=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert EXP in out
 return subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
def adb(*args,check=True):return subprocess.run([ADB,*args],check=check,capture_output=True,text=True,timeout=900)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def records(p):
 out=[]
 for l in Path(p).read_text().splitlines():
  try:out.append(json.loads(l))
  except ValueError:pass
 return out

def verify():
 head=preflight();assert sha(P/'manifest.json')==MANIFEST
 m=read(P/'manifest.json');n=0
 for name,z in m['files'].items():
  assert sha(P/name)==z['sha256'],name;n+=1
  if n%200==0:print('LOCAL_HASHED',n,flush=True)
 assert adb('shell','sha256sum '+REMOTE_PACKAGE+'/manifest.json').stdout.split()[0]==MANIFEST
 names=list(m['files'])
 for i in range(0,len(names),32):
  out=adb('shell','sha256sum '+' '.join(shlex.quote(REMOTE_PACKAGE+'/'+n) for n in names[i:i+32])).stdout
  pairs=[l.split(None,1) for l in out.splitlines()];assert len(pairs)==len(names[i:i+32])
  for h,n in pairs:assert h==m['files'][n.removeprefix(REMOTE_PACKAGE+'/')]['sha256'],n
  if i%192==0:print('REMOTE_HASHED',i,flush=True)
 write(R/'package-verified.json',dict(source_head=head,package=str(P),remote=REMOTE_PACKAGE,manifest_sha256=MANIFEST,files=len(names)))

def stage(count):
 head=preflight();seal=read(S/'build'/('llama-build-seal.json' if LLAMA else 'qwen3-sp2-build-seal.json'));assert seal['source_head']==head
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  c=(S/b/'CMakeCache.txt').read_text();assert f"{'QBH_LLAMA_LAYER_COUNT' if LLAMA else 'QBH_EXP0257_LAYER_COUNT'}:STRING={count}\n" in c
 attempt=1
 while (R/f'binaries/l{count}-a{attempt}').exists():attempt+=1
 d=R/f'binaries/l{count}-a{attempt}';d.mkdir(parents=True)
 remote=f'/data/local/tmp/{S.name}/{EXP.lower()}-l{count}-a{attempt}'
 assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h;shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli')
 write(d/'seal.json',seal);cfg=dict(remote=remote,source_head=head,seal=seal,count=count,archive=str(d),boot_id=adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip())
 (R/f'runtime-l{count}.json').write_text(json.dumps(cfg,indent=2));print('STAGED',count,flush=True)

def run(count,opt,repeat,tag,audit=False,package=None):
 preflight();assert read(R/'package-verified.json')['manifest_sha256']==MANIFEST
 runtime=read(R/f'runtime-l{count}.json');remote=runtime['remote'];d=R/tag;d.mkdir(parents=True,exist_ok=False)
 e=dict(LD_LIBRARY_PATH=remote,DSP_LIBRARY_PATH=remote,ADSP_LIBRARY_PATH=remote,QBH_VERTICAL_SLICE='1',QBH_REPLAY_SEQUENCE='1',QBH_SCAN_MODE='prefill',QBH_LOGICAL_M='64',QBH_KV_CACHE_LENGTH='0',QBH_KV_CACHE_CAPACITY='80',QBH_KV_CACHE_LAYOUT='hmx_native_f16',QBH_F16F16_OPT=str(opt))
 if audit and opt:e['QBH_W4F16_DECODE_AUDIT']='1'
 if audit:
  target=remote+'/'+tag.replace('/','_');adb('shell','mkdir -p '+target)
  e['QBH_GENERATION_AUDIT_DIR' if count==COUNT else 'QBH_REPLAY_DUMP_DIR']=target
 if count==COUNT:
  e.update(QBH_GENERATION_SEQUENCE='10',QBH_GENERATION_STEPS='16')
  ids=list(struct.unpack('<64I',(P/'generation_prompt_token_ids_u32.bin').read_bytes()))
  row=[0,2,16]+ids+[0]*16
  evalfile=d/'eval.bin';evalfile.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)))
  rfile=remote+'/'+tag.replace('/','_')+'-eval.bin';adb('push',win(evalfile),rfile);e['QBH_EVAL_FILE']=rfile
 else:e['QBH_REPLAY_DECODE_STEPS']='8' if not LLAMA else '1'
 cmd='cd '+remote+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {package or REMOTE_PACKAGE} F16F16 {1 if count==COUNT else repeat} {ARGS}'
 write(d/'protocol.json',dict(runtime=runtime,command=cmd,manifest_sha256=MANIFEST,opt=opt,repeat=repeat,count=count,audit=audit))
 z=adb('shell',cmd,check=False);(d/'stdout.jsonl').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode))
 if audit:adb('pull',target+'/.',win(d),check=False)
 assert z.returncode==0,(tag,z.stderr[-2000:],z.stdout[-2500:])
 rs=records(d/'stdout.jsonl');ps=[r for r in rs if r.get('record') in ['exp0240_profile','generation_profile','replay_profile','vertical_slice_replay_profile']]
 if not ps:raise ValueError(('No profile records',[(r.get('record'),list(r)[:4]) for r in rs[-4:]]))
 for q in ps:
  assert q['variant']=='F16F16' and q['block_invocation_count']==count
  assert q['vtcm_requested_bytes']==q['vtcm_acquired_bytes']==8388608 and q['vtcm_peak_plan_bytes']<=8388608
  assert all(q[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','w4f16_decode_conversion_audit_mismatches'])
  assert q['w4f16_decode_opt']==opt
 out=dict(tag=tag,opt=opt,repeat=repeat,count=count,profiles=len(ps),prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode'),output_hashes=[q.get('output_hash') for q in ps],selected_codes=[q['selected_logit_half_bits'] for q in rs if 'selected_logit_half_bits' in q],token_sequences=[q['token_ids'] for q in rs if q.get('generation_sequence_complete')],dump_hashes={p.name:sha(p) for p in d.glob('*.bin') if p.name!='eval.bin'},physical_pass=True)
 if count==COUNT:assert len(ps)==repeat*16 and len(out['selected_codes'])==repeat*16 and len(out['token_sequences'])==repeat
 write(d/'validated.json',out);print('PASS',tag,'pre_us',out['prefill_ns']/1000,'dec_us',out['decode_ns']/1000,flush=True)
 return out

def compare(tags):
 zs=[read(R/t/'validated.json') for t in tags];a=zs[0]
 for b in zs[1:]:
  for k in ['output_hashes','selected_codes','token_sequences','dump_hashes']:assert a[k]==b[k],(tags,k)
 write(R/(tags[0].replace('/','_')+'-paired-gate.json'),dict(pass_all=True,tags=tags,bytes_exact=len(a['dump_hashes']),profiles=a['profiles']))
 print('EXACT_PAIRED_PASS',tags,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--count',type=int,default=1);ap.add_argument('--opt',type=int,default=0);ap.add_argument('--repeat',type=int,default=1);ap.add_argument('--tag');ap.add_argument('--audit',action='store_true');ap.add_argument('--tags',nargs='+');a=ap.parse_args()
 if a.action=='verify':verify()
 elif a.action=='stage':stage(a.count)
 elif a.action=='compare':compare(a.tags)
 else:run(a.count,a.opt,a.repeat,a.tag,a.audit)
