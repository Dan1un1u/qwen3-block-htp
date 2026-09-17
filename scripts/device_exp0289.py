"""EXP0271 provenance-sealed no-rotation FP32 residual device checks."""
import argparse,subprocess,shutil,shlex,statistics,json,struct,os
import numpy as np
from common_exp0289 import *
import exp0240_device as old
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
REMOTE='/data/local/tmp/qwen3-block-htp/exp0289'
def adb(*args,check=True):return subprocess.run([old.ADB,*args],check=check,capture_output=True,text=True,timeout=600)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def records(p):
 out=[]
 for line in Path(p).read_text().splitlines():
  try:out.append(json.loads(line))
  except ValueError:pass
 return out
def stage(count,trace=False):
 preflight();seal=read(S/'build/qwen3-sp2-build-seal.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();assert seal['source_head']==head;assert seal['paper_trace']==trace;assert seal['model_size']=='0.6B'
 for n,h in seal['files'].items():assert sha(n)==h
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  c=(S/b/'CMakeCache.txt').read_text();assert ('QBH_PAPER_TRACE:BOOL='+('ON' if trace else 'OFF')) in c;assert 'QBH_MODEL_LLAMA32:BOOL=OFF' in c and f'QBH_EXP0257_LAYER_COUNT:STRING={count}\n' in c
 a=1
 while (R/'binaries'/f'l{count}-a{a}').exists():a+=1
 d=R/'binaries'/f'l{count}-a{a}';d.mkdir(parents=True,exist_ok=False);remote=REMOTE+f'-l{count}-a{a}';assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 shutil.copy2(S/'build/qwen3-sp2-build-seal.json',d/'build-seal.json');adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli')
 (R/f'runtime-l{count}.json').write_text(json.dumps(dict(remote=remote,seal=seal,archive=str(d)),indent=2)+'\n');print('STAGED',count,flush=True)


def deploy(package):
 preflight();p=O/package;m=read(p/'manifest.json');remote=REMOTE+'-models/'+package
 for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
 assert adb('shell','test ! -e '+shlex.quote(remote),check=False).returncode==0
 adb('shell','mkdir -p '+shlex.quote(str(Path(remote).parent)));adb('push',win(p),remote)
 names=list(m['files'])
 for k in range(0,len(names),32):
  got=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[k:k+32])).stdout
  for line in got.splitlines():h,n=line.split(None,1);assert h==m['files'][n.removeprefix(remote+'/')]['sha256']
 write(R/('deployment-'+package+'.json'),dict(package=str(p),remote=remote,manifest_sha256=sha(p/'manifest.json'),files=len(names),verified=True));print('DEPLOY_PASS',package,flush=True)

ARGS='2 32 hvx on off fused gate8_interleaved control hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0'
def run(package,tag,count=1,repeat=1,audit=True,full=False,greedy=False):
 preflight();state=read(R/f'runtime-l{count}.json');root=state['remote'];d=R/tag;d.mkdir(parents=True,exist_ok=False)
 recipe='W4F16' if 'w4f16' in package else 'F16F16'
 remote=read(R/('deployment-'+package+'.json'))['remote']
 e=dict(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_VERTICAL_SLICE='1',QBH_REPLAY_SEQUENCE='1',QBH_SCAN_MODE='prefill',QBH_LOGICAL_M='64',QBH_KV_CACHE_LENGTH='0',QBH_KV_CACHE_CAPACITY='128',QBH_KV_CACHE_LAYOUT='hmx_native_f16',QBH_F16F16_OPT='3',QBH_W4F16_DECODE_OPT='2',QBH_REPLAY_DECODE_STEPS='1')
 args=ARGS
 if recipe=='W4F16':
  from device_exp0260 import ARGS as WARGS,ENV as WENV
  args=WARGS
  for k,value in WENV.items():
   if k not in e:e[k]=value
 if audit:
  target=root+'/'+tag.replace('/','_');adb('shell','mkdir -p '+target)
  e['QBH_GENERATION_AUDIT_DIR' if full else 'QBH_REPLAY_DUMP_DIR']=target
 if full:
  e.update(QBH_GENERATION_SEQUENCE='7' if recipe=='W4F16' else '10',QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='43')
  if audit:e['QBH_GENERATION_BOUNDARY_AUDIT']='1'
  ids=read(R/(package+'-reference.json'))['prompt_ids'];fixed=read(R/(package+'-reference.json'))['fixed_input_tokens']
  row=[0,2 if greedy else 3,43]+ids+fixed
  f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,2,repeat,len(row))+b''.join(struct.pack('<'+str(len(row))+'I',i,*row[1:]) for i in range(repeat)))
  rf=root+'/'+tag.replace('/','_')+'.bin';adb('push',win(f),rf);e['QBH_EVAL_FILE']=rf
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {remote} {recipe} {1 if full else repeat} {args}'
 write(d/'protocol.json',dict(runtime=state,command=cmd,package=package,repeat=repeat,full=full,greedy=greedy,audit=audit))
 z=adb('shell',cmd,check=False);(d/'stdout.jsonl').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode))
 if audit:adb('pull',target+'/.',win(d),check=False)
 if z.returncode:raise RuntimeError((tag,z.returncode,z.stderr[-2000:],z.stdout[-2500:]))
 rs=records(d/'stdout.jsonl');ps=[v for v in rs if v.get('record') in ['generation_profile','exp0240_profile','vertical_slice_replay_profile','replay_profile']]
 assert len(ps)==repeat*(43 if full else 2),(tag,len(ps))
 for v in ps:
  assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
  assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks'])
  assert v['block_invocation_count']==count
  zz=normalized([v]);assert sum(zz[k] for _,k in LEDGER)==v['invocation_ticks']
 errs=[]
 if audit:
  for step in range(43 if full else 2):
   f=d/(f'generation_hidden_norm_step{step:02d}_f16.bin' if full else f'step{step:02d}_output.bin')
   rows=1 if full or step else 64
   a=np.fromfile(f,'<f2').astype('f8')[:rows*1024]
   ref=O/package/(f'audit_hidden_{step:02d}_f16.bin' if full else f'reference_{recipe.lower()}_block_output_f16.bin' if not step else 'replay_decode_reference_00_f16.bin')
   b=np.fromfile(ref,'<f2').astype('f8')[:rows*1024]
   delta=a-b;cos=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30));err=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30))
   errs.append(dict(step=step,nrmse=err,cosine=cos,max_abs=float(abs(delta).max()),finite=bool(np.isfinite(a).all()),pass_all=bool(np.isfinite(a).all() and err<=.003 and cos>=.99999)))
 out=dict(physical_pass=True,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),independent_output=errs,numerical_pass=all(z['pass_all'] for z in errs) if errs else None,token_sequences=[v['token_ids'] for v in rs if v.get('generation_sequence_complete')])
 write(d/'validated.json',out);print('PASS',tag,out['prefill_ns']/1000,out['decode_ns']/1000,'numerical',out['numerical_pass'],flush=True)
 if audit and not greedy:assert out['numerical_pass'],errs
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=1);p.add_argument('--package');p.add_argument('--tag');p.add_argument('--repeat',type=int,default=1);p.add_argument('--audit',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--greedy',action='store_true');a=p.parse_args()
 if a.action=='stage':stage(a.count)
 elif a.action=='deploy':deploy(a.package)
 else:run(a.package,a.tag,a.count,a.repeat,a.audit,a.full,a.greedy)
