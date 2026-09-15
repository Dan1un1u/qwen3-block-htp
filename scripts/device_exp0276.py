"""EXP0271 provenance-sealed no-rotation FP32 residual device checks."""
import argparse,subprocess,shutil,shlex,statistics,json,struct,os
import numpy as np
from common_exp0276 import *
import exp0240_device as old
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
REMOTE='/data/local/tmp/qwen3-block-htp/exp0276'
def adb(*args,check=True):return subprocess.run([old.ADB,*args],check=check,capture_output=True,text=True,timeout=600)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def records(p):
 out=[]
 for line in Path(p).read_text().splitlines():
  try:out.append(json.loads(line))
  except ValueError:pass
 return out
def stage(count,trace=False):
 preflight();seal=read(S/'build/qwen3-sp2-build-seal.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();assert seal['source_head']==head;assert seal['paper_trace']==trace
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
 adb('shell','mkdir -p '+REMOTE+'-models');adb('push',win(p),remote)
 names=list(m['files'])
 for k in range(0,len(names),32):
  got=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[k:k+32])).stdout
  for line in got.splitlines():h,n=line.split(None,1);assert h==m['files'][n.removeprefix(remote+'/')]['sha256']
 write(R/('deployment-'+package+'.json'),dict(package=str(p),remote=remote,manifest_sha256=sha(p/'manifest.json'),files=len(names),verified=True));print('DEPLOY_PASS',package,flush=True)

def run(package,tag,count=1,repeat=1,fp32=1,dump=False):
 preflight();runtime=read(R/f'runtime-l{count}.json');root=runtime['remote'];d=R/tag;d.mkdir(parents=True,exist_ok=False)
 remote=read(R/('deployment-'+package+'.json'))['remote']
 env=dict(old.ENV,LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_SP2='8',QBH_U8_PREFILL_OPT='0',QBH_FP32_RESIDUAL=str(fp32),QBH_PAPER_FORMAT_DISABLE=os.environ.get('QBH_PAPER_FORMAT_DISABLE','0'),QBH_PAPER_PIPELINE_DISABLE=os.environ.get('QBH_PAPER_PIPELINE_DISABLE','0'),QBH_WIDE_SCORE='4',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_KV_CACHE_CAPACITY='128',QBH_REPLAY_DECODE_STEPS='1')
 if dump:
  env['QBH_REPLAY_DUMP_DIR']=root+'/'+tag.replace('/','_');env['QBH_DENSE_R3_AUDIT']='1';adb('shell','mkdir '+env['QBH_REPLAY_DUMP_DIR'])
 command='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+f' ./qwen3_block_cli {remote} W4U8 {repeat} {old.ARGS}'
 write(d/'protocol.json',dict(source_head=runtime['seal']['source_head'],runtime=runtime,command=command,package=package,fp32_residual=fp32,repeat=repeat))
 result=adb('shell',command,check=False);(d/'stdout.jsonl').write_text(result.stdout);(d/'stderr.txt').write_text(result.stderr);write(d/'exit.json',dict(returncode=result.returncode))
 if dump:adb('pull',env['QBH_REPLAY_DUMP_DIR']+'/.',win(d),check=False)
 if result.returncode:raise RuntimeError((tag,result.returncode,result.stderr[-1800:],result.stdout[-2000:]))
 ps=[v for v in records(d/'stdout.jsonl') if v.get('record')=='exp0240_profile'];assert len(ps)==repeat*2,(len(ps),repeat)
 for v in ps:
  assert v['paper_format_disable']==int(os.environ.get('QBH_PAPER_FORMAT_DISABLE','0')) and v['paper_pipeline_disable']==int(os.environ.get('QBH_PAPER_PIPELINE_DISABLE','0'))
  assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
  for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks']:assert v[k]==0,(k,v[k])
  assert v['block_invocation_count']==count and v['dense_r3_mode']==v['dense_r4_mode']==0 and v['wide_score_mode']==4
  z=normalized([v]);assert sum(z[k] for _,k in LEDGER)==v['invocation_ticks']
 z=dict(physical_pass=True,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),output_hashes=[v['output_hash'] for v in ps],peak=max(v['vtcm_peak_plan_bytes'] for v in ps))
 assert all(v==z['output_hashes'][i%2] for i,v in enumerate(z['output_hashes']))
 if dump and fp32:
  errs=[]
  for step in range(2):
   rows=64 if not step else 1;a=np.fromfile(d/f'step{step:02d}_output.bin','<f4').reshape(rows,2048);b=np.fromfile(package_path(package)/('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin'),'<f4').reshape(64,2048)[:rows];delta=a.astype('f8')-b
   errs.append(dict(step=step,mismatches=int(np.count_nonzero(delta)),max_abs=float(np.abs(delta).max()),finite=bool(np.isfinite(a).all()),nrmse=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30))))
  z['independent_output']=errs;z['numerical_pass']=all(x['mismatches']==0 and x['finite'] for x in errs)
 write(d/'validated.json',z);print('RUN',tag,json.dumps(z),flush=True)
 if dump and fp32:assert z['numerical_pass'],z['independent_output']
 return z
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=1);p.add_argument('--package',default='layer0-fp32-a01');p.add_argument('--tag',default='layer0-a01');p.add_argument('--repeat',type=int,default=1);p.add_argument('--fp32',type=int,default=1);p.add_argument('--dump',action='store_true');a=p.parse_args()
 if a.action=='stage':stage(a.count)
 elif a.action=='deploy':deploy(a.package)
 else:run(a.package,a.tag,a.count,a.repeat,a.fp32,a.dump)
