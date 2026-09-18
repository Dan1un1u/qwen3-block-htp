"""L32-0051 isolated long-prefill fixtures, build deployment and immutable runs."""
import os,sys,json,shlex,subprocess,shutil,hashlib,argparse
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'tools'))
from run_llama32_layer import adb,windows
from llama_reference import sha256 as sha
from run_llama32_frontend import records
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0051')
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0051')
BASE=R.parent/'l32-0040/package-full.json'
REMOTE='/data/local/tmp/llama32-htp/l32-0051'
def read(p):return json.loads(Path(p).read_text())
def put(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'ACTIVE_EXPERIMENT=L32-0051' in z
def stage():
 preflight();seal=read(S/'build/llama-build-seal.json')
 assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 assert seal['model_size']=='1B' and str(seal['layer_count'])=='16'
 i=1
 while (R/f'binaries-a{i}').exists():i+=1
 dst=R/f'binaries-a{i}';dst.mkdir();remote=f'{REMOTE}/binaries-a{i}'
 adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h;shutil.copy2(p,dst/p.name);adb('push',windows(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli')
 put(dst/'seal.json',seal);put(R/'runtime.json',dict(remote=remote,seal=seal,archive=str(dst)))
 print('STAGED',remote,flush=True)
def run(length,tag,decode=3,audit=False,repeat=1,legacy=False):
 preflight();base=read(BASE);cfg=base if legacy else read(R.parent/f'l32-0050/package-{length}.json');rt=read(R.parent/'l32-0050/runtime.json' if os.environ.get('QBH_LONG_CONTROL') else R/'runtime.json');root=rt['remote']
 assert os.environ.get('QBH_LONG_CONTROL') or rt['seal']['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 d=R/tag;d.mkdir(parents=True,exist_ok=False)
 prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
 env=dict(x.split('=',1) for x in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=cfg['remote']
 env.update(QBH_LONG_OPT=os.environ.get('QBH_LONG_OPT','1'),LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0')
 if not legacy:env.update(QBH_LONG_PREFILL_TOKENS=str(length),QBH_LONG_DECODE_STEPS=str(decode),QBH_LONG_REPEATS=str(repeat),QBH_KV_CACHE_CAPACITY='832',QBH_GENERATION_EXPECTED_TOKENS='64')
 else:env['QBH_GENERATION_STEPS']='4'
 if os.environ.get('QBH_LONG_DIAGNOSTIC'):env['QBH_LONG_DIAGNOSTIC']='1'
 if os.environ.get('QBH_LONG_PADDING_POISON'):env['QBH_LONG_PADDING_POISON']='1'
 if audit:
  env['QBH_GENERATION_BOUNDARY_AUDIT']='1';env['QBH_GENERATION_AUDIT_DIR']=root+'/'+tag.replace('/','_')
  adb('shell','mkdir -p '+env['QBH_GENERATION_AUDIT_DIR'])
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
 put(d/'protocol.json',dict(command=cmd,package=cfg,runtime=rt,length=length,decode=decode,repeat=repeat,audit=audit))
 z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);put(d/'exit.json',dict(returncode=z.returncode))
 if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'),check=False)
 rr=records(z.stdout);put(d/'records.json',rr)
 print('RUN',tag,'EXIT',z.returncode,flush=True)
 if z.returncode:print(z.stdout[-1300:],z.stderr[-1800:],flush=True)
 else:print([x for x in rr if isinstance(x,dict) and x.get('record')=='long_complete'],flush=True)
 return z.returncode
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('--length',type=int,default=64);a.add_argument('--tag',default='check64-a01');a.add_argument('--decode',type=int,default=3);a.add_argument('--repeat',type=int,default=1);a.add_argument('--audit',action='store_true');a.add_argument('--legacy',action='store_true');v=a.parse_args()
 if v.action=='run':sys.exit(run(v.length,v.tag,v.decode,v.audit,v.repeat,v.legacy))
 else:globals()[v.action]()
