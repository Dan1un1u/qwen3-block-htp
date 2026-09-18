"""L32-0052 isolated long-prefill fixtures, build deployment and immutable runs."""
import os,sys,json,shlex,subprocess,shutil,hashlib,argparse
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'tools'))
from run_llama32_layer import adb,windows
from llama_reference import sha256 as sha
from run_llama32_frontend import records
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0052')
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0052')
BASE=R/'base.json'
REMOTE='/data/local/tmp/llama32-htp/l32-0052'
def read(p):return json.loads(Path(p).read_text())
def put(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'ACTIVE_EXPERIMENT=L32-0052' in z
def prepare():
 preflight()
 import torch
 from transformers import AutoTokenizer
 from llama_reference import rope
 base=read(BASE);src=Path(base['package']);assert sha(src/'manifest.json')==base['manifest_sha256']
 mf=read(src/'manifest.json')
 for n,v in mf['files'].items():assert sha(src/n)==v['sha256'],n
 M.mkdir(exist_ok=True)
 assert not any(M.iterdir()), 'Refuse to overwrite prepared fixtures'
 original=Path('/mnt/d/llm_exp/models/llama3.2-3B-Instruct-origin')
 cfg=read(original/'config.json');cfg['head_dim']=128
 tok=AutoTokenizer.from_pretrained(original,local_files_only=True)
 text=("This is a fixed runtime validation passage. A library keeps books in labelled rooms. "
       "Visitors follow directions, compare dates and read short notes. The river runs east of the town. "
       "We test continuous context, without resetting positions or forgetting earlier words. ")
 ids=np.fromfile(src/'generation_prompt_token_ids_u32.bin','<u4').tolist()
 ids+=tok.encode(text*40,add_special_tokens=False)
 ids=ids[:768];fixed=np.fromfile(src/'generation_expected_token_ids_u32.bin','<u4').tolist()
 fixed=(fixed*4)[:64]
 cos,sin=rope(cfg,torch.arange(832)[None],torch.float16)
 cos=cos.cpu().numpy().reshape(832,128);sin=sin.cpu().numpy().reshape(832,128)
 assert np.array_equal(cos[:64],np.fromfile(src/'rope_cos_f16.bin','<f2').reshape(64,128))
 assert np.array_equal(sin[:64],np.fromfile(src/'rope_sin_f16.bin','<f2').reshape(64,128))
 put(R/'fixture.json',dict(prompt_ids=ids,fixed=fixed,text=text,original_config_sha256=sha(original/'config.json'),tokenizer_sha256=sha(original/'tokenizer.json'),purpose='project-owned correctness and timing fixture, not named dataset'))
 for length in [64,65,128,129,536,741]:
  dst=M/str(length);dst.mkdir();changed=[]
  for n,v in mf['files'].items():
   if n.endswith('.npy'):continue
   target=dst/n;target.parent.mkdir(parents=True,exist_ok=True)
   if 'kv_cache_' in Path(n).name:
    a=np.fromfile(src/n,'u1').reshape(8,-1,128);z=np.zeros((8,832,128),'u1');z[:,:a.shape[1]]=a;z.tofile(target);changed.append(n)
   elif n=='generation_expected_token_ids_u32.bin':
    np.array(fixed,'<u4').tofile(target);changed.append(n)
   else:os.link(src/n,target)
  np.array(ids[:length],'<u4').tofile(dst/'long_prompt_u32.bin')
  np.array(fixed,'<u4').tofile(dst/'long_fixed_u32.bin')
  cos.astype('<f2').tofile(dst/'long_rope_cos_f16.bin');sin.astype('<f2').tofile(dst/'long_rope_sin_f16.bin')
  files={str(p.relative_to(dst)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in dst.rglob('*') if p.is_file()}
  put(dst/'manifest.json',dict(experiment='L32-0052',parent=base,parent_sha256=base['manifest_sha256'],files=files,changed=changed))
  remote=f'{REMOTE}/models/{length}'
  assert adb('shell','test ! -e '+remote,check=False).returncode==0
  names=list(files);dirs=sorted({str(Path(n).parent) for n in names})
  adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs))
  links=[n for n in names if n in mf['files'] and files[n]['sha256']==mf['files'][n]['sha256']]
  for i in range(0,len(links),24):
   adb('shell',' && '.join('ln -s '+shlex.quote(base['remote']+'/'+n)+' '+shlex.quote(remote+'/'+n) for n in links[i:i+24]))
  for n in set(names)-set(links):adb('push',windows(dst/n),remote+'/'+n)
  for i in range(0,len(names),32):
   lines=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines()
   assert len(lines)==len(names[i:i+32])
   for line in lines:
    h,n=line.split(None,1);assert h==files[n.removeprefix(remote+'/')]['sha256'],n
  put(R/f'package-{length}.json',dict(package=str(dst),remote=remote,manifest_sha256=sha(dst/'manifest.json')))
  print('PREPARED',length,flush=True)
def stage():
 preflight();seal=read(S/'build/llama-build-seal.json')
 assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 assert seal['model_size']=='3B' and str(seal['layer_count'])=='28'
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
 preflight();base=read(BASE);cfg=base if legacy else read(R/f'package-{length}.json');rt=read(R/'runtime.json');root=rt['remote']
 assert rt['seal']['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 d=R/tag;d.mkdir(parents=True,exist_ok=False)
 prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
 env=dict(x.split('=',1) for x in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=cfg['remote']
 env.pop('QBH_EVAL_FILE',None)
 env.update(QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='8',QBH_LONG_OPT=os.environ.get('QBH_LONG_OPT','31'),LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0')
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
