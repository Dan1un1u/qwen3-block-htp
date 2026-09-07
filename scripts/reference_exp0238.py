#!/usr/bin/env python3
"""Pinned official Qronos dependency and common per-channel reference helpers."""
import os,json,hashlib,subprocess,sys
from pathlib import Path
S=Path(__file__).resolve().parents[1]
M=Path('/home/daniuniu/work/qwen3-block-htp-project-memory')
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0238')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0238')
V=Path('/home/daniuniu/.cache/qwen3-block-htp-qronos-py')
UP=Path('/home/daniuniu/.cache/qwen3-block-htp-brevitas-v0.13.0')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def write(n,d):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2,default=str);f.write('\n')
def preflight():subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
def setup():
 preflight();R.mkdir(parents=True,exist_ok=True)
 if not UP.exists():subprocess.run(['git','clone','--depth','1','--branch','v0.13.0','https://github.com/Xilinx/brevitas.git',str(UP)],check=True)
 assert subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True)==''
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
 if not V.exists():subprocess.run(['python3','-m','venv',str(V)],check=True)
 site=next((V/'lib').glob('python*/site-packages'))
 (site/'frozen_gpu_parent.pth').write_text(''.join('/home/daniuniu/.cache/qwen3-block-htp-'+x+'-py/lib/python3.10/site-packages\n' for x in ['autoround','spinquant'])+'/home/daniuniu/.cache/qwen3-block-htp-py/lib/python3.10/site-packages\n')
 subprocess.run([str(V/'bin/python'),'-m','pip','install','--no-deps','setuptools==69.5.1','setuptools-scm==8.1.0','wheel==0.45.1','tomli==2.2.1'],check=True)
 subprocess.run([str(V/'bin/python'),'-m','pip','install','--no-deps',str(UP),'dependencies==2.0.1','unfoldNd'],check=True)
 subprocess.run([str(V/'bin/python'),'-c','from brevitas.graph.qronos import Qronos;import torch,transformers;print(torch.__version__,transformers.__version__,Qronos)'],check=True)
 archive=R/'brevitas-v0.13.0.tar';subprocess.run(['git','archive','--format=tar','-o',str(archive),head],cwd=UP,check=True)
 write('upstream.json',dict(repository='https://github.com/Xilinx/brevitas',tag='v0.13.0',commit=head,archive_sha256=sha(archive),files={str(p.relative_to(UP/'src')):sha(p) for p in (UP/'src/brevitas').rglob('*.py')}))
 subprocess.run([str(V/'bin/python'),'-m','pip','freeze'],stdout=(R/'environment_freeze.txt').open('x'),check=True)
 print('QRONOS_SETUP_COMPLETE',head,flush=True)
def upstream_verified():
 import brevitas
 d=json.loads((R/'upstream.json').read_text());root=Path(brevitas.__file__).parent.parent
 for n,h in d['files'].items():assert sha(root/n)==h,n
 return d
if __name__=='__main__':setup()
