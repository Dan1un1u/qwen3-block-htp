#!/usr/bin/env python3
"""Pinned official AutoRound reference setup; numerical stages appended separately."""
import argparse,json,os,subprocess,sys,time,urllib.request,hashlib
from pathlib import Path
S=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0233')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0233')
V=Path('/home/daniuniu/.cache/qwen3-block-htp-autoround-py')
UP=Path('/home/daniuniu/.cache/qwen3-block-htp-autoround-v0.5.1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def write(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def preflight():
 subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],check=True)
def setup():
 preflight()
 if not UP.exists():subprocess.run(['git','clone','--depth','1','--branch','v0.5.1','https://github.com/intel/auto-round.git',str(UP)],check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
 assert subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True)==''
 if not V.exists():subprocess.run(['python3','-m','venv',str(V)],check=True)
 site=next((V/'lib').glob('python*/site-packages'))
 (site/'frozen_gpu_parent.pth').write_text('/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/lib/python3.10/site-packages\n/home/daniuniu/.cache/qwen3-block-htp-py/lib/python3.10/site-packages\n')
 subprocess.run([str(V/'bin/python'),'-m','pip','install','--no-deps',str(UP),'numpy==1.26.4','py-cpuinfo==9.0.0','pillow==11.3.0','threadpoolctl==3.6.0'],check=True)
 subprocess.run([str(V/'bin/python'),'-c','import auto_round,torch,transformers;print(auto_round.__version__,torch.__version__,transformers.__version__)'],check=True)
 archive=O/'upstream';archive.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','archive','--format=tar','-o',str(archive/'auto-round-v0.5.1.tar'),head],cwd=UP,check=True)
 write('upstream.json',dict(repository='https://github.com/intel/auto-round',tag='v0.5.1',commit=head,archive_sha256=sha(archive/'auto-round-v0.5.1.tar'),license_sha256=sha(UP/'LICENSE'),files={str(p.relative_to(UP)):sha(p) for p in sorted((UP/'auto_round').rglob('*.py'))}))
 subprocess.run([str(V/'bin/python'),'-m','pip','freeze'],stdout=(R/'environment_freeze.txt').open('x'),check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['setup']);a=p.parse_args();setup()
