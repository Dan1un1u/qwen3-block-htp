"""EXP0268 immutable package reuse, isolated results and authorized preflight."""
import sys,json,hashlib,subprocess
from pathlib import Path
S=Path('/home/daniuniu/work/qwen3-block-htp');R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0268');OLD=R.parent/'exp0267'
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0267');P=O.parent/'exp0257/package'
sys.path.insert(0,str(S/'tools'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
def preflight():
 r=subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],check=True,capture_output=True,text=True)
 assert 'EXPERIMENT=EXP-0268\n' in r.stdout,r.stdout
