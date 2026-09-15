from pathlib import Path
import hashlib,json,subprocess,sys
S=Path('/home/daniuniu/work/qwen3-block-htp');R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0278');O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0271');OLD=O.parent/'exp0267';P=O.parent/'exp0257/package'
sys.path.insert(0,str(S/'tools'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def read(p):return json.loads(Path(p).read_text())
def preflight():
 z=subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],check=True,capture_output=True,text=True)
 assert 'EXPERIMENT=EXP-0278\n' in z.stdout

def package_path(name):
 if name in ['layer14-fp','full-fp']:return Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0278')/name
 return O/name if name=='sp2-fp32' else O.parent/'exp0269'/name
