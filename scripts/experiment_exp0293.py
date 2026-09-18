"""EXP0293: matched full-M64 floating decode rows; frozen0292 references."""
import os,sys,json,subprocess,hashlib,shutil
from pathlib import Path
import numpy as np
import measure_exp0292 as m
import check_exp0292_components as component
d=m.d;S=d.S;OLD=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0292')
R=OLD.parent/'exp0293'
d.R=m.R=R
d.REMOTE='/data/local/tmp/qwen3-block-htp/exp0293'
def preflight():
 z=subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],capture_output=True,text=True,check=True)
 assert 'EXPERIMENT=EXP-0293\n' in z.stdout
d.preflight=preflight
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def initialize():
 preflight();R.mkdir(exist_ok=False)
 ledger=OLD/'EVIDENCE_SHA256.json'
 assert sha(ledger)=='4e5745d6e160c39583e70b1569d681cdac6c289cd60cb07d0f6fdb317f5abdcd'
 entries=json.loads(ledger.read_text())['files']
 for n,x in entries.items():
  p=OLD/n;assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],n
 write(R/'prior-evidence-verified.json',dict(pass_all=True,ledger_sha256=sha(ledger),files=len(entries),bytes=sum(x['bytes'] for x in entries.values())))
 for p in list(OLD.glob('deployment-*.json'))+[OLD/'f16f16-reference.json']:
  shutil.copy2(p,R/p.name)
 # Frozen payloads remain in their original directories and on device; verify both.
 packages=[(d.O,p) for p in ['f16f16','f16f16-layer0','f16f16-layer14','f16f16-layer27','f16f16-chain3']]
 packages.append((d.O.parent/'exp0289','f16f16'))
 checks=[];cache={}
 for root,name in packages:
  package=root/name;manifest=m.read(package/'manifest.json')
  deployment=m.read((OLD if root==d.O else OLD.parent/'exp0289')/('deployment-'+name+'.json'))
  remote=deployment['remote'];names=list(manifest['files'])
  for n,x in manifest['files'].items():
   p=package/n;stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
   if key not in cache:cache[key]=sha(p)
   assert cache[key]==x['sha256'],p
  for i in range(0,len(names),32):
   text=d.adb('shell','sha256sum '+' '.join(d.shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout
   rows=text.splitlines();assert len(rows)==len(names[i:i+32])
   for line in rows:
    h,n=line.split(None,1);assert h==manifest['files'][n.removeprefix(remote+'/')]['sha256'],n
  checks.append(dict(package=str(package),remote=remote,files=len(names),manifest_sha256=sha(package/'manifest.json')))
 write(R/'inputs-verified.json',dict(pass_all=True,packages=checks))
 print('INITIALIZATION_PASS',flush=True)
def profile_rows(path,count):
 ps=[x for x in d.records(path/'stdout.jsonl') if x.get('record') in ['generation_profile','exp0240_profile','vertical_slice_replay_profile','replay_profile']]
 for x in ps:
  rows_per=x['fp16_norm_rows_per_task'];expected=((64+rows_per-1)//rows_per)*count
  assert x['fp16_input_norm_task_count']==expected,(path,'input',x['fp16_input_norm_task_count'],expected)
  assert x['fp16_post_residual_norm_task_count']==expected,(path,'post',x['fp16_post_residual_norm_task_count'],expected)
  assert x['fp16_input_norm_active_contexts']==x['fp16_post_residual_norm_active_contexts']==x['fp16_norm_contexts']
 return dict(pass_all=True,profiles=len(ps),rows=64,scope='both input and fused post Norm; final add source count M64*hidden',tasks_per_profile=expected)
def selected(layer):
 tag='selected-layer'+str(layer);package='f16f16-layer'+str(layer)
 d.run(package,tag,count=1,audit=True,component=True)
 verify_replay(tag,'single'+str(layer)+'-components-a1',1)
def chain():
 d.run('f16f16-chain3','chain3',count=3,audit=True,component=True)
 verify_replay('chain3','chain3-components-a1',3)
def verify_replay(tag,oldtag,count):
 root=R/tag
 equality=[]
 for step,rows in [(0,64),(1,1)]:
  name=f'step{step:02d}_output.bin';size=rows*1024*4
  assert (root/name).read_bytes()[:size]==(OLD/oldtag/name).read_bytes()[:size],name
  equality.append(dict(step=step,valid_rows=rows,bytes=size))
 component.check(root)
 write(root/'row-parity.json',dict(schedule=profile_rows(root,count),valid_output_exact=equality,reference=str(OLD/oldtag)))
 print('SELECTED_EXACT',tag,flush=True)
def audit(case):
 out=m.one(case,'audit/'+case,1,True)
 root=R/'audit'/case;reference=OLD/'audit'/case
 files=sorted(reference.glob('generation*bin'));assert len(files)==2451
 for p in files:assert sha(root/p.name)==sha(p),p.name
 assert m.signatures(root)==m.signatures(reference)
 write(root/'row-parity.json',dict(pass_all=True,byte_exact_files=len(files),reference=str(reference),schedule=profile_rows(root,28)))
 if case=='fp32':
  oldhead=m.read(OLD/'head-boundary-checks.json');assert oldhead['pass_all']
  write(R/'head-boundary-checks.json',dict(pass_all=True,reference=str(OLD/'head-boundary-checks.json'),reference_sha256=sha(OLD/'head-boundary-checks.json'),reuse_reason='All43 actual final hidden/norm bytes and selected ID/logit codes match frozen independently checked0292 operands exactly; no new head arithmetic.',fresh_reference_evaluation=False))
 print('FULL_EXACT',case,len(files),flush=True)
def one(case,tag,repeat):
 out=m.one(case,tag,repeat)
 write(R/tag/'row-parity.json',profile_rows(R/tag,28))
 return out
def campaign():
 for case in ['fp16','fp32']:assert m.read(R/'audit'/case/'row-parity.json')['pass_all']
 write(R/'repeat1.json',[one(c,'repeat1/'+c,1) for c in ['fp16','fp32']])
 for phase,n in [('short',5),('formal',10)]:
  rows=[]
  for i in range(n):
   seq=['fp16','fp32'] if i%2==0 else ['fp32','fp16']
   block=[one(c,f'{phase}/{i:02d}-{c}',10) for c in seq]
   rows+=block;write(R/f'{phase}-round-{i:02d}.json',block)
   print('ROUND_DONE',phase,i,flush=True)
  write(R/(phase+'.json'),rows)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='init':initialize()
 elif action=='stage':d.stage(int(sys.argv[2]))
 elif action=='selected':selected(int(sys.argv[2]))
 elif action=='chain':chain()
 elif action=='audit':audit(sys.argv[2])
 elif action=='campaign':campaign()
