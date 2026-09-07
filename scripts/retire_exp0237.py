#!/usr/bin/env python3
"""Explicit user-authorized payload retirement. Never follows directory links."""
import os,stat,json,hashlib,subprocess,time,shutil,sys
from pathlib import Path
S=Path(__file__).resolve().parents[1]
M=Path('/home/daniuniu/work/qwen3-block-htp-project-memory')
B=Path('/mnt/d/llm_exp/models'); H=B/'qwen3-block-htp'
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0237')
PRESERVE=[B/'Qwen3-origin',H/'exp0217',H/'exp0230/C64',H/'exp0230/checkpoints/C64',H/'exp0233/AR-P',H/'exp0221/A',H/'exp0224/A']
# Earlier device packages (including all selected baselines) are outside retirement scope.
ENTIRE=[220,222,223,225,226,227,231,232,234,236]
TARGETS=[(H/f'exp{n:04}', 'obsolete_grouped_or_superseded_quantization') for n in ENTIRE]
TARGETS += [(H/'exp0219','obsolete_fold_rotation_payload'),(H/'exp0221','superseded_fold_rotation_and_export_caches'),(H/'exp0224','superseded_rotation_and_export_caches'),(H/'exp0230','superseded_C8_and_deployment_archives'),(H/'exp0233/AR-G','obsolete_grouped_reference')]
TARGETS += [(B/'Qwen3-1.7B-G32-base','legacy_mllm_grouped_weights'),(B/'qwen3_sm8750_v79','legacy_mllm_quantized_models')]
TARGETS += [(p,'legacy_mllm_weight_file') for p in sorted(B.glob('*.mllm'))]
EXT={'.bin','.npy','.npz','.pt','.pth','.safetensors','.onnx','.mllm','.pkl','.so'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def preflight():subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
def write(n,d):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def walk(root):
 if root.is_symlink():return
 if root.is_file():yield root;return
 stack=[root]
 while stack:
  d=stack.pop()
  with os.scandir(d) as entries:
   for e in entries:
    if e.is_symlink():continue
    if e.is_dir(follow_symlinks=False):
     p=Path(e.path)
     if not any(p==q for q in PRESERVE):stack.append(p)
    elif e.is_file(follow_symlinks=False):yield Path(e.path)
def protected_links():
 # Resolve every link under retained base packages; protect target files explicitly.
 protected=set()
 for n in [158,164,169,217,218,221,224,230,233]:
  root=H/f'exp{n:04}'
  for d,ds,fs in os.walk(root,followlinks=False):
   ds[:]=[x for x in ds if not Path(d,x).is_symlink()]
   for x in fs:
    p=Path(d,x)
    if p.is_symlink():
     try:protected.add(str(p.resolve(strict=True)))
     except OSError:pass
 return protected

def inventory():
 preflight();R.mkdir(parents=True,exist_ok=True);protected=protected_links();rows=[];seen=set();totals={}
 for root,reason in TARGETS:
  if not root.exists():continue
  base=root.resolve(strict=True);assert base==root and base.is_relative_to(B)
  for p in walk(root):
   if str(p) in seen:continue
   seen.add(str(p))
   if any(p==q or p.is_relative_to(q) for q in PRESERVE) or str(p) in protected:continue
   st=p.lstat()
   if st.st_size<1024**2 or not stat.S_ISREG(st.st_mode):continue
   if p.suffix not in EXT and not(p.suffix=='.tar' and ('deploy' in p.name or 'model' in p.name)):continue
   rows.append(dict(path=str(p),root=str(root),reason=reason,bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev,nlink=st.st_nlink))
   totals[reason]=totals.get(reason,0)+st.st_size
  print('INVENTORIED',root,len(rows),flush=True)
 write('inventory.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),free_before=shutil.disk_usage('/mnt/d').free,protected_roots=list(map(str,PRESERVE)),protected_symlink_targets=sorted(protected),files=rows,totals=totals,note='Filesystem metadata only. Byte sum counts hardlinks separately; actual reclaimed space measured independently. Historical hash ledgers preserved.'))
 print('INVENTORY_COMPLETE',len(rows),sum(x['bytes'] for x in rows),totals,flush=True)

def execute(expected_hash):
 preflight();assert sha(R/'inventory.json')==expected_hash
 inv=json.loads((R/'inventory.json').read_text());roots={str(p):why for p,why in TARGETS};protected=set(inv['protected_symlink_targets'])
 # Validate the complete plan before any deletion.
 for r in inv['files']:
  p=Path(r['path']);root=Path(r['root'])
  assert r['root'] in roots and roots[r['root']]==r['reason']
  assert root.resolve(strict=True)==root and root.is_relative_to(B)
  assert p==root or p.is_relative_to(root)
  assert not p.is_symlink() and p.resolve(strict=True)==p
  assert not any(p==q or p.is_relative_to(q) for q in PRESERVE) and str(p) not in protected
  st=p.lstat();assert stat.S_ISREG(st.st_mode)
  assert (st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)==(r['bytes'],r['mtime_ns'],r['inode'],r['device'])
 before=shutil.disk_usage('/mnt/d').free;done=0
 with (R/'deletions.jsonl').open('x') as log:
  for r in inv['files']:
   p=Path(r['path']);st=p.lstat()
   assert not p.is_symlink() and p.resolve(strict=True)==p and stat.S_ISREG(st.st_mode)
   assert (st.st_size,st.st_mtime_ns,st.st_ino)==(r['bytes'],r['mtime_ns'],r['inode'])
   p.unlink();assert not p.exists();log.write(json.dumps(dict(**r,deleted_at=time.time()))+'\n');log.flush();done+=1
   if done%100==0:print('RETIRED',done,len(inv['files']),flush=True)
 after=shutil.disk_usage('/mnt/d').free
 write('retirement.json',dict(inventory_sha256=expected_hash,deletion_ledger_sha256=sha(R/'deletions.jsonl'),files_deleted=done,logical_bytes=sum(r['bytes'] for r in inv['files']),free_before=before,free_after=after,actual_free_increase=after-before,all_planned_paths_absent=all(not Path(r['path']).exists() for r in inv['files']),historical_results_and_hashes_unchanged=True))
 print('RETIREMENT_COMPLETE',done,'FREE_INCREASE_GIB',(after-before)/1024**3,flush=True)
if __name__=='__main__':
 inventory() if sys.argv[1]=='inventory' else execute(sys.argv[2])
