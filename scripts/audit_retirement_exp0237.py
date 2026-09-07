#!/usr/bin/env python3
"""Bounded secondary legacy payload cleanup and retained-package verification."""
import os,json,stat,shutil,sys
from pathlib import Path
from retire_exp0237 import preflight,sha,write,R,H,B
LEGACY=Path('/mnt/d/llm_exp/mllm-v2/artifacts')
def legacy_inventory():
 preflight();assert LEGACY.resolve(strict=True)==LEGACY
 rows=[]
 for d,ds,fs in os.walk(LEGACY,followlinks=False):
  ds[:]=[n for n in ds if not Path(d,n).is_symlink()]
  for n in fs:
   p=Path(d,n)
   if p.is_symlink():continue
   t=p.stat()
   if t.st_size>=1048576 and p.suffix in {'.safetensors','.bin','.npy','.npz','.pt','.pth','.mllm','.onnx'}:
    assert p.resolve(strict=True)==p
    rows.append(dict(path=str(p),bytes=t.st_size,mtime_ns=t.st_mtime_ns,inode=t.st_ino,device=t.st_dev,reason='legacy_mllm_v2_quantized_weight_payload'))
 write('legacy_inventory.json',dict(root=str(LEGACY),files=rows,logical_bytes=sum(r['bytes'] for r in rows)))
 print('LEGACY_INVENTORY',len(rows),sum(r['bytes'] for r in rows),sha(R/'legacy_inventory.json'),flush=True)
def legacy_execute(expected):
 preflight();assert sha(R/'legacy_inventory.json')==expected
 inv=json.loads((R/'legacy_inventory.json').read_text());assert LEGACY.resolve(strict=True)==LEGACY
 def check(r):
  p=Path(r['path']);assert p.is_relative_to(LEGACY) and p.resolve(strict=True)==p and not p.is_symlink()
  t=p.lstat();assert stat.S_ISREG(t.st_mode) and (t.st_size,t.st_mtime_ns,t.st_ino,t.st_dev)==(r['bytes'],r['mtime_ns'],r['inode'],r['device'])
 for r in inv['files']:check(r)
 before=shutil.disk_usage('/mnt/d').free
 with (R/'legacy_deletions.jsonl').open('x') as f:
  for r in inv['files']:
   check(r);Path(r['path']).unlink();f.write(json.dumps(r)+'\n');f.flush()
 after=shutil.disk_usage('/mnt/d').free
 write('legacy_retirement.json',dict(files_deleted=len(inv['files']),inventory_sha256=expected,ledger_sha256=sha(R/'legacy_deletions.jsonl'),free_before=before,free_after=after,actual_free_increase=after-before,all_planned_paths_absent=all(not Path(r['path']).exists() for r in inv['files'])))
 print('LEGACY_RETIREMENT',after-before,flush=True)
def audit():
 preflight();out=[];cache={}
 def digest(p):
  t=p.stat();k=(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns)
  if k not in cache:cache[k]=sha(p)
  return cache[k]
 rbase=R.parent
 ledger=json.loads((rbase/'exp0218/original_checkpoint_sha256.json').read_text())
 for n,h in ledger.items():assert digest(B/'Qwen3-origin'/n)==h
 out.append(dict(root=str(B/'Qwen3-origin'),files=len(ledger),all_hashes_match=True))
 roots=['exp0217/f16f16_greedy16','exp0230/C64','exp0233/AR-P/attempt2','exp0158/f16f16','exp0158/w4f16','exp0164/w4f16_greedy16','exp0169/w4u8_greedy193_overlay','exp0167/w4u8_greedy16','exp0163/candidate_segmented_capacity257','exp0221/A','exp0224/A']
 for n in roots:
  p=H/n;m=json.loads((p/'manifest.json').read_text());old=m['files'];files={k.replace(chr(92),'/'):old.get(k.replace(chr(92),'/'),v) for k,v in old.items()}
  bad=[]
  for k,v in files.items():
   try:
    if digest(p/k)!=v['sha256']:bad.append(dict(path=k,error='hash_mismatch'))
   except OSError as e:bad.append(dict(path=k,error=str(e)))
  out.append(dict(root=str(p),manifest_sha256=sha(p/'manifest.json'),files=len(files),all_hashes_match=not bad,errors=bad))
  print('AUDITED',n,len(files),'errors',len(bad),flush=True)
 write('retained_artifact_audit.json',dict(packages=out,all_hashes_match=all(x['all_hashes_match'] for x in out),unique_payloads_hashed=len(cache)))
 assert all(x['all_hashes_match'] for x in out),out
if __name__=='__main__':
 {'inventory':legacy_inventory,'execute':lambda:legacy_execute(sys.argv[2]),'audit':audit}[sys.argv[1]]()
