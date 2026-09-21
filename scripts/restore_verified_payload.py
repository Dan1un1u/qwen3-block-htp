#!/usr/bin/env python3
"""Restore only hash-verified archived payloads, without changing old manifests."""
import argparse,json,hashlib,tarfile,subprocess,shutil
from pathlib import Path

def run(remote,manifest,dest,report):
 m=json.loads(Path(manifest).read_text());files=m['files'];dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
 adb=['/mnt/c/adb/adb.exe','-s','3B15C8007Z300000'];restored=[];unknown=[]
 with Path(str(report)+'.stderr').open('wb') as err:
  proc=subprocess.Popen(adb+['exec-out','tar -chf - -C '+remote+' .'],stdout=subprocess.PIPE,stderr=err)
  with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
   for item in tar:
    if not item.isfile():continue
    name=item.name.removeprefix('./');relative=Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts,name
    if name not in files:unknown.append(name);continue
    expected=files[name];expected=expected['sha256'] if isinstance(expected,dict) else expected
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
    with tar.extractfile(item) as inp,target.open('xb') as out:
     while chunk:=inp.read(8<<20):h.update(chunk);out.write(chunk)
    assert h.hexdigest()==expected,(name,h.hexdigest(),expected)
    restored.append(name)
  assert proc.wait()==0
 shutil.copy2(manifest,dest/'manifest.json')
 missing=sorted(set(files)-set(restored));Path(report).write_text(json.dumps(dict(remote=remote,manifest=str(manifest),manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),destination=str(dest),restored=restored,not_on_device=missing,unlisted_skipped=unknown),indent=2)+'\n');print('RESTORED',len(restored),'MISSING',len(missing),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('remote');p.add_argument('manifest');p.add_argument('dest');p.add_argument('report');a=p.parse_args();run(a.remote,a.manifest,a.dest,a.report)
