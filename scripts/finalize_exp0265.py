#!/usr/bin/env python3
"""Final immutable remote/native/model verification and evidence seal."""
from device_exp0265 import *
from datetime import datetime,timezone
import hashlib

def main():
 preflight();assert not (R/'EVIDENCE_SHA256.json').exists();z=read(R/'summary.json');assert z['timed_fullmodel_RPCs']==5280 and read(R/'independent_integrity.json')['pass_all'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();runtimes={}
 for count in [3,28]:
  runtime=read(read(R/f'runtime_l{count}.json')['manifest']);subprocess.run(['git','diff','--exit-code',runtime['source_head'],head,'--','src','include','CMakeLists.txt'],cwd=S,check=True);root=runtime['remote'];out=base.adb('shell',f'cd {root} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so');hs={x.split()[1]:x.split()[0] for x in out.splitlines()};assert hs==runtime['binaries'];local=Path(read(R/f'runtime_l{count}.json')['manifest']).parent
  for n,h in hs.items():assert sha(local/n)==h
  assert base.adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==runtime['boot'];runtimes[str(count)]=runtime
 manifests={};export=read(R/'export_audit.json')
 for name,path,remote in [('control',P,CONTROL),('r4',O/'r4',REMOTE)]:
  m=read(path/'manifest.json');assert sha(path/'manifest.json')==export['parent_manifest_sha256' if name=='control' else 'manifest_sha256']
  for n,v in m['files'].items():assert sha(path/n)==v['sha256']
  raw=base.adb('shell','cd '+remote+' && sha256sum '+' '.join(shlex.quote(n) for n in m['files']),timeout=600);hs={x.split(None,1)[1].strip():x.split()[0] for x in raw.splitlines()};assert all(hs[n]==v['sha256'] for n,v in m['files'].items());manifests[name]=dict(local=str(path),remote=remote,manifest_sha256=sha(path/'manifest.json'),files=m['files'])
 assert export['frozen_prefix_sha256'] in base.adb('shell','sha256sum '+SEED)
 for a,b in [('EXP-0265-RESULTS.md','REPORT.md'),('EXP-0265-PROFILE.md','FULL_PROFILING_REPORT.md')]:assert (S/'docs/experiments'/a).read_bytes()==(R/b).read_bytes()
 native={str(p.relative_to(S)):sha(p) for d in ['src','include'] for p in sorted((S/d).rglob('*')) if p.is_file()}
 write(R/'ARTIFACT_PROVENANCE.json',dict(experiment='EXP-0265',source_head=head,source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=S,text=True).strip(),runtimes=runtimes,native_file_hashes=native,packages=manifests,original_shards=export['original_shards'],prefix_sha256=export['frozen_prefix_sha256'],ABI=127,performance_policy='PC079 repeat10_only repeat1_auxiliary',source_and_remote_verified=True,model_quality='R4 speedfixture sample semantic failure; no PPL acceptance',device_PPL=None,baseline_promoted=False,closed_utc=datetime.now(timezone.utc).isoformat()))
 files={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file()};write(R/'EVIDENCE_SHA256.json',dict(experiment='EXP-0265',files=files))
 for n,v in files.items():assert sha(R/n)==v['sha256']
 print('SEALED',len(files),'files',sha(R/'EVIDENCE_SHA256.json'),'source',head,flush=True)
if __name__=='__main__':main()
