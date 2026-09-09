#!/usr/bin/env python3
"""Seal EXP0263 after source/report closure and immutable remote verification."""
from device_exp0263 import *
from datetime import datetime,timezone
import hashlib

def main():
 preflight();assert not (R/'EVIDENCE_SHA256.json').exists()
 z=read(R/'summary.json');assert z['timed_RPCs']==2970 and z['selected_opt']==3
 assert read(R/'independent_integrity.json')['pass_all']
 runtime=read(read(R/'runtime_l1.json')['manifest']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
 subprocess.run(['git','diff','--exit-code',runtime['source_head'],head,'--','src','include','CMakeLists.txt'],cwd=S,check=True)
 root=runtime['remote'];got=base.adb('shell',f'cd {root} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
 hashes={x.split()[1]:x.split()[0] for x in got.splitlines()}
 assert hashes==runtime['binaries']
 local=Path(read(R/'runtime_l1.json')['manifest']).parent
 for n,h in hashes.items():assert sha(local/n)==h
 boot=base.adb('shell','cat /proc/sys/kernel/random/boot_id').strip();assert boot==runtime['boot']
 manifests={}
 for name,path,remote in [('control',P,PARENT),('r4',O/'r4',REMOTE)]:
  m=read(path/'manifest.json');export=read(R/'export_audit.json');assert sha(path/'manifest.json')==export['parent_manifest_sha256' if name=='control' else 'manifest_sha256']
  for n,v in m['files'].items():assert sha(path/n)==v['sha256'],n
  got=base.adb('shell','cd '+remote+' && sha256sum '+' '.join(shlex.quote(n) for n in m['files']))
  hs={x.split(None,1)[1].strip():x.split()[0] for x in got.splitlines()}
  assert all(hs[n]==v['sha256'] for n,v in m['files'].items())
  manifests[name]=dict(local=str(path),remote=remote,manifest_sha256=sha(path/'manifest.json'),files=m['files'])
 for file,report in [('EXP-0263-RESULTS.md','REPORT.md'),('EXP-0263-PROFILE.md','FULL_PROFILING_REPORT.md')]:assert (S/'docs/experiments'/file).read_bytes()==(R/report).read_bytes()
 native={str(p.relative_to(S)):sha(p) for d in ['src','include'] for p in sorted((S/d).rglob('*')) if p.is_file()}
 provenance=dict(experiment='EXP-0263',source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=S,text=True).strip(),source_head=head,runtime=runtime,native_file_hashes=native,packages=manifests,boot=boot,ABI=126,parent_exp0262_head='53dde49331bb1b7d256924a24aa335fff02a1428',parent_exp0262_evidence='d526f6193fabb6b4cd7279c55907b06c2fcebcbcf9a875ec934c6e55fe2fe1de',source_and_remote_verified=True,device_PPL=None,E2E=None,baseline_promoted=False,closed_utc=datetime.now(timezone.utc).isoformat())
 write(R/'ARTIFACT_PROVENANCE.json',provenance)
 files={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file()}
 write(R/'EVIDENCE_SHA256.json',dict(experiment='EXP-0263',files=files))
 for n,v in files.items():assert sha(R/n)==v['sha256']
 print('SEALED',len(files),'files',sha(R/'EVIDENCE_SHA256.json'),'source',head,flush=True)
if __name__=='__main__':main()
