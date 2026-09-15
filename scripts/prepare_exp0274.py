"""Reverify frozen package provenance; never inherit numerical gates."""
from common_exp0274 import *
from device_exp0274 import adb
import shlex
PRIOR=R.parent/'exp0272'
def prepare():
 preflight()
 ledger=PRIOR/'EVIDENCE_SHA256.json'
 assert sha(ledger)=='7fa56656cbc385a2733146df45d1c5b399a302a0a16898e9fc135ae496016d61'
 entries=read(ledger)['files']
 for n,v in entries.items():
  p=PRIOR/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
 write(R/'prior_evidence_verified.json',dict(sha256=sha(ledger),files=len(entries),fresh_numerical_gate=False))
 for package in ['layer0-fp32-a01','layer14-fp32-a01','layer27-fp32-a01','sp2-fp32']:
  old=read(PRIOR/('deployment-'+package+'.json'));p=Path(old['package']);m=read(p/'manifest.json');assert sha(p/'manifest.json')==old['manifest_sha256']
  for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
  names=list(m['files'])
  for k in range(0,len(names),32):
   got=adb('shell','sha256sum '+' '.join(shlex.quote(old['remote']+'/'+n) for n in names[k:k+32])).stdout
   parsed=[line.split(None,1) for line in got.splitlines()];assert len(parsed)==len(names[k:k+32])
   for h,n in parsed:assert h==m['files'][n.removeprefix(old['remote']+'/')]['sha256'],n
  old.update(reused_from=str(PRIOR/('deployment-'+package+'.json')),verified_in_experiment='EXP-0274');write(R/('deployment-'+package+'.json'),old)
  print('MODEL_VERIFIED',package,flush=True)
if __name__=='__main__':prepare()
