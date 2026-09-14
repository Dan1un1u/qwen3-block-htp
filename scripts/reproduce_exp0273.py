"""A0 reproduction only: immutable EXP0272 arithmetic, FP32 residual fixed."""
from common_exp0273 import *
from device_exp0273 import adb, stage
from full_exp0273 import full
import shlex
PRIOR=R.parent/'exp0272'
def prepare():
 preflight()
 ledger=PRIOR/'EVIDENCE_SHA256.json'
 assert sha(ledger)=='7fa56656cbc385a2733146df45d1c5b399a302a0a16898e9fc135ae496016d61'
 entries=read(ledger)['files']
 for n,v in entries.items():
  p=PRIOR/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
 diff=subprocess.check_output(['git','diff','5f859339a62461b3c6fb7dc99c9dc243c1385c45','HEAD','--','src','include','CMakeLists.txt'],cwd=S,text=True)
 assert not diff, 'A0 must have identical native implementation'
 assert read(PRIOR/'single_gate.json')['pass_all'] and read(PRIOR/'slice_gate.json')['pass_all']
 write(R/'prerequisite_provenance.json',dict(inherited_exact_prerequisites_verified=True,fresh_numerical_gate=False,reason='Identical native source/build configuration; exact independent single and chain3 proofs inherited with sealed hashes',prior_ledger_sha256=sha(ledger),verified_evidence_files=len(entries),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()))
 old=read(PRIOR/'deployment-sp2-fp32.json');p=Path(old['package']);m=read(p/'manifest.json');assert sha(p/'manifest.json')==old['manifest_sha256']
 for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
 names=list(m['files'])
 for k in range(0,len(names),32):
  got=adb('shell','sha256sum '+' '.join(shlex.quote(old['remote']+'/'+n) for n in names[k:k+32])).stdout
  parsed=[line.split(None,1) for line in got.splitlines()];assert len(parsed)==len(names[k:k+32])
  for h,n in parsed:assert h==m['files'][n.removeprefix(old['remote']+'/')]['sha256'],n
 old.update(reused_from=str(PRIOR/'deployment-sp2-fp32.json'),verified_in_experiment='EXP-0273');write(R/'deployment-sp2-fp32.json',old)
 print('PREREQUISITES_AND_MODEL_VERIFIED',flush=True)
def campaign():
 preflight();prior=read(PRIOR/'full-formal.json')
 # Use sealed full model token/logit values, not only within-new-run determinism.
 fs=list(PRIOR.glob('**/validated.json'));ref=next(read(p) for p in fs if read(p).get('fp32')==2 and read(p).get('repeat')==10)
 results=[]
 for i,repeat in enumerate([1,10,10,10,10,10]):
  z=full(2,repeat,f'baseline-{i:02d}-r{repeat}')
  assert z['selected_codes']==ref['selected_codes'],'baseline output mismatch'
  results.append(z)
 write(R/'baseline_reproduction.json',dict(experiment='EXP-0273',phase='A0',pass_all=True,comparative_formal_gate=False,integer_residual_tested=False,runs=results,prior_output_reference=str(next(p for p in fs if read(p)==ref))))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','stage','run']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='stage':stage(28)
 else:campaign()
