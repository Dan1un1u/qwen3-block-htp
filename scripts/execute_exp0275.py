"""Approved fixed pipeline knockouts; production timing never uses trace builds."""
import os,sys,shlex,shutil
from common_exp0275 import *
from device_exp0275 import run,stage,adb
from selected_audit_exp0275 import audit
from audit_exp0275 import chain,head
from full_exp0275 import full
ARMS={'ALL':0,'QKV':1,'FFN':2,'EPILOGUE':12,'LOOKAHEAD':16}
PRIOR=R.parent/'exp0274/consumer-row1-a02'
def setarm(a):
 os.environ['QBH_PAPER_FORMAT_DISABLE']='0';os.environ['QBH_PAPER_PIPELINE_DISABLE']=str(ARMS[a])
def prepare():
 preflight();ledger=PRIOR.parent/'EVIDENCE_SHA256.json';assert sha(ledger)=='ce6873fee5fb0c66d0fdb63043341b7353aaf664875580b6b8fa00e25978f5d6'
 for n,v in read(ledger)['files'].items():assert (ledger.parent/n).stat().st_size==v['bytes'] and sha(ledger.parent/n)==v['sha256'],n
 for name in ['layer14-fp32-a01','sp2-fp32']:
  cfg=read(PRIOR/f'deployment-{name}.json');pkg=package_path(name);m=read(pkg/'manifest.json');assert sha(pkg/'manifest.json')==cfg['manifest_sha256']
  for n,v in m['files'].items():assert sha(pkg/n)==v['sha256'],n
  names=list(m['files'])
  for i in range(0,len(names),32):
   lines=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
   for l in lines:
    h,n=l.split(None,1);assert h==m['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
  write(R/f'deployment-{name}.json',cfg)
 write(R/'provenance.json',dict(prior_ledger_sha256=sha(ledger),verified_files=len(read(ledger)['files'])))
def selected():
 out=[]
 for a in ARMS:
  setarm(a);tag=f'selected-{a}-a01';run('layer14-fp32-a01',tag,fp32=2,dump=True);out.append(audit('layer14-fp32-a01',tag))
 write(R/'single_gate.json',dict(pass_all=True,runs=out))
def slices():
 assert read(R/'single_gate.json')['pass_all'];out=[]
 for a in ARMS:
  setarm(a);tag=f'slice-{a}-a01';rep=f'slice-{a}-r10';run('sp2-fp32',tag,count=3,fp32=2,dump=True);run('sp2-fp32',rep,count=3,repeat=10,fp32=2);chain(tag,rep);out.append(read(R/tag/'chain_gate.json'))
 write(R/'slice_gate.json',dict(pass_all=True,runs=out))
def fullgates():
 ref=read(PRIOR/'full-B11-audit/validated.json')['selected_codes'];out=[]
 for a in ARMS:
  setarm(a);tag=f'full-{a}-audit';z=full(2,1,tag,audit=a=='ALL');assert z['selected_codes']==ref
  if a=='ALL':
   names=[p.name for p in (PRIOR/'full-B11-audit').glob('*.bin') if p.name!='eval.bin'];assert len(names)==88
   for n in names:assert sha(R/tag/n)==sha(PRIOR/'full-B11-audit'/n),n
   head(tag)
  out.append(dict(arm=a,**z))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing(scope):
 assert read(R/('full_gate.json' if scope==28 else 'slice_gate.json' if scope==3 else 'single_gate.json'))['pass_all']
 assert read(R/f'runtime-l{scope}.json')['seal']['paper_trace'] is False
 ref=read(PRIOR/'full-B11-audit/validated.json')['selected_codes'];out=[]
 def one(a,tag,repeat):
  setarm(a)
  if scope==28:
   z=full(2,repeat,tag);assert z['selected_codes']==ref
  else:
   z=run('sp2-fp32' if scope==3 else 'layer14-fp32-a01',tag,count=scope,repeat=repeat,fp32=2);f=read(R/(f'slice-{a}-a01' if scope==3 else f'selected-{a}-a01')/'validated.json');assert z['output_hashes']==f['output_hashes']*repeat
  return dict(arm=a,**z)
 if scope==28:
  for a in ARMS:out.append(one(a,f'full-aux-{a}-r1',1))
  write(R/'full_auxiliary.json',out)
 for phase,cycles in ([('short',5),('formal',10)] if scope==28 else [('formal',10)]):
  out=[];keys=list(ARMS)
  for c in range(cycles):
   j=c%len(keys)
   for a in keys[j:]+keys[:j]:out.append(dict(cycle=c,**one(a,f'l{scope}-{phase}/{c:02d}-{a}',10)))
  write(R/f'l{scope}_{phase}.json',dict(pass_all=True,runs=out))
if __name__=='__main__':
 action=sys.argv[1]
 if action=='stage':stage(int(sys.argv[2]),trace='trace' in sys.argv[3:])
 elif action=='timing':timing(int(sys.argv[2]))
 else:{'prepare':prepare,'selected':selected,'slices':slices,'fullgates':fullgates}[action]()
