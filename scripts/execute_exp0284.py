"""Approved ordinary-A8/SP2 comparison, fixed FP32 residual and legal shared pipeline."""
import os,sys,shlex
from common_exp0284 import *
from device_exp0284 import adb,win,stage,run
from selected_audit_exp0284 import audit
from audit_exp0284 import chain,head
from full_exp0284 import full
ARMS=['A8','SP2','INT16'];PRIOR=R.parent/'exp0282/A5-fair-a02'
def setarm(a):
 os.environ.update(QBH_INT16_DOWN='1' if a=='INT16' else '0',QBH_SP2='0' if a=='A8' else '8',QBH_U8_PREFILL_OPT='3' if a=='A8' else '0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0')
def prepare():
 preflight();import shutil
 ledger=R.parent/'exp0282/EVIDENCE_SHA256.json';assert sha(ledger)=='0a7d69354e253d5f755255b4e44c09d723c3d2210752845139642531d94d24a2';entries=read(ledger)['files']
 for name in ['layer14-fp32-a01','sp2-fp32','layer14-a8','full-a8']:
  src=PRIOR/('deployment-'+name+'.json');assert sha(src)==entries[str(src.relative_to(ledger.parent))]['sha256'];cfg=read(src);p=package_path(name);m=read(p/'manifest.json');assert sha(p/'manifest.json')==cfg['manifest_sha256']
  for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
  names=[n for n in m['files'] if not n.endswith('.npy')]
  for i in range(0,len(names),32):
   lines=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
   for line in lines:
    h,n=line.split(None,1);assert h==m['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
  write(R/src.name,cfg);print('CONTROL_VERIFIED',name,flush=True)
 for name,old in [('layer14-int16','layer14-fp32-a01'),('full-int16','sp2-fp32')]:
  p=package_path(name);m=read(p/'manifest.json');a=read(R/(name+'-package.json'));assert a['manifest_sha256']==sha(p/'manifest.json');parent=read(R/f'deployment-{old}.json');pm=read(package_path(old)/'manifest.json');remote='/data/local/tmp/qwen3-block-htp/exp0284-models/'+name
  assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
  names=[n for n in m['files'] if not n.endswith('.npy')];dirs=sorted({str(Path(n).parent) for n in names if str(Path(n).parent)!='.'});adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs));links=[];copies=[]
  for n in names:
   v=m['files'][n];assert sha(p/n)==v['sha256'],n
   if n in pm['files'] and v['sha256']==pm['files'][n]['sha256']:links.append((parent['remote']+'/'+n,remote+'/'+n))
   else:copies.append(n)
  for i in range(0,len(links),24):adb('shell',' && '.join('ln -s '+shlex.quote(x)+' '+shlex.quote(y) for x,y in links[i:i+24]))
  for n in copies:adb('push',win(p/n),remote+'/'+n)
  for i in range(0,len(names),32):
   lines=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
   for line in lines:
    h,n=line.split(None,1);assert h==m['files'][n.removeprefix(remote+'/')]['sha256'],n
  write(R/f'deployment-{name}.json',dict(remote=remote,manifest_sha256=sha(p/'manifest.json'),package=str(p),immutable_links=len(links),fresh_files=copies));print('INT16_DEPLOYED',name,flush=True)
 src=PRIOR/'fixed_tokens.json';assert sha(src)==entries[str(src.relative_to(ledger.parent))]['sha256'];shutil.copyfile(src,R/'fixed_tokens.json')
 write(R/'provenance.json',dict(prior_ledger_sha256=sha(ledger),native_DSP_unchanged=True,int16_oracles={m:sha(R/(m+'-int16-teacher.json')) for m in ['fixed','greedy']}))
def selected():
 out=[]
 for a in ARMS:
  setarm(a);name='layer14-int16' if a=='INT16' else 'layer14-fp32-a01' if a=='SP2' else 'layer14-a8';tag=f'selected-{a}-a01';run(name,tag,fp32=2,dump=True);out.append(audit(name,tag))
 write(R/'single_gate.json',dict(pass_all=True,runs=out))
def slices():
 assert read(R/'single_gate.json')['pass_all'];out=[]
 for a in ARMS:
  setarm(a);name='full-int16' if a=='INT16' else 'sp2-fp32' if a=='SP2' else 'full-a8';tag=f'slice-{a}-a01';rep=f'slice-{a}-r10';run(name,tag,count=3,fp32=2,dump=True);run(name,rep,count=3,fp32=2,repeat=10);chain(tag,rep);out.append(read(R/tag/'chain_gate.json'))
 write(R/'slice_gate.json',dict(pass_all=True,runs=out))
def fullgates():
 out=[]
 for forced in [False,True]:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if forced else '0'
  for a in ARMS:
   setarm(a);tag=f'full-{a}-'+('fixed' if forced else 'greedy')+'-audit-a02';z=full(2,1,tag,audit=True);head(tag)
   if a!='INT16':assert z['selected_codes']==read(PRIOR/tag.removesuffix('-a02')/'validated.json')['selected_codes']
   out.append(dict(arm=a,forced=forced,tag=tag,**z))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 for mode in ['fixed','greedy']:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if mode=='fixed' else '0'
  def one(a,tag,rep):
   setarm(a);z=full(2,rep,tag);assert z['selected_codes']==read(R/f'full-{a}-{mode}-audit-a02/validated.json')['selected_codes'];return dict(arm=a,**z)
  for a in ARMS:one(a,f'{mode}-warmup-{a}',1)
  write(R/f'{mode}_auxiliary.json',[one(a,f'{mode}-aux-{a}-r1',1) for a in ARMS])
  for phase,n in [('short',5),('formal',10)]:
   out=[]
   for c in range(n):
    j=c%3;order=ARMS[j:]+ARMS[:j]
    if c%2:order=order[::-1]
    for a in order:out.append(dict(cycle=c,**one(a,f'{mode}-{phase}/{c:02d}-{a}',10)))
   write(R/f'{mode}_{phase}.json',dict(pass_all=True,runs=out))
if __name__=='__main__':
 a=sys.argv[1]
 if a=='stage':stage(int(sys.argv[2]))
 else:{'prepare':prepare,'selected':selected,'slices':slices,'fullgates':fullgates,'timing':timing}[a]()
