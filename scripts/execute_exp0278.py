"""Approved ordinary-A8/SP2 comparison, fixed FP32 residual and legal shared pipeline."""
import os,sys,shlex
from common_exp0278 import *
from device_exp0278 import adb,win,stage,run
from selected_audit_exp0278 import audit
from audit_exp0278 import chain,head
from full_exp0278 import full
ARMS=['LOG2','FP'];PRIOR=R.parent/'exp0277'
def setarm(a):
 os.environ.update(QBH_SP2='8',QBH_WIDE_SCORE='4' if a=='LOG2' else '7',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0')
def prepare():
 preflight();ledger=PRIOR/'EVIDENCE_SHA256.json';assert sha(ledger)=='160ef619a2017a11913246d7459ca031c14f3fa00f7a9dedf9da5fb11e6a8eef'
 for n,v in read(ledger)['files'].items():assert sha(PRIOR/n)==v['sha256'],n
 for name in ['layer14-fp32-a01','sp2-fp32']:
  cfg=read(PRIOR/f'deployment-{name}.json');p=package_path(name);m=read(p/'manifest.json');assert sha(p/'manifest.json')==cfg['manifest_sha256']
  for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
  write(R/f'deployment-{name}.json',cfg)
 for name,old in [('layer14-fp','layer14-fp32-a01'),('full-fp','sp2-fp32')]:
  p=package_path(name);m=read(p/'manifest.json');a=read(R/(name+'-package.json'));assert a['manifest_sha256']==sha(p/'manifest.json');parent=read(R/f'deployment-{old}.json');pm=read(package_path(old)/'manifest.json');remote='/data/local/tmp/qwen3-block-htp/exp0278-models/'+name
  assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
  dirs=sorted({str(Path(n).parent) for n in m['files'] if str(Path(n).parent)!='.'});adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs))
  links=[];copies=[]
  for n,v in m['files'].items():
   assert sha(p/n)==v['sha256'],n
   if n in pm['files'] and v['sha256']==pm['files'][n]['sha256']:links.append((parent['remote']+'/'+n,remote+'/'+n))
   else:copies.append(n)
  # Links only for immutable identical payload; changed metadata always fresh regular files.
  for i in range(0,len(links),24):adb('shell',' && '.join('ln -s '+shlex.quote(x)+' '+shlex.quote(y) for x,y in links[i:i+24]))
  for n in copies:adb('push',win(p/n),remote+'/'+n)
  names=list(m['files'])
  for i in range(0,len(names),32):
   lines=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
   for l in lines:
    h,n=l.split(None,1);assert h==m['files'][n.removeprefix(remote+'/')]['sha256'],n
  write(R/f'deployment-{name}.json',dict(remote=remote,manifest_sha256=sha(p/'manifest.json'),package=str(p),immutable_links=len(links),fresh_files=copies))
 write(R/'fixed_tokens.json',read(PRIOR/'fixed_tokens.json'))

def selected():
 out=[]
 for a in ARMS:
  setarm(a);name='layer14-fp32-a01' if a=='LOG2' else 'layer14-fp';tag=f'selected-{a}-a01';run(name,tag,fp32=2,dump=True);out.append(audit(name,tag))
 write(R/'single_gate.json',dict(pass_all=True,runs=out))
def slices():
 assert read(R/'single_gate.json')['pass_all'];out=[]
 for a in ARMS:
  setarm(a);name='sp2-fp32' if a=='LOG2' else 'full-fp';tag=f'slice-{a}-a01';rep=f'slice-{a}-r10';run(name,tag,count=3,fp32=2,dump=True);run(name,rep,count=3,fp32=2,repeat=10);chain(tag,rep);out.append(read(R/tag/'chain_gate.json'))
 write(R/'slice_gate.json',dict(pass_all=True,runs=out))
def fullgates():
 out=[]
 for forced in [False,True]:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if forced else '0'
  for a in ARMS:
   setarm(a);tag=f'full-{a}-'+('fixed' if forced else 'greedy')+'-audit';z=full(2,1,tag,audit=True);head(tag)
   if a=='LOG2':assert z['selected_codes']==read(PRIOR/'full-SP2-greedy-audit/validated.json')['selected_codes']
   out.append(dict(arm=a,forced=forced,tag=tag,**z))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 for mode in ['greedy','fixed']:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if mode=='fixed' else '0'
  def one(a,tag,rep):
   setarm(a);z=full(2,rep,tag);assert z['selected_codes']==read(R/f'full-{a}-{mode}-audit/validated.json')['selected_codes'];return dict(arm=a,**z)
  aux=[one(a,f'{mode}-aux-{a}-r1',1) for a in ARMS];write(R/f'{mode}_auxiliary.json',aux)
  for phase,n in [('short',5),('formal',10)]:
   out=[]
   for c in range(n):
    for a in (ARMS if c%2==0 else ARMS[::-1]):out.append(dict(cycle=c,**one(a,f'{mode}-{phase}/{c:02d}-{a}',10)))
   write(R/f'{mode}_{phase}.json',dict(pass_all=True,runs=out))
if __name__=='__main__':
 a=sys.argv[1]
 if a=='stage':stage(int(sys.argv[2]))
 else:{'prepare':prepare,'selected':selected,'slices':slices,'fullgates':fullgates,'timing':timing}[a]()
