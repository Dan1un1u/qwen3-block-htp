"""Approved ordinary-A8/SP2 comparison, fixed FP32 residual and legal shared pipeline."""
import os,sys,shlex
from common_exp0278_vsum import *
from device_exp0278_vsum import adb,win,stage,run
from selected_audit_exp0278_vsum import audit
from audit_exp0278_vsum import chain,head
from full_exp0278_vsum import full
ARMS=['LOG2','FP'];PRIOR=R.parent.parent/'exp0277'
def setarm(a):
 os.environ.update(QBH_SP2='8',QBH_WIDE_SCORE='4' if a=='LOG2' else '7',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0')
def prepare():
 preflight();R.mkdir(exist_ok=False)
 for name in ['layer14-fp32-a01','sp2-fp32','layer14-fp','full-fp']:
  cfg=read(R.parent/f'deployment-{name}.json');p=package_path(name);assert sha(p/'manifest.json')==cfg['manifest_sha256']
  write(R/f'deployment-{name}.json',cfg)
 write(R/'fixed_tokens.json',read(R.parent/'fixed_tokens.json'))
 write(R/'protocol.json',dict(primary='same FP32 vector exp/rounding with vector probability-code rowmass; no scalar telemetry loop',prototype=str(R.parent/'PROTOTYPE_A01.json'),timing='5short10formal repeat10 each greedy/fixed, no pooling with prototype',component='matched telemetry enabled for both arms'))
def verify_models():
 preflight()
 for name in ['layer14-fp32-a01','sp2-fp32','layer14-fp','full-fp']:
  cfg=read(R/f'deployment-{name}.json');p=package_path(name);manifest=read(p/'manifest.json');assert sha(p/'manifest.json')==cfg['manifest_sha256']
  for n,v in manifest['files'].items():assert sha(p/n)==v['sha256'],n
  names=list(manifest['files'])
  for i in range(0,len(names),32):
   lines=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
   for line in lines:
    h,n=line.split(None,1);assert h==manifest['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
  print('MODEL_VERIFIED',name,flush=True)
 write(R/'models_verified.json',dict(pass_all=True))
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
 else:{'prepare':prepare,'verify_models':verify_models,'selected':selected,'slices':slices,'fullgates':fullgates,'timing':timing}[a]()
