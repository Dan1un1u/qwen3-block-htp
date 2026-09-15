"""Fixed EXP0274 four-arm campaign; fail closed, no timing-based resampling."""
import sys,os
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from common_exp0274 import *
from device_exp0274 import run
from selected_audit_exp0274 import audit
from audit_exp0274 import chain,head
from full_exp0274 import full
# Attempt2 owns separate outputs; frozen model paths and original oracle remain.
import common_exp0274,device_exp0274,selected_audit_exp0274,audit_exp0274,full_exp0274
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0274/consumer-row1-a02')
for mod in [common_exp0274,device_exp0274,selected_audit_exp0274,audit_exp0274,full_exp0274]:mod.R=R
device_exp0274.REMOTE='/data/local/tmp/qwen3-block-htp/exp0274-consumer-row1-a02'
ARMS={'B11':(0,0),'B01':(3,0),'B10':(0,31),'B00':(3,31)}
def setarm(a):
 f,p=ARMS[a];os.environ['QBH_PAPER_FORMAT_DISABLE']=str(f);os.environ['QBH_PAPER_PIPELINE_DISABLE']=str(p)
def selected():
 out=[]
 for layer in [0,14,27]:
  for a in ARMS:
   setarm(a);tag=f'selected-l{layer}-{a}-a01';pkg=f'layer{layer}-fp32-a01'
   run(pkg,tag,fp32=2,dump=True);out.append(audit(pkg,tag))
 write(R/'single_gate.json',dict(pass_all=True,runs=out))
def slices():
 assert read(R/'single_gate.json')['pass_all'];out=[]
 for a in ARMS:
  setarm(a);tag=f'slice-{a}-a01';rep=f'slice-{a}-r10'
  run('sp2-fp32',tag,count=3,fp32=2,dump=True)
  run('sp2-fp32',rep,count=3,repeat=10,fp32=2)
  chain(tag,rep);out.append(read(R/tag/'chain_gate.json'))
 write(R/'slice_gate.json',dict(pass_all=True,runs=out))
def fullgates():
 prior=R.parent.parent/'exp0272/full-fp32-audit';ref=read(prior/'validated.json');out=[]
 names=[x.name for x in prior.glob('*.bin') if x.name!='eval.bin']
 assert len(names)==88,len(names)
 for a in ARMS:
  setarm(a);tag=f'full-{a}-audit';z=full(2,1,tag,audit=True)
  assert z['selected_codes']==ref['selected_codes']
  for n in names:assert sha(R/tag/n)==sha(prior/n),(a,n)
  head(tag);out.append(dict(arm=a,reference=str(prior),exact_files=names,frontend=read(R/tag/'independent_gate.json')))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all'];ref=read(R/'full-B11-audit/validated.json')['selected_codes']
 aux=[]
 for a in ARMS:
  setarm(a);z=full(2,1,f'aux-{a}-r1');assert z['selected_codes']==ref;aux.append(dict(arm=a,**z))
 write(R/'auxiliary.json',aux)
 for phase,cycles in [('short',5),('formal',10)]:
  out=[];keys=list(ARMS)
  for cycle in range(cycles):
   shift=cycle%4
   for a in keys[shift:]+keys[:shift]:
    setarm(a);z=full(2,10,f'{phase}/{cycle:02d}-{a}');assert z['selected_codes']==ref
    out.append(dict(cycle=cycle,arm=a,**z))
  write(R/f'{phase}.json',dict(pass_all=True,runs=out))
if __name__=='__main__':
 if sys.argv[1]=='stage':device_exp0274.stage(int(sys.argv[2]))
 else:{'selected':selected,'slices':slices,'fullgates':fullgates,'timing':timing}[sys.argv[1]]()
