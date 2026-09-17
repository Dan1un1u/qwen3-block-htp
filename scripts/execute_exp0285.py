"""Fixed M64+15 paired FP32/FP16-residual speed, five short + ten formal rounds."""
from common_exp0285 import *
from full_exp0285 import full
from audit_exp0285 import fullaudit
import os
os.environ.update(QBH_SP2='8',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_PAPER_FIXED_TOKENS='1')
ARMS={'FP32':2,'FP16':3}
def audits():
 for name,fp in ARMS.items():
  tag=f'full-{name}-audit-a01';full(fp,1,tag,audit=True);fullaudit(fp,tag)
 write(R/'full_gate.json',dict(pass_all=True,arms=list(ARMS)))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 def one(name,tag,repeat):
  v=full(ARMS[name],repeat,tag)
  assert v['selected_codes']==read(R/f'full-{name}-audit-a01/validated.json')['selected_codes']
  return dict(arm=name,**v)
 for name in ARMS:one(name,f'warmup-{name}',1)
 write(R/'auxiliary.json',[one(name,f'aux-{name}',1) for name in ARMS])
 for phase,n in [('short',5),('formal',10)]:
  result=[]
  for cycle in range(n):
   order=['FP32','FP16'] if cycle%2==0 else ['FP16','FP32']
   for name in order:result.append(dict(cycle=cycle,**one(name,f'{phase}/{cycle:02d}-{name}',10)))
  write(R/f'{phase}.json',dict(pass_all=True,runs=result))
if __name__=='__main__':
 {'audits':audits,'timing':timing}[sys.argv[1]]()
