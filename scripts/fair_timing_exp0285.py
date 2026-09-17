import sys,os
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from common_exp0285 import *
import full_exp0285 as f
from device_exp0285 import adb
preflight();old=read(R/'original-baseline-provenance.json');new=read(R/'runtime-l28.json')
for state in [old,new]:
 for n,h in state['seal']['files'].items():
  assert sha(Path(state['archive'])/Path(n).name)==h
  assert adb('shell','sha256sum '+state['remote']+'/'+Path(n).name).stdout.split()[0]==h
reader=f.read;chosen=new
def choose_read(p):return chosen if Path(p)==R/'runtime-l28.json' else reader(p)
f.read=choose_read
os.environ.update(QBH_SP2='8',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_PAPER_FIXED_TOKENS='1')
def one(a,tag,rep):
 global chosen
 chosen=old if a=='FP32' else new
 z=f.full(2 if a=='FP32' else 3,rep,tag)
 assert z['selected_codes']==read(R/f'full-{a}-audit-a01/validated.json')['selected_codes']
 return dict(arm=a,**z)
for a in ['FP32','FP16']:one(a,f'original-paired/warmup-{a}',1)
write(R/'original-paired/auxiliary.json',[one(a,f'original-paired/aux-{a}',1) for a in ['FP32','FP16']])
for phase,n in [('short',5),('formal',10)]:
 out=[]
 for c in range(n):
  for a in (['FP32','FP16'] if c%2==0 else ['FP16','FP32']):
   out.append(dict(cycle=c,**one(a,f'original-paired/{phase}/{c:02d}-{a}',10)))
 write(R/f'original-paired/{phase}.json',dict(pass_all=True,runs=out))
