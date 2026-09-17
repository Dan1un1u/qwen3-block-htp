"""Five short / ten formal paired ABBA blocks; repeat10 primary."""
import os
from full_exp0288 import *
def run():
 assert read(R/"full64-control-a03/independent_gate.json")["pass_all"]
 assert read(R/"full64-opt1-a01/independent_gate.json")["pass_all"]
 gold=read(R/"frontend64-a03-teacher.json");codes=[list(v) for v in zip(gold["u8_generated_ids"][:43],gold["u8_selected_codes"][:43])]
 os.environ["QBH_PAPER_FIXED_TOKENS"]="1"
 def one(arm,tag,repeat=10,steps=43):
  os.environ["QBH_RUNTIME_FILE"]="runtime-control-l28.json" if arm=="CONTROL" else "runtime-opt1-l28.json"
  z=full(2,repeat,tag,steps=steps);expected=[list(v) for v in zip(gold["u8_generated_ids"][:steps],gold["u8_selected_codes"][:steps])];assert z["selected_codes"]==expected
  return dict(arm=arm,tag=tag,**z)
 for arm in ["CONTROL","OPT1"]:one(arm,f"warmup-{arm}",1)
 write(R/"auxiliary.json",[one(arm,f"aux-{arm}",1) for arm in ["CONTROL","OPT1"]])
 for phase,n in [("short",5),("formal",10)]:
  rows=[]
  for i in range(n):
   order=["CONTROL","OPT1","OPT1","CONTROL"] if i%2==0 else ["OPT1","CONTROL","CONTROL","OPT1"]
   for j,arm in enumerate(order):rows.append(dict(cycle=i,position=j,**one(arm,f"{phase}/{i:02d}-{j}-{arm}")))
   write(R/f"{phase}-cycle-{i:02d}.json",rows[-4:])
  write(R/f"{phase}.json",dict(pass_all=True,repeat=10,blocks=n,runs=rows))
 write(R/"long63-supplement.json",[one(arm,f"long63-{arm}-r10",10,64) for arm in ["CONTROL","OPT1"]])
if __name__=="__main__":run()
