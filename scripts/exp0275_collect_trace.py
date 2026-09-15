import sys,subprocess,json,time,re,os
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import execute_exp0275 as e
from device_exp0275 import old,run
from selected_audit_exp0275 import audit
r=e.R
assert e.read(r/'runtime-l1.json')['seal']['paper_trace'] is True
for a in e.ARMS:
 e.setarm(a);tag=f'trace-{a}-a01';log=r/f'{tag}-logcat.txt'
 with log.open('xb') as f:
  listener=subprocess.Popen([old.ADB,'logcat','-v','brief','-T','1','-s','adsprpc:V'],stdout=f,stderr=subprocess.PIPE)
  try:
   run('layer14-fp32-a01',tag,fp32=2,dump=True);audit('layer14-fp32-a01',tag);time.sleep(2)
  finally:
   listener.terminate();_,err=listener.communicate(timeout=15)
   (r/f'{tag}-logcat-stderr.txt').write_bytes(err)
 events=[];counts=[]
 for line in log.read_text(errors='replace').splitlines():
  m=re.search(r'QBH_PAPER_TRACE (\d+) (\d+) (\w+) (\d+) ([0-9a-fA-F]+)',line)
  if m:events.append(dict(tick=int(m[1]),tid=int(m[2]),event=m[3],kind=int(m[4]),object=int(m[5],16)))
  m=re.search(r'QBH_PAPER_TRACE_COUNT (\d+) (\d+) (\d+)',line)
  if m:counts.append(dict(total=int(m[1]),kept=int(m[2]),dropped=int(m[3])))
 # Count mismatch is an explicit incomplete trace, never a fabricated timeline.
 complete=len(counts)==2 and all(v['dropped']==0 for v in counts) and len(events)==sum(v['kept'] for v in counts)
 e.write(r/f'{tag}-events.json',dict(events=events,counts=counts,trace_complete=complete,timing_eligible=False,numerical_pass=True))
 print('TRACE',a,len(events),counts,complete,flush=True)
