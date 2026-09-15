import sys,json,subprocess,time,re
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
import execute_llama32_pipeline_ablation as e
r=e.R;e.preflight();state=e.read(r/'runtime-l1.json');assert state['seal']['paper_trace'] is True
farf=state['remote']+'/qwen3_block_cli.farf';assert e.adb('shell','test ! -e '+farf,check=False).returncode==0;e.adb('shell','touch '+farf)
e.write(r/'trace_configuration.json',dict(remote_farf=farf,timing_eligible=False,source=state['seal']['source_head']))
for a in e.ARMS:
 tag=f'trace-{a}-a01';log=r/f'{tag}-logcat.txt'
 with log.open('xb') as f:
  listener=subprocess.Popen(['/mnt/c/adb/adb.exe','logcat','-v','brief','-T','1','-s','adsprpc:V'],stdout=f,stderr=subprocess.PIPE)
  try:e.execute('l7',a,tag,audit=True);time.sleep(2)
  finally:
   listener.terminate();_,err=listener.communicate(timeout=15);(r/f'{tag}-logcat-stderr.txt').write_bytes(err)
 events=[];counts=[]
 for line in log.read_text(errors='replace').splitlines():
  m=re.search(r'QBH_PAPER_TRACE (\d+) (\d+) (\w+) (\d+) ([0-9a-fA-F]+)',line)
  if m:events.append(dict(tick=int(m[1]),tid=int(m[2]),event=m[3],kind=int(m[4]),object=int(m[5],16)))
  m=re.search(r'QBH_PAPER_TRACE_COUNT (\d+) (\d+) (\d+)',line)
  if m:counts.append(dict(total=int(m[1]),kept=int(m[2]),dropped=int(m[3])))
 complete=len(counts)==2 and all(v['dropped']==0 for v in counts) and len(events)==sum(v['kept'] for v in counts)
 e.write(r/f'{tag}-events.json',dict(events=events,counts=counts,trace_complete=complete,timing_eligible=False,numerical_pass=True));print('TRACE',a,len(events),counts,complete,flush=True)
