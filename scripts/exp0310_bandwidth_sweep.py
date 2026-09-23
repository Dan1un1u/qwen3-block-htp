from pathlib import Path
import subprocess,json,statistics,time
import os
out=Path(os.environ.get('QBH_BW_RESULTS','/mnt/d/llm_exp/results/qwen3-block-htp/exp0310'))
out.mkdir(parents=True,exist_ok=True)
adb=['/mnt/c/adb/adb.exe','-s','3B15C8007Z300000']
def run(args,label):
 cmd='cd /data/local/tmp/qbh_bw0310 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./qwen3_bandwidth_cli '+' '.join(map(str,args))
 t=time.time();p=subprocess.run(adb+['shell',cmd],capture_output=True,text=True,timeout=90)
 (out/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((label,p.returncode,p.stdout,p.stderr))
 x=json.loads(next(l for l in p.stdout.splitlines() if l.startswith('{')))
 assert x['errors']==0 and x['status']==0 and min(x['ticks'])>0 and max(x['ticks'])<192000000
 x['label']=label;x['host_seconds']=time.time()-t;x['gbps']=x['payload']*0.0192/statistics.median(x['ticks'][2:]);x['gbps_best']=x['payload']*0.0192/min(x['ticks'][2:]);x['bpc']=x['payload']/statistics.median(x['cycles'][2:]);x['mhz']=statistics.median([c/t*19.2 for c,t in zip(x['cycles'][2:],x['ticks'][2:])])
 with (out/'sweep.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(label,'GB/s',round(x['gbps'],2),'B/cycle',round(x['bpc'],2),'MHz',round(x['mhz'],1),flush=True)
 return x
if __name__=='__main__':
 cases=[]
 for mode in [0,1,2]:
  for workers in [1,2,4,6]:
   for size in [16384,65536,524288]:cases.append([mode,size,workers,268435456//size,32,1,1])
 for mode in [3,4,5]:
  for size in [65536,1048576,3145728]:
   for stream in [1,4,16,32]:cases.append([mode,size,1,max(1,1073741824//size),stream,1,1])
 for chunk in [4096,16384,65536,262144,1048576,4194304]:
  for depth in [1,4,16,64]:
   if chunk*depth<=8388608:cases.append([6,chunk,1,min(65536,536870912//(chunk*depth)),32,depth,1])
 for i,c in enumerate(cases):run(c,'scan_%03d'%i)
