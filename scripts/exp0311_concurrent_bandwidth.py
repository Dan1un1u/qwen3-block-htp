from pathlib import Path
import subprocess,json,statistics,time
import os
out=Path(os.environ.get('QBH_BW_RESULTS','/mnt/d/llm_exp/results/qwen3-block-htp/exp0311'))
out.mkdir(parents=True,exist_ok=True)
adb=['/mnt/c/adb/adb.exe','-s','3B15C8007Z300000']
def run(mode,workers,layout,fmt,label,duration=50000):
 cmd=f'cd /data/local/tmp/qbh_bw0311 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./qwen3_bandwidth_cli {mode} 1048576 {workers} {duration} 32 {layout} {fmt}'
 p=subprocess.run(adb+['shell',cmd],text=True,capture_output=True,timeout=45)
 (out/(label+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,(label,p.returncode,p.stdout,p.stderr)
 x=json.loads(next(l for l in p.stdout.splitlines() if l.startswith('{')))
 assert not x['errors'] and not x['status'],x
 assert max(x['start_delay'][2:])<duration*19.2*.01,x
 assert max(x['finish_delay'][2:])<duration*19.2*.01,x
 x['label']=label;x['hvx_GBps']=statistics.median([b/t*.0192 for b,t in zip(x['hvx_bytes'][2:],x['ticks'][2:])]);x['hmx_GBps']=statistics.median([b/t*.0192 for b,t in zip(x['hmx_bytes'][2:],x['ticks'][2:])]);x['mhz']=statistics.median([c/t*19.2 for c,t in zip(x['cycles'][2:],x['ticks'][2:])])
 with (out/'results.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(label,round(x['hvx_GBps'],2),round(x['hmx_GBps'],2),'MHz',round(x['mhz'],1),'start_us',round(max(x['start_delay'][2:])/19.2,2),'tail_us',round(max(x['finish_delay'][2:])/19.2,2),flush=True)
 return x

if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]
 subprocess.run(['python3',str(root)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(root)],check=True)
 cases=json.loads((root/'experiments/exp0311_concurrent_formal_cases.json').read_text())
 for session in range(3):
  keys=list(cases);keys=keys[session*8:]+keys[:session*8]
  if session==1:keys=list(reversed(keys))
  (out/f'formal_battery_{session}_before.txt').write_text(subprocess.check_output(adb+['shell','dumpsys battery'],text=True))
  for key in keys:run(*cases[key],f'formal_{session}_{key}')
  (out/f'formal_battery_{session}_after.txt').write_text(subprocess.check_output(adb+['shell','dumpsys battery'],text=True))
