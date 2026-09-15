"""EXP0280 targeted format diagnosis; frozen references, exclusive new evidence."""
import os,sys,subprocess,shlex
from pathlib import Path
import common_exp0274 as c, device_exp0274 as d, full_exp0274 as f, selected_audit_exp0274 as a, audit_exp0274 as aa
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0280')
def preflight():
 z=subprocess.check_output(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(c.S)],text=True);assert 'EXPERIMENT=EXP-0280' in z
for mod in [c,d,f,a,aa]:mod.R=R;mod.preflight=preflight
d.REMOTE='/data/local/tmp/qwen3-block-htp/exp0280'
from common_exp0274 import read,write,sha,package_path
ARMS={'NATIVE':0,'NORM_COMPACT':1,'SWIGLU_COMPACT':2,'COMPACT':3}
def arm(x):os.environ.update(QBH_PAPER_FORMAT_DISABLE=str(ARMS[x]),QBH_PAPER_PIPELINE_DISABLE='0')
def prepare():
 preflight();R.mkdir(exist_ok=True)
 assert not d.adb('shell','pidof qwen3_block_cli',check=False).stdout.strip()
 prior=R.parent/'exp0274/consumer-row1-a02'
 for pkg in ['layer0-fp32-a01','layer14-fp32-a01','layer27-fp32-a01','sp2-fp32']:
  cfg=read(prior/('deployment-'+pkg+'.json'));p=Path(cfg['package']);assert sha(p/'manifest.json')==cfg['manifest_sha256'];man=read(p/'manifest.json')
  for n,v in man['files'].items():assert sha(p/n)==v['sha256'],n
  names=list(man['files'])
  for k in range(0,len(names),32):
   lines=d.adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[k:k+32])).stdout.splitlines();assert len(lines)==len(names[k:k+32])
   for line in lines:
    h,n=line.split(None,1);assert h==man['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
  write(R/('deployment-'+pkg+'.json'),cfg);print('VERIFIED',pkg,flush=True)
 write(R/'device_owner.json',dict(experiment='EXP-0280',exclusive=True))
def selected():
 out=[]
 for l in [0,14,27]:
  for x in ARMS:
   arm(x);tag=f'diagnosis/l{l}-{x}';pkg=f'layer{l}-fp32-a01';d.run(pkg,tag,fp32=2,dump=True);out.append(a.audit(pkg,tag))
 write(R/'single_gate.json',dict(pass_all=True,runs=out))
def diagnose():
 assert read(R/'single_gate.json')['pass_all'];out=[];keys=list(ARMS)
 for cycle in range(5):
  for x in keys[cycle%4:]+keys[:cycle%4]:
   arm(x);out.append(dict(arm=x,cycle=cycle,**d.run('layer14-fp32-a01',f'diag-timing/{cycle:02d}-{x}',repeat=10,fp32=2)))
 write(R/'diagnosis.json',dict(pass_all=True,runs=out))
if __name__=='__main__':
 if sys.argv[1]=='stage':d.stage(int(sys.argv[2]))
 else:dict(prepare=prepare,selected=selected,diagnose=diagnose)[sys.argv[1]]()
