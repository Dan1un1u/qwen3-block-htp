"""EXP0280 targeted format diagnosis; frozen references, exclusive new evidence."""
import os,sys,subprocess,shlex
from pathlib import Path
import common_exp0274 as c, device_exp0274 as d, full_exp0274 as f, selected_audit_exp0274 as a, audit_exp0274 as aa
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0280/opt-a2')
def preflight():
 z=subprocess.check_output(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(c.S)],text=True);assert 'EXPERIMENT=EXP-0280' in z
for mod in [c,d,f,a,aa]:mod.R=R;mod.preflight=preflight
d.REMOTE='/data/local/tmp/qwen3-block-htp/exp0280-opt-a2'
from common_exp0274 import read,write,sha,package_path
ARMS={'OPT':0,'ORIGINAL':8,'DUAL':16,'NORM_COMPACT':17,'COMPACT':3}
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
def reuse():
 preflight();R.mkdir(exist_ok=False)
 for pkg in ['layer0-fp32-a01','layer14-fp32-a01','layer27-fp32-a01','sp2-fp32']:
  cfg=read(R.parent/('deployment-'+pkg+'.json'));assert sha(Path(cfg['package'])/'manifest.json')==cfg['manifest_sha256'];write(R/('deployment-'+pkg+'.json'),cfg)
 write(R/'device_owner.json',dict(experiment='EXP-0280',exclusive=True,models_verified_by=str(R.parent)))
def slices():
 assert read(R/'single_gate.json')['pass_all'];out=[]
 for x in ARMS:
  arm(x);tag='slice-'+x;rep=tag+'-r10';d.run('sp2-fp32',tag,count=3,fp32=2,dump=True);d.run('sp2-fp32',rep,count=3,repeat=10,fp32=2);aa.chain(tag,rep);out.append(read(R/tag/'chain_gate.json'))
 write(R/'slice_gate.json',dict(pass_all=True,runs=out))
def fullgates():
 prior=R.parent.parent/'exp0272/full-fp32-audit';ref=read(prior/'validated.json');out=[];names=[p.name for p in prior.glob('*.bin') if p.name!='eval.bin'];assert len(names)==88
 for x in ARMS:
  arm(x);tag='full-'+x+'-audit';z=f.full(2,1,tag,audit=True);assert z['selected_codes']==ref['selected_codes']
  for n in names:assert sha(R/tag/n)==sha(prior/n),(x,n)
  aa.head(tag);out.append(dict(arm=x,pass_all=True,exact_files=len(names)))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all'];ref=read(R/'full-OPT-audit/validated.json')['selected_codes'];keys=['OPT','ORIGINAL','NORM_COMPACT']
 for x in keys:
  arm(x);z=f.full(2,1,'aux-'+x+'-r1');assert z['selected_codes']==ref
 for phase,count in [('short',5),('formal',10)]:
  out=[]
  for cycle in range(count):
   shift=cycle%len(keys)
   for x in keys[shift:]+keys[:shift]:
    arm(x);z=f.full(2,10,f'{phase}/{cycle:02d}-{x}');assert z['selected_codes']==ref;out.append(dict(arm=x,cycle=cycle,**z))
  write(R/(phase+'.json'),dict(pass_all=True,runs=out))
if __name__=='__main__':
 if sys.argv[1]=='stage':d.stage(int(sys.argv[2]))
 else:dict(prepare=prepare,reuse=reuse,selected=selected,diagnose=diagnose,slices=slices,fullgates=fullgates,timing=timing)[sys.argv[1]]()
