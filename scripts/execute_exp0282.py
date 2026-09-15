"""Frozen latest-kernel rerun; old evidence read-only, each phase has exclusive outputs."""
import importlib,os,sys,subprocess,shlex,shutil,json
from pathlib import Path
S=Path('/home/daniuniu/work/qwen3-block-htp');ROOT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0282')
PHASES={'A1':('0274','exp0274/consumer-row1-a02'),'A2':('0275','exp0275'),'A3':('0276','exp0276'),'A5':('0277','exp0277'),'A9':('0278_vsum','exp0278/vsum')}
LEDGERS={'A1':'ce6873fee5fb0c66d0fdb63043341b7353aaf664875580b6b8fa00e25978f5d6','A2':'75b66ce1ef9f3e3c7491b63b22106199aaa51d0ac81a2d62a366f9e201336d3b','A3':'07fa8eb7877a063519f0b450fcd087e33506ae5931591644cae68bcc93077814','A5':'160ef619a2017a11913246d7459ca031c14f3fa00f7a9dedf9da5fb11e6a8eef','A9':'dc52fb7ad240babd04f54d0b06363e148ff0e41365213b6738410a03ee9f13c2'}
def preflight():
 t=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'EXPERIMENT=EXP-0282' in t

def configure(phase):
 num,old=PHASES[phase];e=importlib.import_module('execute_exp'+num);mods=[]
 for stem in ['common','device','selected_audit','audit','full']:
  mods.append(importlib.import_module(stem+'_exp'+num))
 for m in mods+[e]:m.R=ROOT/phase;m.preflight=preflight
 c,d,a,aa,f=mods;d.REMOTE='/data/local/tmp/qwen3-block-htp/exp0282-'+phase
 # Preserve immutable prior proof paths rather than deriving them from new result nesting.
 e.PRIOR=Path('/mnt/d/llm_exp/results/qwen3-block-htp')/({'A2':'exp0274/consumer-row1-a02','A3':'exp0275','A5':'exp0276','A9':'exp0277'}.get(phase,old))
 os.environ.update(QBH_SP2='8',QBH_WIDE_SCORE='4',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FIXED_TOKENS='0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0')
 return e,c,d,a,aa,f

def prepare():
 preflight();ROOT.mkdir(exist_ok=False);verified={};refs=[]
 for phase,(num,old) in PHASES.items():
  e,c,d,a,aa,f=configure(phase);e.R.mkdir();prior=ROOT.parent/old;ledger=prior/'EVIDENCE_SHA256.json'
  if not ledger.exists():ledger=prior.parent/'EVIDENCE_SHA256.json'
  assert c.sha(ledger)==LEDGERS[phase];entries=c.read(ledger)['files']
  for p in list(prior.glob('deployment-*.json'))+list(prior.glob('fixed_tokens.json')):
   key=str(p.relative_to(ledger.parent));assert c.sha(p)==entries[key]['sha256'];shutil.copy2(p,e.R/p.name)
   if not p.name.startswith('deployment-'):continue
   cfg=c.read(p);name=p.stem.removeprefix('deployment-');pkg=c.package_path(name);mf=pkg/'manifest.json';assert c.sha(mf)==cfg['manifest_sha256'];manifest=c.read(mf)
   signature=(str(pkg),cfg['remote'])
   if signature not in verified:
    for n,v in manifest['files'].items():assert c.sha(pkg/n)==v['sha256'],(pkg,n)
    names=[n for n in manifest['files'] if not n.endswith('.npy')]
    for i in range(0,len(names),32):
     lines=d.adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
     for line in lines:
      h,n=line.split(None,1);assert h==manifest['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
    verified[signature]=True;print('MODEL_VERIFIED',pkg,flush=True)
  refs.append(dict(phase=phase,prior=str(prior),ledger_sha256=c.sha(ledger)))
 c.write(ROOT/'provenance.json',dict(pass_all=True,prior=refs,models=len(verified),generalization=False))

def component():
 e,c,d,a,aa,f=configure('A1');x=importlib.import_module('execute_exp0280');x.R=e.R;x.preflight=preflight;x.d=d;x.package_path=c.package_path
 x.component()

def action(phase,what,count=0):
 e,c,d,a,aa,f=configure(phase)
 if what=='stage':d.stage(count)
 elif what=='selected':e.selected()
 elif what=='slices':e.slices()
 elif what=='fullgates':
  # EXP0274 derives original proof paths locally; the equivalent fixed explicit proof is handled here.
  if phase=='A1':
   ref=ROOT.parent/'exp0272/full-fp32-audit';out=[];names=[p.name for p in ref.glob('*.bin') if p.name!='eval.bin'];assert len(names)==88
   for arm in e.ARMS:
    e.setarm(arm);tag='full-'+arm+'-audit';z=f.full(2,1,tag,audit=True);assert z['selected_codes']==c.read(ref/'validated.json')['selected_codes']
    for n in names:assert c.sha(e.R/tag/n)==c.sha(ref/n),(arm,n)
    aa.head(tag);out.append(dict(arm=arm,**z))
   c.write(e.R/'full_gate.json',dict(pass_all=True,runs=out))
  else:e.fullgates()
 elif what=='timing':
  if phase in ['A2','A3']:
   if count!=28:
    out=[]
    for arm in e.ARMS:
     e.setarm(arm);pkg='layer14-fp32-a01' if count==1 else 'sp2-fp32';out.append(dict(arm=arm,**d.run(pkg,f'l{count}-aux-{arm}-r1',count=count,repeat=1,fp32=2)))
    c.write(e.R/f'l{count}_auxiliary.json',out)
   e.timing(count)
  else:e.timing()
 else:raise ValueError(what)
 print('ACTION_COMPLETE',phase,what,count,flush=True)

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='component':component()
 else:action(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 0)
