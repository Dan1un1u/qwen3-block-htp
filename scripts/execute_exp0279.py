"""Full-model same-arithmetic pipeline ablation at longer supported KV."""
import os,sys,shlex,subprocess
from common_exp0279 import *
from device_exp0279 import adb,stage
from full_exp0279 import full
from audit_exp0279 import head
PRIOR=R.parent/'exp0278';ARMS={'ALL':0,'FFN':2}
def setarm(a):os.environ.update(QBH_SP2='8',QBH_WIDE_SCORE='4',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE=str(ARMS[a]),QBH_PAPER_FIXED_TOKENS='0')
def prepare():
 preflight();R.mkdir(exist_ok=True);ledger=PRIOR/'EVIDENCE_SHA256.json';assert sha(ledger)=='dc52fb7ad240babd04f54d0b06363e148ff0e41365213b6738410a03ee9f13c2'
 for n,v in read(ledger)['files'].items():assert sha(PRIOR/n)==v['sha256'],n
 cfg=read(PRIOR/'vsum/deployment-sp2-fp32.json');p=O/'sp2-fp32';assert sha(p/'manifest.json')==cfg['manifest_sha256'];m=read(p/'manifest.json')
 for n,v in m['files'].items():assert sha(p/n)==v['sha256'],n
 for i in range(33):
  for kind in ['cos','sin']:assert f'generation_decode_rope_{kind}_{i:02d}_f16.bin' in m['files']
 diff=subprocess.check_output(['git','-C',str(S),'diff','140454fd24b4fa2f8ca0d3c4dced41aefcb02529','HEAD','--','src/dsp','include'],text=True);assert not diff
 for name in ['single_gate.json','slice_gate.json','full_gate.json','component_gate.json']:assert read(PRIOR/'vsum'/name)['pass_all']
 write(R/'inherited_gates.json',dict(pass_all=True,source_native_exact='140454fd24b4fa2f8ca0d3c4dced41aefcb02529',previous_ledger_sha256=sha(ledger),scope='prior selected14/chain3 own exact, NR64valid128 component; only host greedy repeat guard changes, new full34 actual tail and paired hidden gates required'))
 write(R/'deployment-sp2-fp32.json',cfg);write(R/'fixed_tokens.json',read(PRIOR/'vsum/fixed_tokens.json'));write(R/'local_models_verified.json',dict(pass_all=True,original_package_unchanged=True,generation_steps=34))
 print('LOCAL_PREPARED_NO_DEVICE',flush=True)
def verify_device():
 preflight();cfg=read(R/'deployment-sp2-fp32.json');m=read(O/'sp2-fp32/manifest.json');names=list(m['files'])
 for i in range(0,len(names),32):
  ls=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(ls)==len(names[i:i+32])
  for l in ls:
   h,n=l.split(None,1);assert h==m['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
 write(R/'device_model_gate.json',dict(pass_all=True))
def gates():
 assert read(R/'device_model_gate.json')['pass_all'];runs=[]
 for a in ARMS:
  setarm(a);tag='full-'+a+'-audit';z=full(2,1,tag,audit=True);head(tag);assert z['selected_codes'][:16]==read(PRIOR/'vsum/full-LOG2-greedy-audit/validated.json')['selected_codes'];runs.append(z)
 assert runs[0]['selected_codes']==runs[1]['selected_codes']
 for step in range(34):
  for n in [f'generation_hidden_step{step:02d}_f32.bin',f'generation_norm_step{step:02d}_u8_native.bin']:assert (R/'full-ALL-audit'/n).read_bytes()==(R/'full-FFN-audit'/n).read_bytes(),n
 write(R/'full_gate.json',dict(pass_all=True,runs=runs,all34_paired_hidden_exact=True,all34_own_finalnorm_head_exact=True))
def timing():
 assert read(R/'full_gate.json')['pass_all'];expected=read(R/'full-ALL-audit/validated.json')['selected_codes']
 def one(a,tag,rep):
  setarm(a);z=full(2,rep,tag);assert z['selected_codes']==expected;return dict(arm=a,**z)
 write(R/'auxiliary.json',[one(a,'aux-'+a+'-r1',1) for a in ARMS])
 for phase,n in [('short',5),('formal',10)]:
  out=[];keys=list(ARMS)
  for c in range(n):
   for a in (keys if c%2==0 else keys[::-1]):out.append(dict(cycle=c,**one(a,f'{phase}/{c:02d}-{a}',10)))
  write(R/(phase+'.json'),dict(pass_all=True,runs=out))
if __name__=='__main__':
 if sys.argv[1]=='stage':stage(28)
 else:{'prepare':prepare,'verify_device':verify_device,'gate':gates,'timing':timing}[sys.argv[1]]()
