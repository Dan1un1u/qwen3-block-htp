"""EXP0290: configuration-only diagnosis of Qwen3-0.6B W4F16 speed."""
import sys,statistics,shlex
import numpy as np
import yaml
import device_exp0289 as d
import device_exp0260 as w
from common_exp0289 import S,O,sha,read,write,Path,subprocess
OLD=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0289')
R=OLD.parent/'exp0290'
PKG='reference-a03/w4f16'
CASES={'control':('control',32),'head4':('head_aligned_batch4',32),
       'row':('control',32),'head4row':('head_aligned_batch4',32)}
BASE_ARGS=w.ARGS
def preflight():
 z=subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],check=True,capture_output=True,text=True)
 assert 'EXPERIMENT=EXP-0290\n' in z.stdout
d.R=R;d.preflight=preflight
def signatures(path):
 return [(v['selected_token_id'],v['selected_logit_half_bits']) for v in d.records(path/'stdout.jsonl') if 'generation_step' in v and v.get('record')!='generation_profile']
def prepare():
 preflight();R.mkdir(exist_ok=True)
 assert yaml.safe_load(Path('/home/daniuniu/work/llama32-htp-project-memory/PROJECT_STATUS.yaml').read_text())['governance']['active_experiment'] is None
 assert not d.adb('shell','pidof qwen3_block_cli llama_sp2_cli',check=False).stdout.strip()
 state=read(OLD/'full-w4-a03/protocol.json')['runtime']
 for n,h in state['seal']['files'].items():
  f=Path(state['archive'])/Path(n).name
  assert sha(f)==h
  assert d.adb('shell','sha256sum '+state['remote']+'/'+f.name).stdout.split()[0]==h
 dep=read(OLD/('deployment-'+PKG+'.json'));m=read(O/PKG/'manifest.json')
 assert sha(O/PKG/'manifest.json')==dep['manifest_sha256']
 for n,v in m['files'].items():assert sha(O/PKG/n)==v['sha256'],n
 names=list(m['files'])
 for start in range(0,len(names),32):
  txt=d.adb('shell','sha256sum '+' '.join(shlex.quote(dep['remote']+'/'+n) for n in names[start:start+32])).stdout
  for line in txt.splitlines():
   h,n=line.split(None,1);assert h==m['files'][n.removeprefix(dep['remote']+'/')]['sha256']
 write(R/'runtime-l28.json',state);write(R/('deployment-'+PKG+'.json'),dep)
 write(R/(PKG+'-reference.json'),read(OLD/(PKG+'-reference.json')))
 write(R/'provenance.json',dict(pass_all=True,package_files=len(names),model_sha256=dep['manifest_sha256'],runtime=state,full_floating_alignment_pass=False))
 print('PROVENANCE_PASS',flush=True)
def one(case,tag,repeat=3,audit=False):
 preflight();schedule,region=CASES[case]
 w.ENV['QBH_QKV_SCHEDULE']=schedule
 args=BASE_ARGS.split();args[1]=str(region);w.ARGS=' '.join(args)
 if case in ('row','head4row'):w.ARGS=w.ARGS.replace('qkv_norms','norms')
 try:
  out=d.run(PKG,tag,count=28,repeat=repeat,audit=audit,full=True)
 except AssertionError:
  if not audit:raise
  # Only the explicitly retained independent floating-reference failure is allowed.
  out=read(R/tag/'validated.json')
  assert out['physical_pass'] and out['numerical_pass'] is False and out['independent_output']
 sig=signatures(R/tag)
 assert sig==signatures(OLD/'full-w4-a03')*repeat,(case,'token/logit mismatch')
 ps=[v for v in d.records(R/tag/'stdout.jsonl') if v.get('record')=='generation_profile']
 assert len(ps)==43*repeat
 for i,p in enumerate(ps):
  assert p['generation_step']==i%43
  assert p['boundary_ddr_write_bytes']==(p['boundary_ddr_write_bytes'] if audit else 0)
  assert p['generation_lm_head_prefetch_count']==593
  assert p['w4f16_decode_opt_calls']==(224 if p['mode']=='decode' else 0)
  for layer in range(28):
   z=p[f'slice_layer_{layer}'];step=i%43
   assert z['status']==3 and z['cache_valid_before']==(0 if step==0 else 63+step) and z['cache_valid_after']==64+step
   assert z['hidden_ddr_read_bytes']==z['hidden_ddr_write_bytes']==z['layer_unattributed_ticks']==0
 if audit:
  files=sorted((OLD/'full-w4-a03').glob('generation*_f16.bin'));assert len(files)==2451,len(files)
  for f in files:
   g=R/tag/f.name;assert g.exists() and sha(g)==sha(f),(case,'full boundary mismatch',f.name)
  out['bytewise_audit_files']=len(files)
 out.update(case=case,tag=tag,repeat=repeat,token_logit_exact=True,config=dict(qkv_schedule=schedule,region_tiles=region))
 out['prefill_tps']=64e9/out['prefill_ns'];out['decode_tps']=1e9/out['decode_ns']
 out['qkv_prefill_us']=statistics.mean(p['qkv_projection_ticks']/19.2 for p in ps if p['mode']=='prefill')
 out['qkv_decode_us']=statistics.mean(p['qkv_projection_ticks']/19.2 for p in ps if p['mode']=='decode')
 write(R/(tag+'-checks.json'),out)
 print('CASE_PASS',case,out['prefill_tps'],out['decode_tps'],out['qkv_prefill_us'],out['qkv_decode_us'],flush=True)
 return out
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='diagnostic':
  rows=[]
  for name in CASES:rows.append(one(name,'diagnostic/'+name))
  write(R/'diagnostic.json',rows)
 elif sys.argv[1]=='one':one(sys.argv[2],sys.argv[3],3)
 elif sys.argv[1]=='audit':one(sys.argv[2],'audit/'+sys.argv[2],1,True)
 elif sys.argv[1]=='campaign':
  winner=sys.argv[2];assert read(R/('audit/'+winner+'-checks.json'))['bytewise_audit_files']==2451
  for phase,n in [('short',5),('formal',10)]:
   rows=[]
   for i in range(n):
    order=['control',winner] if i%2==0 else [winner,'control']
    block=[one(c,f'{phase}/{i:02d}-{c}',10) for c in order]
    rows+=block;write(R/f'{phase}-round-{i:02d}.json',block)
    print('ROUND_DONE',phase,i,flush=True)
   write(R/(phase+'.json'),rows)
