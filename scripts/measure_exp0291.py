"""EXP0291 exact-boundary controls and balanced full-model measurement."""
import sys,statistics
import device_exp0291 as d
import device_exp0260 as w
from common_exp0289 import Path,read,write,sha,subprocess
R=d.R;OLD=R.parent/'exp0289'
CASES={'original':('reference-a03/w4f16',2,'control',True),
 'f16':('f16f16',3,'control',True),
 'f16rows':('f16f16',4,'control',False),
 'control':('reference-a03/w4f16',2,'control',False),
 'rows':('reference-a03/w4f16',4,'control',False),
 'candidate':('reference-a03/w4f16',4,'control',False),
 'main':('reference-a03/w4f16',5,'control',False),
 'mainhead4':('reference-a03/w4f16',5,'head_aligned_batch4',False)}
reader=d.read
def signatures(path):
 return [(v['selected_token_id'],v['selected_logit_half_bits']) for v in d.records(path/'stdout.jsonl') if 'generation_step' in v and v.get('record')!='generation_profile']
def one(case,tag,repeat=3,audit=False):
 d.preflight()
 pkg,opt,schedule,original=CASES[case]
 state=read(R/('runtime-original.json' if original else 'runtime-l28.json'))
 d.read=lambda p:state if Path(p)==R/'runtime-l28.json' else reader(p)
 w.ENV['QBH_QKV_SCHEDULE']=schedule
 try:out=d.run(pkg,tag,count=28,repeat=repeat,audit=audit,full=True,opt=opt,f16opt=opt if pkg=="f16f16" else 3)
 except AssertionError:
  if not audit:raise
  out=read(R/tag/'validated.json')
  assert out['physical_pass'] and out['numerical_pass'] is False and out['independent_output']
 gold=OLD/('full-f16-a01' if pkg=='f16f16' else 'full-w4-a03')
 assert signatures(R/tag)==signatures(gold)*repeat,(tag,'token/logit mismatch')
 ps=[z for z in d.records(R/tag/'stdout.jsonl') if z.get('record')=='generation_profile']
 assert len(ps)==43*repeat
 fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
 for i,p in enumerate(ps):
  step=i%43
  assert p['generation_step']==step and p['projection_failure_result']==0
  if not audit:assert p['boundary_ddr_write_bytes']==0
  assert p['generation_lm_head_prefetch_count']==593
  assert p['w4f16_decode_opt_calls']==(224 if step else 0)
  for layer in range(28):
   z=p[f'slice_layer_{layer}']
   assert z['status']==3 and z['cache_valid_before']==(0 if step==0 else 63+step) and z['cache_valid_after']==64+step
   assert z['hidden_ddr_read_bytes']==z['hidden_ddr_write_bytes']==z['layer_unattributed_ticks']==0
   assert sum(z[k] for k in fields)==z['layer_ticks']
 if audit:
  files=sorted(gold.glob('generation*_f16.bin'));assert len(files)==2451
  for f in files:assert sha(R/tag/f.name)==sha(f),(case,'full boundary mismatch',f.name)
  out['bytewise_audit_files']=len(files)
 out.update(case=case,tag=tag,repeat=repeat,token_logit_exact=True,runtime=state)
 out['prefill_tps']=64e9/out['prefill_ns'];out['decode_tps']=1e9/out['decode_ns']
 out['qkv_prefill_us']=statistics.mean(p['qkv_projection_ticks']/19.2 for p in ps if p['mode']=='prefill')
 out['qkv_decode_us']=statistics.mean(p['qkv_projection_ticks']/19.2 for p in ps if p['mode']=='decode')
 write(R/(tag+'-checks.json'),out)
 print('CASE_PASS',case,out['prefill_tps'],out['decode_tps'],out['qkv_prefill_us'],out['qkv_decode_us'],flush=True)
 return out
if __name__=='__main__':
 if sys.argv[1]=='diagnostic':
  rows=[one(c,'diagnostic/'+c) for c in CASES]
  write(R/'diagnostic.json',rows)
 elif sys.argv[1]=='one':one(sys.argv[2],sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else 3)
 elif sys.argv[1]=='audit':one(sys.argv[2],sys.argv[3],1,True)
 elif sys.argv[1]=='campaign':
  winner=sys.argv[2]
  assert read(R/('audit/'+winner+'-checks.json'))['bytewise_audit_files']==2451
  assert read(R/('audit/'+winner+'-checks.json'))['runtime']==read(R/'runtime-l28.json')
  assert read(R/'audit/f16rows-checks.json')['bytewise_audit_files']==2451
  assert read(R/'audit/f16rows-checks.json')['runtime']==read(R/'runtime-l28.json')
  write(R/'repeat1.json',[one(c,'repeat1/'+c,1) for c in ['original','f16','f16rows',winner]])
  for phase,n in [('short',5),('formal',10)]:
   rows=[]
   for i in range(n):
    seq=['original','f16','f16rows',winner];order=seq[i%4:]+seq[:i%4]
    if i%2:order=order[::-1]
    block=[one(c,f'{phase}/{i:02d}-{c}',10) for c in order]
    rows+=block;write(R/f'{phase}-round-{i:02d}.json',block)
    print('ROUND_DONE',phase,i,flush=True)
   write(R/(phase+'.json'),rows)
