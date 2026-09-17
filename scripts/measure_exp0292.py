"""EXP0292 samebinary FP16/FP32 residual comparison; old baseline preserved."""
import sys,statistics
import device_exp0292 as d
from common_exp0289 import Path,read,write,sha
R=d.R;OLD=R.parent/'exp0289';reader=d.read;newO=d.O
def signatures(path):
 return [(v['selected_token_id'],v['selected_logit_half_bits']) for v in d.records(path/'stdout.jsonl') if 'generation_step' in v and v.get('record')!='generation_profile']
def one(case,tag,repeat=1,audit=False):
 assert case in ['fp16','fp32'];fp32=case=='fp32'
 d.O=newO if fp32 else newO.parent/'exp0289'
 def patched(p):
  p=Path(p)
  if not fp32 and p.name in ['deployment-f16f16.json','f16f16-reference.json']:return reader(OLD/p.name)
  return reader(p)
 d.read=patched
 try:out=d.run('f16f16',tag,count=28,repeat=repeat,audit=audit,full=True,f16opt=4,fp32=int(fp32))
 except AssertionError:
  if not audit:raise
  out=read(R/tag/'validated.json')
  assert out['physical_pass'] and out['numerical_pass'] is False and out['independent_output']
 finally:d.read=reader;d.O=newO
 sig=signatures(R/tag);assert len(sig)==43*repeat
 if not fp32:
  assert sig==signatures(OLD/'full-f16-a01')*repeat
  if audit:
   files=sorted((OLD/'full-f16-a01').glob('generation*_f16.bin'));assert len(files)==2451
   for f in files:assert sha(R/tag/f.name)==sha(f),f.name
   out['original_byte_exact_files']=len(files)
 elif not audit:
  assert sig==signatures(R/'audit/fp32')*repeat
 ps=[v for v in d.records(R/tag/'stdout.jsonl') if v.get('record')=='generation_profile']
 fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
 for i,p in enumerate(ps):
  step=i%43;assert p['generation_step']==step
  assert p['fp32_residual']==int(fp32) and p['projection_failure_result']==0
  if not audit:assert p['boundary_ddr_write_bytes']==0
  assert p['generation_lm_head_prefetch_count']==593
  for layer in range(28):
   z=p[f'slice_layer_{layer}']
   assert z['status']==3 and z['cache_valid_before']==(0 if step==0 else 63+step) and z['cache_valid_after']==64+step
   assert z['hidden_ddr_read_bytes']==z['hidden_ddr_write_bytes']==z['layer_unattributed_ticks']==0
   assert sum(z[k] for k in fields)==z['layer_ticks']
 out.update(case=case,tag=tag,repeat=repeat,prefill_tps=64e9/out['prefill_ns'],decode_tps=1e9/out['decode_ns'],runtime=read(R/'runtime-l28.json'))
 write(R/(tag+'-checks.json'),out);print('CASE_PASS',case,out['prefill_tps'],out['decode_tps'],flush=True)
 return out
if __name__=='__main__':
 if sys.argv[1]=='audit':one(sys.argv[2],'audit/'+sys.argv[2],1,True)
 elif sys.argv[1]=='one':one(sys.argv[2],sys.argv[3],int(sys.argv[4]))
 elif sys.argv[1]=='campaign':
  assert read(R/'head-boundary-checks.json')['pass_all']
  assert read(R/'audit/fp16-checks.json')['original_byte_exact_files']==2451
  assert read(R/'repeat-audit-checks.json')['bytewise_equal']
  write(R/'repeat1.json',[one(c,'repeat1/'+c,1) for c in ['fp16','fp32']])
  for phase,n in [('short',5),('formal',10)]:
   rows=[]
   for i in range(n):
    seq=['fp16','fp32'] if i%2==0 else ['fp32','fp16']
    block=[one(c,f'{phase}/{i:02d}-{c}',10) for c in seq]
    rows+=block;write(R/f'{phase}-round-{i:02d}.json',block)
    print('ROUND_DONE',phase,i,flush=True)
   write(R/(phase+'.json'),rows)
