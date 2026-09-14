"""User-authorized bounded true full-model speed diagnostic, no short speed veto."""
from common_exp0271 import *
from device_exp0271 import adb,win,records
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
import shlex,struct,statistics,numpy as np
SEED='/data/local/tmp/qwen3-block-htp/exp0257-prefix/prefix_kv_u8.bin'

def full(fp,repeat,tag,audit=False):
 preflight();assert read(R/'slice_gate.json')['pass_all'];state=read(R/'runtime-l28.json');root=state['remote'];d=R/tag;d.mkdir(parents=True,exist_ok=False)
 import exp0240_device as old
 pkg=read(R/'deployment-sp2-fp32.json')['remote'];e=dict(old.ENV,LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_SP2='8',QBH_U8_PREFILL_OPT='0',QBH_FP32_RESIDUAL=str(fp),QBH_WIDE_SCORE='4',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_KV_CACHE_CAPACITY='128',QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS='16',QBH_PREFIX_KV='1',QBH_PREFIX_FILE=SEED)
 assert adb('shell','sha256sum '+SEED).stdout.split()[0]=='7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91'
 ids=np.fromfile(O/'sp2-fp32/generation_prompt_token_ids_u32.bin','<u4').tolist();assert len(ids)==64
 row=[0,2,16]+ids+[0]*16;f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)));remote_eval=root+'/'+tag.replace('/','_')+'.bin';adb('push',win(f),remote_eval);e['QBH_EVAL_FILE']=remote_eval
 if audit:
  e.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=root+'/'+tag.replace('/','_')+'-audit');adb('shell','mkdir '+e['QBH_GENERATION_AUDIT_DIR'])
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {pkg} W4U8 1 {old.ARGS}'
 write(d/'protocol.json',dict(source_head=state['seal']['source_head'],runtime=state,command=cmd,repeat=repeat,fp32=fp,audit=audit));z=adb('shell',cmd,check=False);(d/'stdout.jsonl').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode))
 if audit:adb('pull',e['QBH_GENERATION_AUDIT_DIR']+'/.',win(d),check=False)
 if z.returncode:raise RuntimeError((tag,z.returncode,z.stderr[-1000:],z.stdout[-2000:]))
 rs=records(d/'stdout.jsonl');ps=[v for v in rs if v.get('record')=='generation_profile'];assert len(ps)==16*repeat,(tag,len(ps))
 fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
 for step,v in enumerate(ps):
  assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
  assert v['block_invocation_count']==28 and v['dense_r3_mode']==v['dense_r4_mode']==0 and v['wide_score_mode']==4
  assert v['backend']=='standalone_fastrpc_dsp' and v['qnn']=='none'
  assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks'])
  assert v['boundary_ddr_write_bytes']==(2048*(68 if fp else 1) if audit else 0)
  n=normalized([v]);assert sum(n[k] for _,k in LEDGER)==v['invocation_ticks']
  for l in range(28):
   a=v[f'slice_layer_{l}'];assert a['status']==3 and a['layer_index']==l and sum(a[k] for k in fields)==a['layer_ticks']
   assert a['hidden_ddr_read_bytes']==a['hidden_ddr_write_bytes']==a['layer_unattributed_ticks']==0
   assert a['cache_valid_before']==(1 if step%16==0 else 64+step%16) and a['cache_valid_after']==65+step%16
 fs=[x for x in rs if x.get('generation_sequence_complete')];assert len(fs)==repeat and all(x['all_steps_pass'] for x in fs)
 codes=[(x['selected_token_id'],x['selected_logit_half_bits']) for x in rs if 'selected_logit_half_bits' in x];assert len(codes)==16*repeat and all(v==codes[i%16] for i,v in enumerate(codes))
 for i,x in enumerate(fs):assert x['total_host_wall_ns']==sum(v['host_wall_ns'] for v in ps[i*16:i*16+16])
 out=dict(pass_all=True,fp32=fp,repeat=repeat,audit=audit,profiles=len(ps),peak=max(v['vtcm_peak_plan_bytes'] for v in ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),selected_codes=[list(v) for v in codes[:16]])
 out['prefill_tps']=64e9/out['prefill_ns'];out['decode_tps']=1e9/out['decode_ns'];write(d/'validated.json',out);print('FULL_PASS',tag,out['prefill_tps'],out['decode_tps'],flush=True);return out

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--fp',type=int,required=True);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',required=True);p.add_argument('--audit',action='store_true');a=p.parse_args();full(a.fp,a.repeat,a.tag,a.audit)
