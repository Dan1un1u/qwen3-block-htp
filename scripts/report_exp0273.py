"""Reconstruct A0 profiles and inherited-reference checks from retained raw data."""
from common_exp0273 import *
from device_exp0273 import records
from summarize_exp0217 import normalized
import statistics
GROUPS=[('I/O、metadata',['input_stage_ticks','metadata_stage_ticks','output_stage_ticks']),('Input RMSNorm',['input_norm_ticks']),('QKV＋Q/K Norm-RoPE',['qkv_projection_ticks','qk_norm_rope_ticks']),('QK–Softmax–AV',['attention_ticks']),('O projection',['o_projection_ticks']),('Post-attention residual＋RMSNorm',['post_attention_residual_ticks','post_attention_norm_ticks']),('Gate/Up＋SwiGLU',['gate_up_ticks','activation_ticks']),('Down',['down_ticks']),('Final residual',['final_residual_ticks']),('KV carrier conversion',['scan_cache_pack_ticks']),('KV append DMA',['scan_cache_append_ticks']),('Block orchestration',['block_orchestration_ticks']),('Layer bookkeeping',['layer_bookkeeping_ticks']),('Stage-boundary bookkeeping',['stage_boundary_ticks']),('DSP unattributed',['ledger_unattributed_ticks']),('Runtime setup/teardown',['runtime_setup_ticks','runtime_teardown_ticks']),('Embedding',['generation_embedding_ticks']),('Final model RMSNorm',['generation_final_norm_ticks']),('LM head＋greedy，不含 final norm',['generation_lm_head_exclusive_ticks'])]
def main():
 z=read(R/'baseline_reproduction.json');assert z['pass_all']; ps=[]
 for i in range(1,6):
  ps += [x for x in records(R/f'baseline-{i:02d}-r10/stdout.jsonl') if x.get('record')=='generation_profile']
 assert len(ps)==800
 out={'experiment':'EXP-0273','phase':'A0 Qwen baseline reproduction only','comparative_formal_gate':False,'hardware_cli':6,'profiles':816,'formal_reproduction_profiles':800,'native_source_unchanged':True,'fp32_mode':2,'integer_residual_tested':False,'rotation':False,'all_selected_token_logit_codes_match_EXP0272':True,'peak_vtcm_bytes':max(p['vtcm_peak_plan_bytes'] for p in ps),'PPL':None,'pending':'All comparative ablations including softmax; Llama A0'}
 for mode in ['prefill','decode']:
  values=[p for p in ps if p['mode']==mode]; n=[normalized([p]) for p in values];host=statistics.mean(p['host_wall_ns']/1000 for p in values)
  rows=[dict(module=name,us=statistics.mean(sum(p[k] for k in keys)/19.2 for p in n)) for name,keys in GROUPS]
  rows += [dict(module='Host–DSP 边界',us=host-sum(p['us'] for p in rows)),dict(module='完整 Host wall',us=host)]
  for row in rows:row['host_share_pct']=100*row['us']/host
  assert abs(sum(p['us'] for p in rows[:-1])-host)<1e-7
  out[mode]={'host_us':host,'tps':(64 if mode=='prefill' else 1)*1e6/host,'modules':rows}
 write(R/'SUMMARY.json',out)
 lines=['# EXP0273 A0：Qwen 基线复现','', '固定 EXP0272 无旋转 SP2mode8 + FP32 residual mode2，原生 src/include 未变。1 次 repeat1 辅助、5 次 repeat10 复现；不是十轮配对消融或新优化验收。已验证旧证据922文件、复用权重本地和设备 hash，新构建独立封存。原独立单层/chain3证明在原生树相同的前提下显式继承；本轮全部16步 token/logit 与旧封存逐项一致。','', '6 CLI / 816 profiles；8MiB VTCM，峰值8365824B，中间 tensor DDR/spill=0。未运行整数残差、PPL、softmax 或其他对比臂。Llama 与后续消融仍待执行。','', '| 模块 | Prefill μs（Host占比） | Decode μs/token（Host占比） |','|---|---:|---:|']
 for a,b in zip(out['prefill']['modules'],out['decode']['modules']):lines.append(f"| {a['module']} | {a['us']:.1f} ({a['host_share_pct']:.2f}%) | {b['us']:.1f} ({b['host_share_pct']:.2f}%) |")
 lines += ['',f"E2E：prefill {out['prefill']['tps']:.4f} token/s；decode {out['decode']['tps']:.4f} token/s。",'', '范围：M64+15，28层，cache128，冻结EOS前缀，warm embedding/transformer/finalnorm/head/greedy/FastRPC；不含外部tokenizer和冷加载。与历史速度只能说明复现接近，不能按非配对差值声称优化收益。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print({m:out[m]['tps'] for m in ['prefill','decode']})
if __name__=='__main__':main()
