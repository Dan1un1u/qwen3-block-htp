#!/usr/bin/env python3
"""Independent raw-evidence closure, arithmetic means of frozen paired rounds."""
import sys,json,statistics,ast,hashlib,re,subprocess
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/qwen3-block-htp');sys.path.insert(0,str(S/'scripts'))
from common_exp0268 import R,O,P,sha,preflight
from device_exp0268 import records,physical,read
from summarize_exp0217 import normalized
from measure_exp0218 import OVERVIEW,LEDGER

def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
def flat(d,p=''):
 z={}
 for k,v in d.items():
  if isinstance(v,dict):z.update(flat(v,p+k+'.'))
  elif type(v) in [int,float]:z[p+k]=v
 return z
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
 preflight();assert all(read(R/f'{g}_gate.json')['pass_all'] for g in ['single','slice','full']);assert read(R/'layer-formal.json')['speed_pass']
 data={};counter={};loops={};calls=0;layer_ledgers=0;maxvtcm=0
 lf=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
 for phase,n in [('short',5),('formal',10)]:
  gate=read(R/f'full-{phase}.json');assert gate['pass_all'];rsums=[]
  for i in range(n):
   for rep in [1,10]:
    for m in [0,1,2,3,8]:
     p=R/f'full-{phase}/round{i:02d}-r{rep}-m{m}';rs=records(p/'stdout.jsonl');ps=[q for q in rs if q.get('record')=='generation_profile'];fs=[q for q in rs if q.get('generation_sequence_complete')];physical(ps,28);assert len(ps)==16*rep and len(fs)==rep
     cmd=(p/'command.txt').read_text();assert f'QBH_SP2={8 if m==8 else 0} ' in cmd and f'QBH_U8_PREFILL_OPT={m if m!=8 else 0} ' in cmd and 'QBH_DENSE_R3_AUDIT' not in cmd and 'QBH_REPLAY_DUMP_DIR' not in cmd
     golden=read(R/f'full-m{m}/validated.json')['selected_codes'];codes=[[q['selected_token_id'],q['selected_logit_half_bits']] for q in rs if 'selected_logit_half_bits' in q];assert codes==golden*rep
     for j,(q,f) in enumerate(zip([ps[k*16] for k in range(rep)],fs)):
      assert f['all_steps_pass'] and f['total_host_wall_ns']==sum(z['host_wall_ns'] for z in ps[j*16:j*16+16]);assert f['generation_loop_wall_ns']>=f['total_host_wall_ns']
     for q in ps:
      assert q['repeat_count']==1 and q['numerical_audit_enabled']==0 and q['u8_attention_audit_ddr_write_bytes']==0 and q['boundary_ddr_write_bytes']==0
      assert q['vtcm_peak_plan_bytes']==8365824 and q['weight_ddr_read_bytes']==866032640
      assert q['hmx_command_count']==(1882 if q['mode']=='prefill' else 1437)
      assert q['hmx_u8s8_tile_pair_count']==((2031360 if m==8 else 1687296) if q['mode']=='prefill' else 1690880)
      if q['mode']=='prefill':
       assert q['w4u8_gate_up_swiglu_publish_count']==q['w4u8_gate_up_swiglu_consume_count']==(168 if m in [2,3,8] else 0)
      maxvtcm=max(maxvtcm,q['vtcm_peak_plan_bytes'])
      for layer in range(28):
       v=q[f'slice_layer_{layer}'];assert sum(v[k] for k in lf)==v['layer_ticks'];step=q['generation_step'];assert v['cache_valid_before']==(0 if step==0 else 63+step) and v['cache_valid_after']==64+step;layer_ledgers+=1
     z=read(p/'validated.json');assert z['prefill_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill') and z['decode_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode');rsums.append(dict(round=i,**z));calls+=len(ps)
     if phase=='formal':
      key=f'm{m}_r{rep}';data.setdefault(key,{});counter.setdefault(key,{})
      for mode in ['prefill','decode']:
       qs=[q for q in ps if q['mode']==mode];data[key].setdefault(mode,[]).append(normalized(qs));a=[flat(q) for q in qs];assert all(set(v)==set(a[0]) for v in a);counter[key].setdefault(mode,[]).append({k:statistics.mean(v[k] for v in a) for k in a[0]})
      loops.setdefault(key,[]).append(dict(model_ns=statistics.mean(f['total_host_wall_ns'] for f in fs),loop_ns=statistics.mean(f['generation_loop_wall_ns'] for f in fs),startup=next(q for q in rs if q.get('record')=='generation_startup')))
  rng=np.random.default_rng(268);idx=rng.integers(0,n,(20000,n))
  for rep in [1,10]:
   for field in ['prefill_ns','decode_ns']:
    for a,b in [(0,1),(0,2),(0,3),(1,2),(2,3),(0,8),(3,8)]:
     pair=np.array([[next(z[field] for z in rsums if z['round']==i and z['repeat']==rep and z['mode']==m) for m in [a,b]] for i in range(n)]);ratio=pair[:,1].mean()/pair[:,0].mean();ci=np.quantile(pair[idx,1].mean(1)/pair[idx,0].mean(1),[.025,.975]);v=gate['performance'][f'r{rep}_m{b}_over_m{a}_{field}'];assert abs(ratio-v['ratio'])<1e-12 and np.max(abs(ci-v['ci95']))<1e-12
 assert calls==13200 and layer_ledgers==369600
 centers={};through={}
 for key,ds in data.items():
  centers[key]={}
  for mode,rounds in ds.items():
   c={k:statistics.mean(v[k] for v in rounds) for k in rounds[0]};mods={name:sum(c[f] if f.endswith('_us') else c[f]/19.2 for f in fields) for name,fields in OVERVIEW};assert abs(sum(list(mods.values())[:-1])-c['host_us'])<1e-6;centers[key][mode]=dict(host_us=c['host_us'],modules_us=mods)
  pre=centers[key]['prefill']['host_us'];dec=centers[key]['decode']['host_us'];model=statistics.mean(z['model_ns'] for z in loops[key]);loop=statistics.mean(z['loop_ns'] for z in loops[key]);assert abs(model/1000-pre-15*dec)<1e-6
  through[key]=dict(prefill_host_us=pre,decode_host_us_per_token=dec,prefill_tokens_per_second=64e6/pre,decode_tokens_per_second=1e6/dec,model_16_output_tokens_per_second=16e9/model,generation_loop_16_output_tokens_per_second=16e9/loop,model_host_us=model/1000,generation_loop_us=loop/1000)
 labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall'];tables={}
 for mode in ['prefill','decode']:
  rows=[]
  for label,(name,_) in zip(labels,OVERVIEW):
   vals=[centers[f'm{m}_r10'][mode]['modules_us'][name] for m in [0,1,2,3,8]];hosts=[centers[f'm{m}_r10'][mode]['host_us'] for m in [0,1,2,3,8]];rows.append([label,*[f'{v:.1f} ({100*v/h:.2f}%)' for v,h in zip(vals,hosts)]])
  tables[mode]=table(['模块 / μs','U8 原始','U8 三 HVX','U8 Up 重叠','U8 交错流水','SP2 优化流水'],rows);(R/f'MODULE_TABLE_{mode}.md').write_text(tables[mode]+'\n')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/Qwen3-origin',local_files_only=True);texts={str(m):dict(token_ids=[x[0] for x in read(R/f'full-m{m}/validated.json')['selected_codes']],text=tok.decode([x[0] for x in read(R/f'full-m{m}/validated.json')['selected_codes']],skip_special_tokens=False)) for m in [0,1,2,3,8]};write(R/'text_output.json',texts)
 summary=dict(experiment='EXP-0268',tested_runtime=read(R/'runtime-l28.json'),scope='Qwen3 28layers M64 plus15feedback decode, capacity128, native peroutputchannelW4, no rotations, frozen staticA8 except SP2 middle',formal_rounds=10,short_rounds=5,repeat10_primary=True,repeat1_auxiliary=True,full_model_RPCs=calls,layer_ledgers=layer_ledgers,physical_gate=True,peak_vtcm_bytes=maxvtcm,performance=read(R/'full-formal.json')['performance'],speed_pass=read(R/'full-formal.json')['speed_pass'],throughput=through,module_centers=centers,bit_exact_scheduling_gate=True,quality_gate='not evaluated',PPL=None,default_promoted=False,text_output=texts)
 write(R/'summary.json',summary);write(R/'counter_round_means.json',counter);write(R/'generation_loop_times.json',loops);write(R/'independent_integrity.json',dict(pass_all=True,full_model_RPCs=calls,layer_ledgers=layer_ledgers,all_selected_tokens_and_codes_exact=True,statistics_independently_recomputed=True,no_outlier_deletion=True,repeat1_no_veto=True))
 perf=summary['performance'];intro=['# EXP-0268 Qwen3 U8 prefill scheduling and SP2 net cost','','All five arms use the same binary and immutable EXP0267 packages. U8 m0 original; m1 three HVX contexts after Gate/Up; m2 consumes completed Up groups; m3 additionally interleaves Gate/Up batches. SP2 m8 retains the optimized two-plane producer and overlapping Down epilogue. U8 LUT gather/saturating pack and one-pass native Down unchanged. Dedicated middle allocation prevents overwrite of live Up input. Decode algorithm unchanged. SP2 still uses its own previously validated pipelined gather; this is a practical optimized-method comparison, not mathematical proof of globally optimal U8.','','Layers0/14/27 and consecutive3 live carriers/KV/output match original U8 exactly. Full28 U8 feedback tokens and selected logit codes match sealed original; SP2 matches sealed SP2. Not full-vocabulary logit or PPL evaluation.5short+10formal five rotated arms, bothrepeat1/10:13200fullmodelRPCs,369600exclusive layer ledgers. Repeat10 primary,repeat1 auxiliary. Pairedbootstrap20000 seed268, no outlier deletion or optional resampling.','','## Primary latency effects','',table(['candidate/control','wall change','95% ratioCI'],[[k,f'{100*(v["ratio"]-1):+.3f}%',f'[{v["ci95"][0]:.6f}, {v["ci95"][1]:.6f}]'] for k,v in perf.items() if k.startswith('r10_')]),'',f'OptimizedU8/originalU8 both phase CI upper<=1.10: {summary["speed_pass"]}. SP2/optimizedU8 is diagnostic net cost. NoPPL/quality/default promotion.','','## M64 modules','',tables['prefill'],'','## Decode modules pertoken','',tables['decode'],'','Text diagnostics; fixed16outputs continue afterEOS.','',json.dumps(texts,ensure_ascii=False,indent=2),'','## Actual E2E','',table(['arm/repeat','M64 token/s','decode15 token/s','16outputs/modelHost token/s','16outputs/generationloop token/s'],[[k,*[f'{v[n]:.3f}' for n in ['prefill_tokens_per_second','decode_tokens_per_second','model_16_output_tokens_per_second','generation_loop_16_output_tokens_per_second']]] for k,v in through.items()]),'','Complete embedding/28layers/finalnorm/head/greedy/FastRPC Host wall. Tokenizer,ADB,coldload excluded from hotinference; generationloop includes deviceHostloop/JSONserialization. No layer extrapolation.']
 (R/'REPORT.md').write_text('\n'.join(intro)+'\n');full=intro+['','All numeric telemetry retained below: ticks19.2/us, ns asnamed, counts/bytes raw; overlapping work is nonadditive.']
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   cs=[counter[f'm{m}_r{rep}'][mode] for m in [0,1,2,3,8]];assert all(set(x[0])==set(cs[0][0]) for x in cs);full+=['',f'## r{rep} {mode}','',table(['Field','U8old','U8parallel','U8Upstream','U8early','SP2m8'],[[k,*[f'{statistics.mean(z[k] for z in x):.6f}' for x in cs]] for k in sorted(cs[0][0])])]
 (R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(full)+'\n');print(json.dumps(dict(speed_pass=summary['speed_pass'],throughput=through,primary={k:v for k,v in perf.items() if k.startswith('r10_m8')}),indent=2))
if __name__=='__main__':main()
