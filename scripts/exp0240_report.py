#!/usr/bin/env python3
"""Freeze EXP0240 report from retained final rotated pairs."""
import json,statistics,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
from exp0240_device import RESULT as R, SOURCE as S, MODELS
FIELDS=['input_stage_ticks','metadata_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','output_stage_ticks','scan_cache_pack_ticks','scan_cache_append_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','stage_boundary_ticks','runtime_setup_ticks','runtime_teardown_ticks']
MODULES=[('I/O、metadata',['input_stage_ticks','metadata_stage_ticks','output_stage_ticks']),('Input RMSNorm',['input_norm_ticks']),('QKV＋Q/K Norm-RoPE',['qkv_projection_ticks','qk_norm_rope_ticks']),('QK–Softmax–AV',['attention_ticks']),('O projection',['o_projection_ticks']),('Post-attention residual＋RMSNorm',['post_attention_residual_ticks','post_attention_norm_ticks']),('Gate/Up＋SwiGLU',['gate_up_ticks','activation_ticks']),('Down',['down_ticks']),('Final residual',['final_residual_ticks']),('KV carrier conversion',['scan_cache_pack_ticks']),('KV append DMA',['scan_cache_append_ticks']),('Block orchestration',['block_orchestration_ticks']),('Layer bookkeeping',['layer_bookkeeping_ticks']),('Stage-boundary bookkeeping',['stage_boundary_ticks']),('DSP unattributed',['ledger_unattributed_ticks']),('Runtime setup/teardown',['runtime_setup_ticks','runtime_teardown_ticks'])]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def records(tag):return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if '"record":"exp0240_profile"' in l]
def means(rs):
 keys=[k for k,v in rs[0].items() if type(v) in [int,float]]
 return {k:statistics.mean(x[k] for x in rs) for k in keys}
def coherent_center(runs):
 order=sorted(range(len(runs)),key=lambda i:runs[i]['host_wall_ns']);selected=order[4:6]
 return {k:statistics.mean(runs[i][k] for i in selected) for k in runs[0]},selected

def main():
 provenance=json.loads((R/'final_formal_provenance.json').read_text());aud=json.loads((R/'projection_audit_02_integer_audit.json').read_text());assert aud['pass']
 result={'experiment':'EXP-0240','source_branch':'codex/exp-0240-w4a8-lpbq32-layer-gate','formal_runtime_source':provenance['source_head'],'reporting_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),'quantization':'AIMET LPBQ signed4/decompressed8 Kgroup32; minmax [-8,7]; multiplier1..16; common FP32 scale/output','runtime':'compressed W4 plus packed integer multipliers DMA to VTCM, HVX reconstruct S8, U8xS8 HMX','rounds':{'short':5,'formal':10},'threshold_percent':10,'full_model_executed':False,'baseline_promoted':False,'quality_evaluated':False,'correctness':{'integer_projection_checks':len(aud['projections']),'elements':sum(x['elements'] for x in aud['projections']),'max_lsb':max(x['max_lsb'] for x in aud['projections']),'mismatches':sum(x['mismatches'] for x in aud['projections']),'bias_exact':True,'scalar_vs_hvx_layer_exact':True,'unit_multiplier_vs_direct_w4_layer_exact':True},'performance':{},'module_centers':{},'physical':{},'schedule_control':{}}
 allruns={};allrecords=[];rng=np.random.default_rng(240)
 for rep in [1,10]:
  allruns[rep]={}
  for mode in ['prefill','decode']:
   runs={c:[] for c in ['control','lpbq32']}
   for c in runs:
    for i in range(10):
     rs=records(f'final_formal_{i:02d}_r{rep}_{c}');assert len(rs)==rep*9
     for x in rs:
      assert sum(x[k] for k in FIELDS)==x['ledger_named_ticks']==x['invocation_ticks']
      assert x['ledger_unattributed_ticks']==0 and x['vtcm_acquired_bytes']==8388608
      assert x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==x['intermediate_spill_fill_count']==0
      assert x['block_invocation_count']==1
     rows=[x for x in rs if (x['replay_step']==0)==(mode=='prefill')];allrecords.extend(rows);runs[c].append(means(rows))
   a=np.array([x['host_wall_ns']/1000 for x in runs['control']]);b=np.array([x['host_wall_ns']/1000 for x in runs['lpbq32']]);ratios=b/a
   sampled=np.median(ratios[rng.integers(0,10,(50000,10))],axis=1);ci=np.quantile(sampled,[.025,.975])
   key=f'repeat{rep}_{mode}';result['performance'][key]={'control_us':float(np.median(a)),'lpbq_us':float(np.median(b)),'ratio_of_medians':float(np.median(b)/np.median(a)),'paired_median_ratio':float(np.median(ratios)),'paired_regression_percent':float(100*(np.median(ratios)-1)),'paired_ratio_ci95':ci.tolist(),'stable_over10percent':bool(ci[0]>1.1),'per_round_control_us':a.tolist(),'per_round_lpbq_us':b.tolist()}
   allruns[rep][mode]=runs
   if rep==10:
    result['module_centers'][mode]={}
    for c in runs:
     center,indices=coherent_center(runs[c]);host=center['host_wall_ns']/1000
     modules={name:sum(center[k] for k in fields)/19.2 for name,fields in MODULES}
     modules['Host–DSP boundary']=host-center['invocation_ticks']/19.2;modules['Complete Host wall']=host
     assert abs(sum(v for k,v in modules.items() if k!='Complete Host wall')-host)<1e-7
     result['module_centers'][mode][c]={'round_indices':indices,'host_us':host,'modules_us':modules}
 for mode in ['prefill','decode']:
  v=[]
  for i in range(5):
   rs=records(f'final_schedule_{i:02d}_matched');v.append(statistics.mean(x['host_wall_ns']/1000 for x in rs if (x['replay_step']==0)==(mode=='prefill')))
  result['schedule_control'][mode]={'median_us':statistics.median(v),'paired':False,'rounds':5,'repeat':10}
 result['physical']={k:sorted({x[k] for x in allrecords}) for k in ['vtcm_requested_bytes','vtcm_acquired_bytes','vtcm_peak_plan_bytes','weight_ddr_read_bytes','hmx_u8s8_tile_pair_count','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','block_invocation_count','ledger_unattributed_ticks']}
 result['decision']='stop_before_full_model_discuss';assert any(result['performance'][f'repeat10_{m}']['stable_over10percent'] for m in ['prefill','decode'])
 (R/'closure.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 text=['# EXP-0240 — LPBQ32 W4A8 single-layer gate','',f"Source: {result['source_branch']} @ {provenance['source_head']}. Reporting source: {result['reporting_source']}.",f'Results: {R}. Models/artifacts: {MODELS}/exp0240. Paired direct control: existing per-channel direct-W4 W4U8; candidate: fresh original Qwen3 layer14 LPBQ4/8 G32. F16F16 and W4F16 frozen. No baseline promotion.','', 'Decision: stable regression exceeds the user-selected10% threshold in both modes. Full-model execution stopped before starting. This bounds the current project-owned W4-to-S8 implementation; it is not a theoretical lower bound or a benchmark of Qualcomm QNN kernels.','', 'Each replay executes one real layer14 M64 prefill followed by8 teacher-input M1 steps with self-computed persistent KV. There is one FastRPC run per step. Repeat10 means ten complete replays in the same loaded Prepared Runtime; state/cache reset before each prefill, and each decode sequence advances once per step. It is not a frozen-snapshot repeat and not full-model token throughput. No warmup is silently discarded. Five short and ten alternating formal pairs per repeat scope; no outlier deletion.','', 'Primary values are medians of10 paired-round within-round means. Bootstrap uses50000 resamples of the10 paired ratios, seed240. Decode latency is mean complete Host wall per M1 step across all8 positions. The additive module overview uses the same two median-Host-ranked rounds for every field within each cell, preserving closure; detailed counter tables below independently median each numeric field.','', '| Scope | Control μs | LPBQ32 μs | Paired regression | Paired ratio95% CI |','|---|---:|---:|---:|---|']
 for k,v in result['performance'].items():text.append(f"|{k}|{v['control_us']:.3f}|{v['lpbq_us']:.3f}|{v['paired_regression_percent']:+.2f}%|[{v['paired_ratio_ci95'][0]:.4f}, {v['paired_ratio_ci95'][1]:.4f}]|")
 text+=['','Correctness: independent pinned AIMET grouping/scale oracle and original/control hashes pass. Fresh S8 reconstruction and seven independently derived bias tables pass. All63 projection/step checks against integer accumulation plus SDK native HMX conversion have0 mismatches across1,474,560 values. Integer products are cross-checked with separate int64 reduction. SIMD vs scalar complete-layer outputs and prefill K/V are exact; unit-multiplier LPBQ vs frozen direct-W4 are exact; all final short/formal output hashes match these frozen controls. Numeric audit is disabled during speed collection. Legacy output/cache comparison fields in exp0240_profile are inapplicable placeholders, explicitly marked historical_reference_used=false; they are not correctness evidence.','', 'Physical gates: exact8MiB requested/acquired for every cell, peak plan6,682,752 bytes, zero intermediate DDR or spill/fill, one HMX owner, standalone FastRPC, no QNN/CPU fallback. Audit-only public projection captures are excluded from formal timing; expanded weights never reside in DDR. Payload overhead is3.125% vs W4 code bytes (0.125bits/weight), and50,331,648 S8 bytes are produced inside VTCM per layer.','', 'Scheduling: LPBQ QKV uses two four-output-tile slots with three HVX expansion workers. O/Gate/Up use double-buffered DMA/HVX batches8, Down batches2; one HMX owner overlaps the previous command with next-slot DMA/expansion. Direct control retains recipe-fastest N4 batch sizes and Gate/Up streaming. The additional same-per-channel-format control disables continuous Gate/Up/prefetch/streaming and uses QKV4, O/Gate/Up8, Down2. Its five repeat10 rounds are unpaired diagnostic measurements, so they do not replace the primary paired control.']
 for mode,v in result['schedule_control'].items():text.append(f"Schedule control {mode}: {v['median_us']:.3f}μs.")
 text+=['','Recovery record: initial runner required token generation even for the isolated layer replay; the allowance is scoped to the one-layer build. The first decoder had scalar multiplier broadcasts and byte loads; final kernel uses aligned HVX metadata reads/rotation and SIMD multiply. Initial public audit copies lacked explicit DSP cache flush; unchanged complete-layer outputs and the subsequent exact native integer audit identified and resolved this capture defect. Original failed captures/results remain retained. A final telemetry-only repair corrected actual O/MLP batch counts, followed by fresh5short/10formal collections under final_* names. Earlier timings remain diagnostic.','']
 overview=[]
 for mode in ['prefill','decode']:
  centers=result['module_centers'][mode];text += [f'## {mode} repeat10 additive module overview','','F16F16/W4F16 are N/A for this new equivalent-scope paired run. Historical full-model results have different execution boundaries and are not substituted.','','|Module|F16F16|W4F16|W4U8 per-channel|W4U8 LPBQ32|LPBQ time change|','|---|---|---|---:|---:|---:|']
  for name in centers['control']['modules_us']:
   a=centers['control']['modules_us'][name];b=centers['lpbq32']['modules_us'][name];delta=f'{100*(b/a-1):+.2f}%' if a else 'N/A zero denominator'
   text.append(f"|{name}|N/A frozen|N/A frozen|{a:.3f} ({100*a/centers['control']['host_us']:.2f}%)|{b:.3f} ({100*b/centers['lpbq32']['host_us']:.2f}%)|{delta}|")
  for name in ['Embedding','Final model RMSNorm','LM head + greedy excluding final norm']:text.append(f'|{name}|N/A|N/A|N/A outside layer|N/A outside layer|N/A|')
 text+=['','## Complete diagnostics','','All engine, expansion, DMA and wait counters can overlap and must not be added to the exclusive ledger. Time fields are in qtimer ticks at19.2ticks/μs; bytes/counts retain native units. Runtime configuration fields are declared controls; actual O/MLP batch counts are corrected in final collection. Missing or out-of-scope external correctness fields are excluded here and explained above.','']
 ignored={'experiment','variant','replay_step','first_position','valid_length','repeat_count'}
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   runs=allruns[rep][mode];text += [f'### Repeat{rep} {mode}','','|Field|Control median|LPBQ median|Change|','|---|---:|---:|---:|']
   for k in runs['control'][0]:
    if k in ignored or k.startswith(('output_','cache_','fp16_')):continue
    a=statistics.median(x[k] for x in runs['control']);b=statistics.median(x[k] for x in runs['lpbq32']);delta=f'{100*(b/a-1):+.2f}%' if a else 'N/A zero denominator'
    text.append(f'|{k}|{a:.4f}|{b:.4f}|{delta}|')
 text+=['','E2E prefill/decode token/s: N/A. This experiment has no full-model token boundary; full-model work was stopped by the10% single-layer gate. No extrapolation, no PPL/quality claim.']
 (R/'full_profiling_report.md').write_text('\n'.join(text)+'\n')
 user=['单位μs，括号为完整Host wall占比；M64 prefill，decode为8个M1步骤的平均延迟。','','|模块|Prefill per-channel|Prefill LPBQ32|Decode per-channel|Decode LPBQ32|','|---|---:|---:|---:|---:|']
 for name in result['module_centers']['prefill']['control']['modules_us']:
  row=[name]
  for mode in ['prefill','decode']:
   for c in ['control','lpbq32']:
    z=result['module_centers'][mode][c];v=z['modules_us'][name];row.append(f'{v:.1f} ({100*v/z["host_us"]:.2f}%)')
  user.append('|'+'|'.join(row)+'|')
 (R/'user_module_table.md').write_text('\n'.join(user)+'\n')
 # Retain immutable executed binary copies with their formal identities.
 art=MODELS/'exp0240/artifacts'/provenance['source_head'][:12];art.mkdir(parents=True,exist_ok=True)
 for name,expected in provenance['binaries'].items():
  paths=[S/'android_ReleaseG_aarch64/ship'/name,S/'hexagon_ReleaseG_toolv19_v79/ship'/name];src=next(p for p in paths if p.exists())
  assert sha(src)==expected;shutil.copy2(src,art/name)
 (R/'artifact_manifest.json').write_text(json.dumps({'binary_directory':str(art),'binaries':provenance['binaries'],'packages':{v:sha(MODELS/'exp0240/device'/v/'device_manifest.json') for v in ['control','lpbq32','unit_multiplier']}},indent=2)+'\n')
 print(json.dumps(result['performance'],indent=2));print('REPORT',R/'full_profiling_report.md')
if __name__=='__main__':main()
