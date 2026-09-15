#!/usr/bin/env python3
import sys,json,statistics,ast,hashlib
from pathlib import Path
S=Path(sys.argv[1]);sys.path.insert(0,str(S/'scripts'));import f16_baseline_refresh as f
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized,TICKS
summary=f.read(f.R/'formal-summary.json');rows=f.read(f.R/'formal-validated.json')['rows'];tables=[];diagnostics={};datasets={}
for opt in [0,3]:
 rs=[]
 for r in rows:
  if r['opt']==opt:rs.extend(q for q in f.records(f.R/r['tag']/'stdout.jsonl') if q.get('record')=='generation_profile')
 assert len(rs)==1600;datasets[opt]=rs
 for mode in ['prefill','decode']:
  ps=[q for q in rs if q['mode']==mode];v=normalized(ps)
  diagnostics[f'{opt}-{mode}']={k:v[k] for k in ['host_us','host_boundary_us','generation_final_norm_ticks','generation_lm_head_exclusive_ticks','generation_lm_head_weight_dma_ticks','generation_lm_head_hmx_ticks','generation_lm_head_argmax_ticks','generation_lm_head_prefetch_count','attention_softmax_ticks','scan_dynamic_attention_ticks','vtcm_peak_plan_bytes','weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count']}
for mode in ['prefill','decode']:
 vs={opt:normalized([q for q in datasets[opt] if q['mode']==mode]) for opt in [0,3]}
 tables.extend([f'## {mode}: F16F16 full-model modules','', 'Microseconds (share of complete Host wall); arithmetic means across all ten formal rounds and all repetitions. Throughput ranking below uses the declared median of round means.','', '| Module | Current original control | Refreshed FP16 | Wall change |','|---|---:|---:|---:|'])
 for label,keys in OVERVIEW:
  vals={opt:sum(vs[opt][k] if k in ['host_us','host_boundary_us'] else vs[opt][k]/TICKS for k in keys) for opt in [0,3]}
  delta=f"{(vals[3]/vals[0]-1)*100:+.2f}%" if vals[0] else 'N/A'
  tables.append(f"| {label} | {vals[0]:.3f} ({100*vals[0]/vs[0]['host_us']:.2f}%) | {vals[3]:.3f} ({100*vals[3]/vs[3]['host_us']:.2f}%) | {delta} |")
 tables.append('')
(f.R/'MODULE_TABLES.md').write_text('\n'.join(tables)+'\n')
work={}
for mode in ['prefill','decode']:
 aa=[q for q in datasets[0] if q['mode']==mode];bb=[q for q in datasets[3] if q['mode']==mode]
 fields=['vtcm_peak_plan_bytes','weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count','generation_lm_head_ddr_read_bytes','generation_lm_head_command_count']
 work[mode]={k:dict(control=sorted(set(q[k] for q in aa)),candidate=sorted(set(q[k] for q in bb)),equal=[q[k] for q in aa]==[q[k] for q in bb]) for k in fields}
 assert all(v['equal'] for v in work[mode].values())
f.write(f.R/'attribution.json',dict(all_work_counts_equal=True,fields=work,module_means=diagnostics))
p=summary['performance'];a=p['prefill_ns'];b=p['decode_ns'];gate=f.read(f.R/'full-control-a01-paired-gate.json')
lines=[f'# {f.EXP}: refreshed W16A16 paper baseline','',f'Model: {"Llama-3.2-1B-Instruct" if f.LLAMA else "Qwen3-1.7B"}. Complete warm M64 prefill plus 15 continuous decode passes; includes embedding, all layers, final norm, FP16 head, greedy and FastRPC. Excludes cold loading, host tokenizer and transport. Same binary, model package and prompt for both arms.','', '| Configuration | Prefill token/s | Decode token/s | Prefill Host ms | Total 15-decode Host ms |','|---|---:|---:|---:|---:|']
for key,label in [('control','Current unmodified mathematics / original schedule, OPT0'),('candidate','Exact conversion + head prefetch + consumed-row final norm, OPT3')]:lines.append(f"| {label} | {a[key+'_tokens_per_second']:.3f} | {b[key+'_tokens_per_second']:.3f} | {a[key+'_ns']/1e6:.6f} | {b[key+'_ns']*15/1e6:.6f} |")
lines+=['',f"Paired candidate/control wall ratio: prefill {a['wall_ratio']:.8f}, bootstrap95 CI {a['ci95']}; decode {b['wall_ratio']:.8f}, CI {b['ci95']}. Median of ten round means, fixed balanced AB/BA; five short rounds passed before ten formal rounds; repeat10 primary, one repeat1 auxiliary pair. All runs preserved.",'', '## Correctness and physical scope','',f"Full-model audit: {gate['bytes_exact']} hidden/Norm/cache files byte-identical across original, conversion-only and final candidates, all 16 token IDs and selected FP16 logit codes exact, every layer output hash exact. Component audit checks finite binary16 patterns and probability rounding midpoints plus actual softmax outputs. Existing single-layer / consecutive-layer checks precede full-model candidate execution. Timed runs disable audit exports and retain zero intermediate and output DDR. No PPL rerun or new model-quality claim.",'',f"VTCM grant remains 8MiB; peak {datasets[0][0]['vtcm_peak_plan_bytes']} bytes, identical across arms. Weight bytes, HMX commands and FP16 tile-pair work are equal for every matched step. Head prefetch changes from zero to {datasets[3][0]['generation_lm_head_prefetch_count']} per pass, using already allocated alternating buffers. One matrix owner and one RPC per full pass/token. See attribution.json and MODULE_TABLES.md.",'','## Changes and limits','', 'The EXP0260 exact A16 conversion/HVX copy path is enabled for FP16 stored weights. Scalar expf, ordered FP32 reduction/division and both FP16 probability roundings are preserved. FP16 head stages the next weight group while the current HMX/argmax completes. Only the final consumed prompt row is normalized for the head. Backbone M64 carrier computation has not been deeply redesigned; intrinsic FP16 weight bandwidth remains a real cost. No floating approximation change, quantization, rotation, or changes to other recipes.','', 'Implementation selector: QBH_F16F16_OPT=0 original, 2 exact decode only, 3 final candidate. Existing W4F16 flags keep their prior meanings. OPT3 is an eligible measured implementation; this report does not promote a Selected Baseline.','',f"Source build: {f.read(f.R/f'runtime-l{f.COUNT}.json')['source_head']}; package manifest {f.MANIFEST}. Full build/runtime seals, commands and hashes are retained. Historical W16 measurements are non-paired context, not the optimization denominator.",'','## Retained issues','', 'No failed hardware or numerical attempt in this Llama refresh. The scientific Python environment and existing build scripts were reused. Six independently referenced selected-layer cases and a three-layer continuous replay passed before the full-model candidate. All attempts and exact outputs are retained.','']
(f.R/'REPORT.md').write_text('\n'.join(lines))
print('REPORT_READY',f.R,flush=True)
