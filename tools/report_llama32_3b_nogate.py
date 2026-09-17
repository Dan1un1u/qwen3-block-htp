#!/usr/bin/env python3
"""Summarize the prespecified long-decode campaign, preserving all rounds."""
import json,statistics
from pathlib import Path
import numpy as np
from report_llama32_pipeline_profile import MODULES
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0045')
def read(p):return json.loads(p.read_text())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def phase(name,rounds,decode):
 assert read(R/(name+'-paired.json'))['rounds']==rounds
 result=dict(rounds=rounds,repeat=10,prefill_tokens=64,decode_tokens=decode,cache_capacity=128,arms={},effects={})
 groups={}
 for arm in ['CONTROL','OPT']:
  gs=[]
  for i in range(rounds):
   d=R/name/f'{i:02d}-{arm}';assert read(d/'validated.json')['pass_all']
   ps=[x for x in read(d/'records.json') if isinstance(x,dict) and x.get('record')=='generation_profile']
   assert len(ps)==(decode+1)*10
   gs.append(ps)
  groups[arm]=gs;result['arms'][arm]={}
  for mode in ['prefill','decode']:
   selected=[[v for v in g if v['mode']==mode] for g in gs]
   host=[statistics.mean(v['host_wall_ns']/1000 for v in g) for g in selected];wall=statistics.mean(host)
   rows=[]
   for title,keys in MODULES:
    value=statistics.mean(statistics.mean((sum(v[k] for k in keys)-(v['generation_final_norm_ticks'] if keys==['generation_lm_head_ticks'] else 0))/19.2 for v in g) for g in selected)
    rows.append(dict(module=title,us=value,share=100*value/wall))
   boundary=wall-sum(v['us'] for v in rows);assert boundary>=0
   rows+=[dict(module='Host–DSP 边界',us=boundary,share=100*boundary/wall),dict(module='完整 Host wall',us=wall,share=100)]
   detail={k:statistics.mean(statistics.mean(v[k]/19.2 for v in g) for g in selected) for k in ['u8_attention_k_pack_ticks','u8_attention_v_pack_ticks','u8_attention_softmax_ticks','generation_lm_head_weight_dma_wait_ticks','generation_lm_head_hmx_tail_wait_ticks']}
   result['arms'][arm][mode]=dict(tps=(64 if mode=='prefill' else 1)*1e6/wall,host_us=wall,round_host_us=host,modules=rows,details=detail)
  # First/later positions reveal KV-length sensitivity, not per-position promotion.
  result['arms'][arm]['decode_windows']=[]
  for lo,hi in [(1,min(15,decode)),(16,min(32,decode)),(33,decode)]:
   if lo>hi:continue
   vals=[v['host_wall_ns']/1000 for g in gs for start in range(0,len(g),decode+1) for v in g[start+lo:start+hi+1]]
   result['arms'][arm]['decode_windows'].append(dict(first=lo,last=hi,tps=1e6/statistics.mean(vals)))
 for a,b in zip(groups['CONTROL'],groups['OPT']):
  for x,y in zip(a,b):
   for k in ['weight_ddr_read_bytes','block_invocation_count','hmx_u8s8_tile_pair_count','scan_cache_ddr_write_bytes']:assert x[k]==y[k],k
   assert x['generation_lm_head_command_count']==y['generation_lm_head_command_count']
   assert x['hmx_command_count']-y['hmx_command_count']==980
 rng=np.random.default_rng(320045+decode)
 for mode in ['prefill','decode']:
  a=np.array(result['arms']['CONTROL'][mode]['round_host_us']);b=np.array(result['arms']['OPT'][mode]['round_host_us'])
  ix=rng.integers(0,rounds,(20000,rounds));ci=np.quantile(b[ix].mean(1)/a[ix].mean(1),[.025,.975]).tolist();ratio=float(b.mean()/a.mean())
  result['effects'][mode]=dict(wall_ratio=ratio,ci95=ci,throughput_gain_percent=100*(1/ratio-1))
 result['vtcm_peak_bytes']=max(v['vtcm_peak_plan_bytes'] for gs in groups.values() for g in gs for v in g)
 result['weight_bytes_identical']=True;result['physical_tile_pairs_identical']=True
 return result
def main():
 result=dict(experiment='L32-0045',model='Llama-3.2-3B-Instruct',recipe='W4A8-SP2mode8 FP32 residual, no rotation; AVrows4; both head32; candidate QKV32 plus Norm/QKV/Down boundary DMA overlap',quality_claim=False,baseline_promoted=False)
 result['short']=phase('short',5,42);result['formal']=phase('formal',10,42)
 result['performance_gate_applied']=False
 result['control_source']=read(R/'control-l28/runtime.json')['seal']['source_head'];result['measured_source']=read(R/'final-l28/runtime.json')['seal']['source_head']
 save(R/'SUMMARY.json',result)
 lines=['# L32-0045 full-model long decode modules','','Units µs; parentheses complete Host-wall shares. M64+42/cache128, five short and ten formal paired AB/BA cycles,repeat10. Frozen weights/scales,SP2,FP32 residual,no rotations.','','| 模块 | CONTROL Prefill | OPT Prefill | CONTROL Decode | OPT Decode |','|---|---:|---:|---:|---:|']
 cols=[result['formal']['arms'][a][m]['modules'] for m,a in [('prefill','CONTROL'),('prefill','OPT'),('decode','CONTROL'),('decode','OPT')]]
 for row in zip(*cols):lines.append('| '+row[0]['module']+' | '+' | '.join(f"{v['us']:.1f} ({v['share']:.2f}%)" for v in row)+' |')
 lines+=['','| Shape | Arm | Prefill E2E token/s | Decode E2E token/s |','|---|---|---:|---:|']
 for phase_name in ['formal']:
  z=result[phase_name]
  for a in ['CONTROL','OPT']:lines.append(f"| M64+{z['decode_tokens']} | {a} | {z['arms'][a]['prefill']['tps']:.2f} | {z['arms'][a]['decode']['tps']:.2f} |")
 lines+=['','63decode is checked for exact arithmetic in untimed boundary-audit runs; no separate formal63 speed claim. Cold loading/session setup and external tokenization excluded. Numerical implementation verification is separate from model quality.']
 (R/'MODULES.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({n:{a:{m:z[m]['tps'] for m in ['prefill','decode']} for a,z in result[n]['arms'].items()} for n in ['formal']},indent=2))
if __name__=='__main__':main()
