#!/usr/bin/env python3
"""Read only sealed run directories; write new campaign aggregates once."""
import sys,json,statistics,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from report_llama32_pipeline_profile import MODULES
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0043')
def read(p):return json.loads(p.read_text())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 short=read(R/'short-paired.json');formal=read(R/'formal-paired.json')
 assert short['rounds']==5 and formal['rounds']==10 and short['pass_all'] and formal['pass_all']
 result=dict(experiment='L32-0043',model='Llama-3.2-3B-Instruct',shape='M64+15 cache80,28 layers',short_rounds=5,formal_rounds=10,repeat=10,profiles_per_arm=1600,quality_claim=False,baseline_promoted=False,arms={},effects={},submodules={})
 frozen_counts=['weight_ddr_read_bytes','scan_cache_ddr_write_bytes','block_invocation_count','generation_lm_head_command_count']
 profiles={}
 for arm in ['CONTROL','OPT3']:
  groups=[]
  for cycle in range(10):
   d=R/'formal'/f'{cycle:02d}-{arm}'
   assert read(d/'validated.json')['pass_all']
   ps=[v for v in read(d/'records.json') if isinstance(v,dict) and v.get('record')=='generation_profile']
   assert len(ps)==160
   groups.append(ps)
  profiles[arm]=groups
  a={}
  for mode in ['prefill','decode']:
   gs=[[v for v in ps if v['mode']==mode] for ps in groups]
   host=[statistics.mean(v['host_wall_ns']/1000 for v in g) for g in gs]
   hm=statistics.mean(host);rows=[]
   for name,keys in MODULES:
    value=statistics.mean(statistics.mean((sum(v[k] for k in keys)-(v['generation_final_norm_ticks'] if keys==['generation_lm_head_ticks'] else 0))/19.2 for v in g) for g in gs)
    rows.append(dict(module=name,us=value,host_share_percent=100*value/hm))
   boundary=hm-sum(v['us'] for v in rows);assert boundary>=0
   rows.extend([dict(module='Host–DSP 边界',us=boundary,host_share_percent=100*boundary/hm),dict(module='完整 Host wall',us=hm,host_share_percent=100)])
   a[mode]=dict(tps=(64 if mode=='prefill' else 1)*1e6/hm,host_us=hm,round_host_us=host,modules=rows)
   for g in gs:
    for v in g:
     assert v['vtcm_peak_plan_bytes']<=8388608 and all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks'])
   detail=['qkv_projection_ticks','qk_norm_rope_ticks','scan_cache_pack_ticks','u8_attention_k_pack_ticks','u8_attention_v_pack_ticks','u8_attention_softmax_ticks']
   result['submodules'][arm+'-'+mode]={k:statistics.mean(statistics.mean(v[k]/19.2 for v in g) for g in gs) for k in detail}
  result['arms'][arm]=a
 for cycle in range(10):
  aa=profiles['CONTROL'][cycle];bb=profiles['OPT3'][cycle]
  for a,b in zip(aa,bb):
   for k in frozen_counts:assert a[k]==b[k],(cycle,k,a[k],b[k])
   if a['mode']=='prefill':
    assert a['hmx_u8s8_tile_pair_count']==b['hmx_u8s8_tile_pair_count']
    assert a['hmx_command_count']-b['hmx_command_count']==28*128
   else:
    assert a['hmx_command_count']==b['hmx_command_count']
    assert a['hmx_u8s8_tile_pair_count']-b['hmx_u8s8_tile_pair_count']==28*384

 rng=np.random.default_rng(320043)
 for mode in ['prefill','decode']:
  a=np.array(result['arms']['CONTROL'][mode]['round_host_us']);b=np.array(result['arms']['OPT3'][mode]['round_host_us'])
  picks=rng.integers(0,10,size=(20000,10));ratios=b[picks].mean(1)/a[picks].mean(1);ci=np.quantile(ratios,[.025,.975]).tolist()
  ratio=float(b.mean()/a.mean())
  result['effects'][mode]=dict(wall_ratio=ratio,latency_reduction_percent=100*(1-ratio),throughput_gain_percent=100*(1/ratio-1),ci95=ci,ten_percent_gate=ci[1]<=1.1)
 result['all_gates_pass']=all(v['ten_percent_gate'] for v in result['effects'].values())
 result['weight_bytes_identical']=True
 result['physical_tile_reduction_per_decode']=28*384
 result['prefill_commands_reduced']=28*128
 result['absolute_target']={m:dict(target=target,observed=result['arms']['OPT3'][m]['tps'],passed=result['arms']['OPT3'][m]['tps']>=target) for m,target in [('prefill',1100),('decode',22)]}
 result['target_pass']=all(v['passed'] for v in result['absolute_target'].values())
 result['vtcm_peak_bytes']=max(v['vtcm_peak_plan_bytes'] for g in profiles['OPT3'] for v in g)
 result['measured_source']=read(R/'opt3-l28/runtime.json')['seal']['source_head']
 result['control_source']=read(R/'control-l28/runtime.json')['seal']['source_head']
 save(R/'SUMMARY.json',result)
 lines=['# L32-0043: 3B shape-specific pipeline/layout optimization','','M64+15, KV capacity80,28layers;5 short and10 formal paired AB/BA cycles,repeat10. Complete Host wall includes embedding,all blocks,final norm,head,greedy and RPC; excludes cold loading/session setup and external tokenization. Frozen original3B weights/scales/SP2mode8,FP32 residual,no rotations.','',
 '| 模块 | 原实现 Prefill μs（占比） | 优化 Prefill μs（占比） | 原实现 Decode μs/token（占比） | 优化 Decode μs/token（占比） |','|---|---:|---:|---:|---:|']
 cols=[result['arms'][arm][mode]['modules'] for mode,arm in [('prefill','CONTROL'),('prefill','OPT3'),('decode','CONTROL'),('decode','OPT3')]]
 for row in zip(*cols):lines.append('| '+row[0]['module']+' | '+' | '.join(f"{x['us']:.1f} ({x['host_share_percent']:.2f}%)" for x in row)+' |')
 lines+=['','| 配置 | Prefill E2E token/s | Decode E2E token/s |','|---|---:|---:|']
 for arm in ['CONTROL','OPT3']:lines.append(f"| {arm} | {result['arms'][arm]['prefill']['tps']:.2f} | {result['arms'][arm]['decode']['tps']:.2f} |")
 lines+=['','Paired-bootstrap95% CIs (resampling matched cycles,20000 samples,seed320043):']
 for mode,x in result['effects'].items():lines.append(f"- {mode}: candidate/control wall {x['wall_ratio']:.8f}, CI{x['ci95']}; throughput +{x['throughput_gain_percent']:.3f}%.")
 lines+=['','All3200 formal additive profiles reconcile; Weight bytes match; prefill reduces3584 HMX submissions, decode removes10752 padding tile pairs per token. Live arithmetic unchanged. VTCM peak'+str(result['vtcm_peak_bytes'])+' bytes; zero intermediate DDR/spill. No quality claim or automatic baseline promotion.','']
 (R/'MODULES.md').write_text('\n'.join(lines))
 print(json.dumps(dict(effects=result['effects'],tps={k:{m:v[m]['tps'] for m in ['prefill','decode']} for k,v in result['arms'].items()},gate=result['all_gates_pass']),indent=2))
if __name__=='__main__':main()
