#!/usr/bin/env python3
"""Aggregate L32-0041 complete Host wall and additive module ledgers."""
import json,statistics
from pathlib import Path
from report_llama32_pipeline_profile import MODULES
from prepare_llama32_3b import RES as R,save

def main():
    assert json.loads((R/'short.json').read_text())['pass_all']
    groups=[]
    for i in range(10):
        p=R/'formal'/f'{i:02d}';v=json.loads((p/'validated.json').read_text());assert v['pass_all'] and v['repeat']==10
        z=json.loads((p/'records.json').read_text());g=[v for v in z if isinstance(v,dict) and v.get('record')=='generation_profile'];assert len(g)==160;groups.append(g)
    out=dict(experiment='L32-0041',model='Llama-3.2-3B-Instruct',recipe='native per-channel W4A8 SP2 mode8 FP32 residual, no rotations',formal_rounds=10,repeat=10,prompt=64,decode_steps=15,kv_capacity=80,profiles=1600,numerical_exact=True,model_quality_accepted=False,modes={})
    for mode in ['prefill','decode']:
        samples=[[q for q in g if q['mode']==mode] for g in groups]
        host=[statistics.mean(q['host_wall_ns']/1000 for q in g) for g in samples];hm=statistics.mean(host);rows=[]
        for name,keys in MODULES:
            value=statistics.mean(statistics.mean((sum(q[k] for k in keys)-(q['generation_final_norm_ticks'] if keys==['generation_lm_head_ticks'] else 0))/19.2 for q in g) for g in samples)
            rows.append(dict(module=name,us=value,host_share_percent=100*value/hm))
        boundary=hm-sum(r['us'] for r in rows);assert boundary>=0
        rows.extend([dict(module='Host–DSP 边界',us=boundary,host_share_percent=boundary/hm*100),dict(module='完整 Host wall',us=hm,host_share_percent=100)])
        out['modes'][mode]=dict(tps=(64 if mode=='prefill' else 1)*1e6/hm,host_us=hm,round_host_us=host,round_min_us=min(host),round_max_us=max(host),modules=rows)
    save(R/'profiling_summary.json',out)
    lines=['# L32-0041 Llama 3.2 3B W4A8-SP2 native profiling','','M64+15, KV capacity80;5 short +10 formal rounds,10 trajectories/round. Timing includes embedding,28 blocks,final RMSNorm,LM head,greedy and FastRPC; excludes cold loading,session preparation and external tokenization. Fixed trajectory from independent greedy oracle. FP32 residual and no rotations. No PPL/text-quality acceptance.','','| 模块 | Prefill μs（Host wall 占比） | Decode μs/token（Host wall 占比） |','|---|---:|---:|']
    for a,b in zip(out['modes']['prefill']['modules'],out['modes']['decode']['modules']):lines.append(f"| {a['module']} | {a['us']:.1f} ({a['host_share_percent']:.2f}%) | {b['us']:.1f} ({b['host_share_percent']:.2f}%) |")
    lines+=['',f"E2E: prefill **{out['modes']['prefill']['tps']:.2f} token/s**, decode **{out['modes']['decode']['tps']:.2f} token/s**.",'','All1600 additive ledgers reconcile exactly; no intermediate DDR spill; one invocation per token boundary,28 blocks per invocation. These3B measurements are not a paired speed comparison with historical1B.','']
    (R/'MODULES.md').write_text('\n'.join(lines));print(json.dumps({k:v['tps'] for k,v in out['modes'].items()}))
if __name__=='__main__':main()