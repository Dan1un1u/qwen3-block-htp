#!/usr/bin/env python3
"""Reconcile the fixed matched L32-0006 three-arm Host/DSP ledgers."""
import json
from pathlib import Path
import numpy as np
from report_llama32_pipeline_profile import MODULES
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0006')
ARMS=[('w4a16','baseline'),('w4a8','baseline'),('w4a8','candidate')]
def main():
    root=OUT/'profile-a01';speed=json.loads((root/'summary.json').read_text());data={};checked=0
    for recipe,variant in ARMS:
        name=recipe+'_'+variant;runs=[]
        for i in range(10):
            d=json.loads((root/f'formal-{recipe}-{i:02d}-{variant}/result.json').read_text());assert d['pass'];runs.append(d)
            for q in d['profiles']:
                ticks=[sum(q[k] for k in fields) for _,fields in MODULES];ticks[-1]-=q['generation_final_norm_ticks']
                assert sum(ticks)==q['invocation_ticks'],(name,i,sum(ticks),q['invocation_ticks'])
                assert q['host_wall_ns']/1000>=q['invocation_ticks']/19.2
                checked+=1
        data[name]={}
        for mode in ['prefill','decode']:
            groups=[[d['profiles'][0]] if mode=='prefill' else d['profiles'][1:] for d in runs]
            host=float(np.mean([np.mean([q['host_wall_ns']/1000 for q in gg]) for gg in groups]));rows=[]
            for module,fields in MODULES:
                value=float(np.mean([np.mean([sum(q[k] for k in fields)-(q['generation_final_norm_ticks'] if fields==['generation_lm_head_ticks'] else 0) for q in gg])/19.2 for gg in groups]))
                rows.append({'module':module,'us':value,'host_share_percent':100*value/host})
            boundary=host-sum(x['us'] for x in rows);rows.extend([{'module':'Host–DSP 边界','us':boundary,'host_share_percent':100*boundary/host},{'module':'完整 Host wall','us':host,'host_share_percent':100.0}]);data[name][mode]=rows
    long_speed=json.loads((root/'summary-long.json').read_text())
    for i in range(10):
        for v in ['baseline','candidate']:
            d=json.loads((root/f'formal-long-{i:02d}-{v}/result.json').read_text());assert d['pass']
            for q in d['profiles']:
                ticks=[sum(q[k] for k in fields) for _,fields in MODULES];ticks[-1]-=q['generation_final_norm_ticks']
                assert sum(ticks)==q['invocation_ticks'] and q['host_wall_ns']/1000>=q['invocation_ticks']/19.2
                checked+=1
    report={'experiment' :'L32-0006','formal_cycles':10,'verified_token_boundaries':checked,'qtimer_ticks_per_us':19.2,'speed':speed,'speed_m64_plus15':long_speed,'modules':data,'quality':'Unchanged sealed weights/qparams. W4A16 prior PPL gate failed; A8 text unusable without quality gate. No PPL rerun or quality promotion.'}
    assert checked==560 and not (OUT/'profiling_summary.json').exists()
    (OUT/'profiling_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# L32-0006: native A8 relative speed','', 'SM8750 / HTP V79. Fixed ten ABC/BCA/CAB cycles; identical original M64 prompt and seven continuous decode tokens for all three arms. Quantized recipe outputs differ; input dimensions and KV lengths match.','']
    for mode in ['prefill','decode']:
        lines+=['## '+mode,'','Microseconds, parentheses are complete Host wall shares. Decode is per token.','','| Module | W4A16 OPT2 | A8 before | A8 after |','|---|---:|---:|---:|']
        for i,row in enumerate(data['w4a16_baseline'][mode]):
            values=[data[r+'_'+v][mode][i] for r,v in ARMS]
            lines.append('| '+row['module']+' | '+' | '.join(f"{v['us']:.1f} ({v['host_share_percent']:.2f}%)" for v in values)+' |')
        lines.append('')
    lines+=['## E2E','','| Mode | W4A16 token/s | A8 before token/s | A8 after token/s |','|---|---:|---:|---:|']
    for mode,x in speed.items():
        lines.append('| '+mode+' | '+' | '.join(f"{x['arms'][r+'_'+v]['tokens_per_second']:.2f}" for r,v in ARMS)+' |')
    lines+=['','Complete warm Host wall includes embedding,16 layers,final norm,LM head,greedy and FastRPC. Model loading/session preparation and external tokenizer are excluded. No layer extrapolation. All560 additive ledgers reconcile.','']
    for mode,x in speed.items():
        for ref,s in x['comparisons'].items():
            lines.append(f"{mode}, new A8 / {ref} Host ratio: {s['candidate_host_ratio']:.4f},95% paired bootstrap CI [{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}].")
    lines+=['','## M64 +15 decode, ten independent AB/BA pairs','','| Mode | A8 before token/s | A8 after token/s |','|---|---:|---:|']
    for mode,x in long_speed.items():
        lines.append(f"| {mode} | {x['baseline_tokens_per_second']:.2f} | {x['candidate_tokens_per_second']:.2f} |")
        assert x['slowdown_gate_pass']
    (OUT/'PROFILE.md').write_text('\n'.join(lines)+'\n');print(json.dumps(speed),flush=True)
if __name__=='__main__':main()
