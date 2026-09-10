#!/usr/bin/env python3
"""Reconcile additive model ledgers and report complete paired Llama timings."""
import json
from pathlib import Path
import numpy as np
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0004')
MODULES=[
('I/O、metadata',['input_stage_ticks','metadata_stage_ticks','output_stage_ticks']),
('Input RMSNorm',['input_norm_ticks']),
('QKV＋RoPE',['qkv_projection_ticks','qk_norm_rope_ticks']),
('QK–Softmax–AV',['attention_ticks']),('O projection',['o_projection_ticks']),
('Post-attention residual＋RMSNorm',['post_attention_residual_ticks','post_attention_norm_ticks']),
('Gate/Up＋SwiGLU',['gate_up_ticks','activation_ticks']),('Down',['down_ticks']),
('Final residual',['final_residual_ticks']),('KV carrier conversion',['scan_cache_pack_ticks']),
('KV append DMA',['scan_cache_append_ticks']),('Block orchestration',['block_orchestration_ticks']),
('Layer bookkeeping',['layer_bookkeeping_ticks']),('Stage-boundary bookkeeping',['stage_boundary_ticks']),
('DSP unattributed',['ledger_unattributed_ticks']),('Runtime setup/teardown',['runtime_setup_ticks','runtime_teardown_ticks']),
('Embedding',['generation_embedding_ticks']),('Final model RMSNorm',['generation_final_norm_ticks']),
('LM head＋greedy（不含 final norm）',['generation_lm_head_ticks'])]
def main():
    root=OUT/'profile-a01';summary=json.loads((root/'summary.json').read_text());data={};checked=0
    for recipe in ['w4a16','w4a8']:
        data[recipe]={}
        for variant in ['baseline','candidate']:
            runs=[]
            for i in range(10):
                d=json.loads((root/f'formal-{recipe}-{i:02d}-{variant}'/'result.json').read_text());assert d['pass'];runs.append(d)
                for q in d['profiles']:
                    ticks=[sum(q[k] for k in fields) for _,fields in MODULES]
                    ticks[-1]-=q['generation_final_norm_ticks']
                    assert sum(ticks)==q['invocation_ticks'],(recipe,variant,i,sum(ticks),q['invocation_ticks'])
                    assert q['host_wall_ns']/1000-q['invocation_ticks']/19.2>=0
                    checked+=1
            modes={}
            for mode in ['prefill','decode']:
                groups=[[d['profiles'][0]] if mode=='prefill' else d['profiles'][1:] for d in runs]
                rows=[]
                host=float(np.mean([np.mean([q['host_wall_ns']/1000 for q in gg]) for gg in groups]))
                for name,fields in MODULES:
                    vals=[]
                    for gg in groups:
                        v=[sum(q[k] for k in fields)-(q['generation_final_norm_ticks'] if fields==['generation_lm_head_ticks'] else 0) for q in gg]
                        vals.append(float(np.mean(v))/19.2)
                    value=float(np.mean(vals));rows.append({'module':name,'us':value,'host_share_percent':value/host*100})
                boundary=host-sum(x['us'] for x in rows);rows.append({'module':'Host–DSP 边界','us':boundary,'host_share_percent':boundary/host*100});rows.append({'module':'完整 Host wall','us':host,'host_share_percent':100.0});modes[mode]=rows
            data[recipe][variant]=modes
    report={'experiment':'L32-0004','formal_pairs_per_recipe':10,'verified_token_boundaries':checked,'qtimer_ticks_per_us':19.2,'module_aggregation':'mean per-token-boundary within run, then mean of ten runs; independent overlapping subcounters excluded','speed':summary,'modules':data,'quality':'Unchanged weights/qparams; W4A16 prior PPL gate failed, A8 unusable with no quality gate; no quality promotion'}
    out=OUT/'profiling_summary.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# L32-0004 native pipeline profiling','', 'SM8750 / HTP V79; ten fixed AB/BA pairs per recipe. Same sealed original-derived weights/qparams. No baseline quality promotion.','']
    cols=[('w4a16','baseline'),('w4a16','candidate'),('w4a8','baseline'),('w4a8','candidate')]
    for mode in ['prefill','decode']:
        lines += ['## '+mode,'','Microseconds; parentheses are share of complete Host wall. Decode rows are per generated token.','', '| Module | W4A16 before | W4A16 after | W4A8 before | W4A8 after |','|---|---:|---:|---:|---:|']
        for i in range(len(data['w4a16']['baseline'][mode])):
            row=[data['w4a16']['baseline'][mode][i]['module']]
            for recipe,v in cols:
                d=data[recipe][v][mode][i];row.append(f"{d['us']:.1f} ({d['host_share_percent']:.2f}%)")
            lines.append('| '+' | '.join(row)+' |')
        lines.append('')
    lines+=['## E2E','','| Recipe / mode | Before token/s | After token/s | After/before Host wall (95% CI) |','|---|---:|---:|---|']
    for recipe,modes in summary.items():
        for mode,s in modes.items():
            assert s['slowdown_gate_pass'];lines.append(f"| {recipe} / {mode} | {s['baseline_tokens_per_second']:.2f} | {s['candidate_tokens_per_second']:.2f} | {s['ratio']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] |")
    lines+=['','M64 prefill, W4A16 seven continuous decode tokens to EOS; W4A8 fifteen. Host timing includes embedding,16 layers,final norm,LM head,greedy and FastRPC; excludes loading/session preparation and external tokenizer. Different decode lengths are not a matched cross-recipe comparison. All480 timed token-boundary ledgers reconcile exactly.','']
    (OUT/'PROFILE.md').write_text('\n'.join(lines));print(json.dumps({'speed':summary,'verified_token_boundaries':checked}),flush=True)
if __name__=='__main__':main()
