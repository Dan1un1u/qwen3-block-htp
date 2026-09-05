"""Close EXP-0217 from raw generation evidence without rerunning devices."""
import ast
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0217')
FORMAL = ROOT/'formal'
TICKS = 19.2


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def normalized(profiles):
    values = {k: sum(p[k] for p in profiles)/len(profiles) for k, v in profiles[0].items()
              if isinstance(v, (float, int)) and all(isinstance(p.get(k), (float, int)) for p in profiles)}
    values['host_us'] = values['host_wall_ns']/1000
    values['host_boundary_us'] = values['host_us']-values['invocation_ticks']/TICKS
    values['generation_lm_head_exclusive_ticks'] = values['generation_lm_head_ticks']-values['generation_final_norm_ticks']
    return values


def main():
    ledger = [('Token embedding', 'generation_embedding_ticks'), ('Input staging', 'input_stage_ticks'), ('Metadata', 'metadata_stage_ticks'), ('Input RMSNorm', 'input_norm_ticks'), ('QKV projection', 'qkv_projection_ticks'), ('Q/K Norm-RoPE (separate tail)', 'qk_norm_rope_ticks'), ('QK-Softmax-AV', 'attention_ticks'), ('O projection', 'o_projection_ticks'), ('Post-attention residual', 'post_attention_residual_ticks'), ('Post-attention RMSNorm (fused)', 'post_attention_norm_ticks'), ('Gate/Up', 'gate_up_ticks'), ('SwiGLU', 'activation_ticks'), ('Down', 'down_ticks'), ('Final residual', 'final_residual_ticks'), ('Output staging', 'output_stage_ticks'), ('KV-cache carrier conversion', 'scan_cache_pack_ticks'), ('KV-cache append DMA', 'scan_cache_append_ticks'), ('Block orchestration', 'block_orchestration_ticks'), ('Layer bookkeeping', 'layer_bookkeeping_ticks'), ('Stage-boundary bookkeeping', 'stage_boundary_ticks'), ('Final model RMSNorm', 'generation_final_norm_ticks'), ('LM head + greedy selection (excluding final norm)', 'generation_lm_head_exclusive_ticks'), ('Runtime setup', 'runtime_setup_ticks'), ('Runtime teardown', 'runtime_teardown_ticks'), ('DSP unattributed', 'ledger_unattributed_ticks')]
    overview = [('I/O and metadata', ['input_stage_ticks', 'metadata_stage_ticks', 'output_stage_ticks']), ('Input RMSNorm', ['input_norm_ticks']), ('QKV + Q/K Norm-RoPE', ['qkv_projection_ticks', 'qk_norm_rope_ticks']), ('QK-Softmax-AV', ['attention_ticks']), ('O projection', ['o_projection_ticks']), ('Post-attention residual + RMSNorm', ['post_attention_residual_ticks', 'post_attention_norm_ticks']), ('Gate/Up + SwiGLU', ['gate_up_ticks', 'activation_ticks']), ('Down projection', ['down_ticks']), ('Final residual', ['final_residual_ticks']), ('KV-cache carrier conversion', ['scan_cache_pack_ticks']), ('KV-cache append DMA', ['scan_cache_append_ticks']), ('Block internal orchestration', ['block_orchestration_ticks']), ('Layer bookkeeping', ['layer_bookkeeping_ticks']), ('Stage-boundary bookkeeping', ['stage_boundary_ticks']), ('DSP unattributed residual', ['ledger_unattributed_ticks']), ('DSP runtime setup/teardown', ['runtime_setup_ticks', 'runtime_teardown_ticks']), ('Token embedding', ['generation_embedding_ticks']), ('Final model RMSNorm', ['generation_final_norm_ticks']), ('LM head + greedy selection (excluding final norm)', ['generation_lm_head_exclusive_ticks']), ('True Host-DSP boundary', ['host_boundary_us']), ('Complete Host wall', ['host_us'])]
    expected = json.loads((ROOT/'semantic_reference.json').read_text())['f16f16']['token_ids']
    paths = sorted((FORMAL/'raw').glob('round_*_f16f16.jsonl'))
    assert len(paths) == 10
    runs = []
    layer_fields = ['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks',
        'qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks',
        'post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks',
        'cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks',
        'layer_unattributed_ticks']
    for path in paths:
        records = read(path)
        profiles = [p for p in records if p.get('record') == 'generation_profile']
        steps = [p for p in records if 'generation_step' in p and p.get('record') != 'generation_profile']
        final = [p for p in records if p.get('generation_sequence_complete')]
        assert len(profiles) == len(steps) == 16 and len(final) == 1
        assert final[0]['token_ids'] == expected and final[0]['all_steps_pass']
        for step, (p, s) in enumerate(zip(profiles, steps)):
            assert s['pass'] and s['token_match'] and s['selected_token_id'] == expected[step]
            assert p['variant'] == 'F16F16' and p['generation_step'] == step
            assert p['backend'] == 'standalone_fastrpc_dsp' and p['qnn'] == 'none'
            assert p['vtcm_requested_bytes'] == p['vtcm_acquired_bytes'] == 8388608
            assert p['repeat_count'] == 1 and p['block_invocation_count'] == 28
            assert p['boundary_ddr_write_bytes'] == p['intermediate_ddr_read_bytes'] == p['intermediate_ddr_write_bytes'] == p['intermediate_spill_fill_count'] == 0
            assert p['generation_lm_head_expand_ticks'] == p['generation_lm_head_scale_dma_ticks'] == 0
            assert p['generation_lm_head_ddr_read_bytes'] == 622329856
            n = normalized([p])
            assert sum(n[k] for _, k in ledger) == p['invocation_ticks'], (path, step, 'invocation ledger')
            for i in range(28):
                lp = p[f'slice_layer_{i}']
                assert lp['status'] == 3 and lp['layer_index'] == i
                assert sum(lp[k] for k in layer_fields) == lp['layer_ticks']
                assert lp['layer_unattributed_ticks'] == lp['hidden_ddr_read_bytes'] == lp['hidden_ddr_write_bytes'] == 0
                assert lp['cache_valid_before'] == (0 if step == 0 else 63+step)
                assert lp['cache_valid_after'] == 64+step
        runs.append(profiles)
    data = {mode: [normalized([p for p in r if p['mode'] == mode]) for r in runs] for mode in ('prefill', 'decode')}
    wroot = ROOT.parent/'exp0166/20260903T_exp0166_8e0dcf8_formal/raw'
    uroot = ROOT.parent/'exp0216/20260905_formal_25b2fdf/raw'
    old = []
    historical_paths = []
    for root, pattern in [(wroot, 'round_*_candidate.jsonl'), (uroot, 'pair_*_direct_qkvo.log')]:
        series = []
        for path in sorted(root.glob(pattern)):
            p = [p for p in read(path) if p.get('record') == 'generation_profile' and p['mode'] == 'prefill']
            assert len(p) == 1
            series.append(normalized(p)); historical_paths.append(path)
        assert len(series) == 10
        old.append(series)
    def value(series, fields, count=10):
        return median(sum(p[f] if f.endswith('_us') else p[f]/TICKS for f in fields) for p in series[:count])
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
    chinese = ['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection',
               'Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual',
               'KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping',
               'Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown',
               'Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
    rows = []
    for label, (_, fields) in zip(chinese, overview):
        series = [data['prefill'], *old]
        vals = [value(s, fields) for s in series]
        cells = [f'{v:.1f} ({100*v/value(s,["host_us"]):.2f}%)' for v, s in zip(vals, series)]
        rows.append([label, *cells, f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else 'N/A'])
    compact = table(['模块','F16F16 EXP-0217','W4F16 EXP-0166','W4U8 EXP-0216','W4U8 相对 W4F16 增速'],rows)
    (FORMAL/'module_table.md').write_text(compact)
    summary = dict(experiment='EXP-0217', sessions=10, selected_token_checks=160,
                   independent_selected_token_mismatches=0, invocation_ledgers_checked=160,
                   layer_ledgers_checked=4480, full_logit_quality_assessment='not_run',
                   text=json.loads((ROOT/'semantic_reference.json').read_text())['f16f16']['text'],
                   data=data, provenance=json.loads((FORMAL/'provenance.json').read_text()))
    for mode in data:
        wall = value(data[mode], ['host_us'])*(1 if mode == 'prefill' else 15)
        tokens = 64 if mode == 'prefill' else 15
        summary[mode] = dict(tokens=tokens, complete_host_us=wall, tokens_per_second=tokens*1e6/wall)
    out = ['# EXP-0217 complete profiling closure',
           'F16F16 owns tokenizer/prepared token IDs, FP16 embedding, 28 transformer layers, final RMSNorm, FP16 stored LM head and greedy feedback. Ten independent M64+15 sessions. No equivalent earlier full-generation F16F16 control exists; control/candidate deltas are N/A, speed is report-only. R1 is the first formal session; R10 is the median of all ten sessions. Each RPC repeat_count=1.',
           json.dumps(summary['provenance'],indent=2),
           'Token IDs match the independent Transformers FP16 reference in all 160 comparisons. This does not prove full-logit equality or general quality. Generation profile hidden/cache error fields with zero compared elements are placeholders, not independent tensor comparisons. All-layer cache lengths and timed DDR/residency are checked. The unchanged F16 transformer retains its previously approved numerical contract. Dedicated transformer replay regression is archived separately.',
           'All 160 invocation and 4480 per-layer exclusive ledgers close exactly. Final norm is subtracted from the inclusive LM-head field for additive reporting. Engine timings may overlap. Individual row medians need not sum to Host median.',
           'Three-recipe table: F16F16 EXP-0217 current full-generation measurement; W4F16 selected EXP-0166 and W4U8 EXP-0216 are non-paired historical references. Historical W4U8 decode uses 192 steps and is not an equivalent 15-step ranking.', compact]
    for mode in data:
        out.append('## '+mode+' complete exported diagnostics')
        out.append('Every numeric exported field is retained. *_ticks are qtimer ticks (19.2 ticks/us), *_us are microseconds; other fields preserve raw units. Nested layer ledgers are verified in every raw record. No same-scope F16 control exists.')
        out.append(table(['Field','R1','R10 median','Control/delta'],[[k,f'{data[mode][0][k]:.6f}',f'{median(p[k] for p in data[mode]):.6f}','N/A: new integration scope'] for k in sorted(data[mode][0])]))
    out.append('## Direct E2E throughput')
    out.append(table(['Mode','Tokens','Complete Host us','tok/s'], [[m,summary[m]['tokens'],f'{summary[m]["complete_host_us"]:.3f}',f'{summary[m]["tokens_per_second"]:.6f}'] for m in ('prefill','decode')]))
    (FORMAL/'full_profiling_report.md').write_text('\n\n'.join(out)+'\n')
    (FORMAL/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (FORMAL/'historical_evidence_sha256.json').write_text(json.dumps({str(p):sha(p) for p in historical_paths},indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='data'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
