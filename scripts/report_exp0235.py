#!/usr/bin/env python3
"""Close fixed precision-localization evidence without selecting a deployable recipe."""
import hashlib,json,math,subprocess,tarfile
from pathlib import Path
from data_exp0235 import RESULT,OUTPUT,SOURCE,MODEL,BASE,FAMILIES,specs,write,sha,verified,preflight,frozen

def stream_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    preflight();frozen();assert not (RESULT/'closure.json').exists()
    summary=json.loads((RESULT/'summary.json').read_text());manifest=json.loads((RESULT/'restoration_manifest.json').read_text());audit=json.loads((RESULT/'restoration_audits.json').read_text())
    assert summary['independent_raw_token_PPL_reductions']==594 and audit['pass_all'] and audit['final_C64_sentinel_exact'] and audit['immutable_snapshots_exact']
    assert len(audit['audits'])==65 and list(manifest['variants'])==list(specs())
    for n,h in summary['inputs'].items():assert sha(RESULT/n)==h,n
    for a in audit['audits']:
        chosen=specs()[a['variant']];assert a['restored_names']==chosen
        expected={n:(manifest['original_state_hashes'][n] if n in chosen else h) for n,h in manifest['C64_state_hashes'].items()}
        assert a['state_hashes']==expected and a['all_parameter_and_buffer_hashes_match']
        assert a['restored_weight_count']==sum(manifest['weight_counts'][n] for n in chosen)
    for v,names in specs().items():assert manifest['variants'][v]['restored_names']==names
    assert sha(BASE.parent.parent/'models/qwen3-block-htp/exp0230/C64/manifest.json')==manifest['C64_manifest_sha256']
    assert sha(BASE.parent.parent/'models/qwen3-block-htp/exp0217/f16f16_greedy16/manifest.json')==manifest['floating_manifest_sha256']
    for n,h in json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text()).items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h),n
    for n,e in json.loads((RESULT/'reused_controls.json').read_text()).items():assert sha(RESULT/n)==sha(Path(e['source']))==e['sha256']
    commands=[]
    for p in sorted((RESULT/'commands').glob('*.json')):
        c=json.loads(p.read_text());assert c['returncode']==0 and sha(p.with_suffix('.log'))==c['log_sha256']
        archive=OUTPUT/'artifacts'/c['source_head']/'source.tar';assert stream_sha(archive)==c['source_archive_sha256'];commands.append(c)
    assert len(commands)==len(list((RESULT/'commands').glob('*.log')))
    actual=next(c for c in commands if any(str(a).endswith('/evaluate_exp0235.py') for a in c['command']))
    assert manifest['source_head']==actual['source_head']
    with tarfile.open(OUTPUT/'artifacts'/actual['source_head']/'source.tar') as archive:
        for n in ['evaluate_exp0235.py','data_exp0235.py','evaluate_exp0230.py']:
            assert hashlib.sha256(archive.extractfile('scripts/'+n).read()).hexdigest()==sha(SOURCE/'scripts'/n),n
    write('execution_provenance.json',dict(actual_evaluation_source=actual['source_head'],commands=commands,entry_and_loading_code_identical_to_archive=True))
    artifacts={str(p.relative_to(OUTPUT)):dict(bytes=p.stat().st_size,sha256=stream_sha(p)) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write('artifacts_sha256.json',dict(root=str(OUTPUT),files=artifacts,model_storage='Original verified checkpoint and frozen C64 referenced, immutable CPU snapshots audited in restoration_manifest; no hybrid package export'))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    conf=summary['confirmation']['variants'];dev=summary['development']['variants']
    text=['# EXP0235 C64 precision sensitivity','',
        'Completed diagnostic:61fixed development variants on128documents/2048targets, then allfive predeclared families on the full shared PC0521024documents/16384targets. No hybrid selection, quantization, deployment or baseline promotion.',
        '', '## Fixed full-panel family confirmation', '',
        '|Variant|Restored weights (millions)|PPL|PPL vs F16|PPL ratio vs C64 [95% CI]|C64 excess NLL removed|Diagnostic thresholds|',
        '|---|---:|---:|---:|---|---:|---|']
    for v in FAMILIES:
        x=conf[v];g=x['statistics']['overall'];q=g['vs_C64'];ci=q['ratio_ci95'];fraction=g['fraction_C64_excess_NLL_removed']
        text.append(f"|{v}|{x['restored_weight_count']/1e6:.6f}|{g['ppl']:.6f}|{100*(g['vs_F16']['ppl_ratio']-1):+.6f}%|{q['ppl_ratio']:.6f} [{ci[0]:.6f},{ci[1]:.6f}]|{100*fraction:.3f}%|{x['diagnostic_reference_threshold_status']}|")
    text+=['','Restored counts cover the196transformer projection matrices plus LMhead, excluding the fixed FP16 embedding and norms. F is the full floating control; C64 restores zero matrices. ATT_ALL restores q/k/v/o across28layers; MLP_ALL restores gate/up/down across28layers; HEAD restores only LMhead. All other parameter and buffer bytes remain C64-identical.',
        '', '|Stratum|F|C64|ATT_ALL|MLP_ALL|HEAD|','|---|---:|---:|---:|---:|---:|']
    for name in conf['F']['statistics']:text.append('|'+name+'|'+'|'.join(f"{conf[v]['statistics'][name]['ppl']:.6f}" for v in FAMILIES)+'|')
    text+=['','All per-stratum paired NLL differences, PPL ratios and95%intervals are retained in summary.json. Reference thresholds remain overall5% and language/domain/cell10%, but passing a diagnostic restoration does not accept a deployable mixed-precision recipe. Both panel halves were fixed and unconditional for every family, regardless of development ranking. F/C64 full-panel scores are verified byte-for-byte reuse of EXP233, not new measurements.',
        '', '## Development family controls','', '|Variant|PPL|C64 excess NLL removed|','|---|---:|---:|']
    for v in FAMILIES:
        g=dev[v]['statistics']['overall'];text.append(f"|{v}|{g['ppl']:.6f}|{100*g['fraction_C64_excess_NLL_removed']:.3f}%|")
    text+=['','## Fixed single-layer localization (development only)','','|Rank|Restore set (zero-based layer)|Weights (millions)|PPL|Delta NLL vs C64 [95% CI]|C64 excess NLL removed|','|---:|---|---:|---:|---|---:|']
    for rank,v in enumerate(summary['development_layer_ranking'],1):
        x=dev[v];g=x['statistics']['overall'];q=g['vs_C64'];ci=q['delta_nll_ci95']
        text.append(f"|{rank}|{v}|{x['restored_weight_count']/1e6:.6f}|{g['ppl']:.6f}|{q['delta_nll']:+.7f} [{ci[0]:+.7f},{ci[1]:+.7f}]|{100*g['fraction_C64_excess_NLL_removed']:.3f}%|")
    text+=['','Each single-layer result is conditional on all other C64 weights. Effects are nonadditive and can compensate or amplify one another; never sum recovered fractions or extrapolate a top-k combination. The56 exploratory intervals are pointwise, not simultaneous multiple-comparison significance. No ranked layer combination was constructed or scored. The128development documents are exposed diagnostic data.',
        '', '## Correctness and provenance','',
        'Fresh original checkpoint files and tokenizer match the verified EXP218 ledger. Full FP16 restoration and empty restoration exactly reproduce historical F/C64 per-token NLL and top1. Every new score passes exact repeat, future-mask invariance, all-logit finiteness and independent CE difference below5e-6. All65 model states match the complete expected parameter/buffer SHA256 map. Original/C64 snapshots remain immutable; the final128document C64 sentinel matches exactly. All594 reported stratum PPL values independently reduced from raw token NLL via math.fsum.',
        '',f"Actual evaluation source {actual['source_head']}; report source {head}. InputfreezeSHA256 {sha(RESULT/'dataset_freeze.json')}; restoration manifestSHA256 {sha(RESULT/'restoration_manifest.json')}. Retained stage commands and source archives establish execution provenance. No original, F16F16, W4U8, C64 package or DSP runtime changed.",
        '', 'The evaluation panel is the shared PC052 panel, independently reconstructed and frozen before EXP233 scoring, and independent of calibration/training. It is exposed paired reuse here, not a newly independent test perphase. No selection or training used these scores.',
        '', '## Completion boundary','',
        'All three PC052 diagnostics are complete. Results support discussion of the next precision-recovery step only. No automatic next optimization or baseline promotion. Device profiling/E2E for restoration hybrids is N/A; full_profiling_report.md retains complete unavailable sections and verified historical references.']
    with (RESULT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    table=verified('exp0230','module_table.md').read_text();prior=verified('exp0230','full_profiling_report.md')
    profile=f"""# EXP0235 complete profiling record: host-only sensitivity

Source branch codex/exp-0235-w4f16-c64-sensitivity; report source {head}; actual evaluation source {actual['source_head']}.
Quality unit: FP16 software M64+16 teacher forcing. Control C64, floating F, fixed ATT/MLP/head restorations. No mixed-precision device runtime is deployed.

|Required section|Repeat1 control/candidates/deltas|Repeat10 control/candidates/deltas|Reason|
|---|---|---|---|
|Complete Host wall / prefill / continuous decode|N/A|N/A|No hybrid device execution|
|DSP invocation / setup / teardown / ledger / unattributed|N/A|N/A|No DSP timestamps|
|All additive block stages|N/A|N/A|Host-only diagnostic|
|Projection DMA / HMX / unpack / waits / lifetime / workers|N/A|N/A|No hybrid runtime|
|Attention QK / Softmax / AV / packing / waits / tasks|N/A|N/A|No hybrid runtime|
|MLP GateUp / SwiGLU / Down / slots / publication|N/A|N/A|No hybrid runtime|
|DDR bytes / descriptors / spill / HMX / VTCM / FastRPC|N/A|N/A|No device use or physical performance claim|
|Device hashes / mismatches / maximum LSB|N/A|N/A|Software state and scoring oracles in REPORT.md|

Five short/ten formal device rounds are N/A. Overlapping engine counters would not be additive; none are measured here. Software exact control/sentinel/repeat/mask checks, all65 complete state maps and independent CE checks pass. N/A is not measured zero. No speed extrapolation from software evaluation duration.

## Historical three-recipe M64 overview

The following table is unchanged verified EXP230 evidence. Its C64 column is a quality candidate, not a promoted baseline; F16F16/W4U8 are nonpaired EXP218 references. Weight bytes differ, so no activation-only attribution. All repeat1/repeat10 historical counters and provenance: {prior}, SHA256 {sha(prior)}.

{table}

## End-to-end throughput

Restoration hybrid prefill/decode: N/A. Historical EXP230 C64 only:64prompt tokens /63262.995us =1011.649859tok/s;15continuous-decode tokens /1389448.9595us =10.795647tok/s. These do not measure a restoration hybrid.
"""
    with (RESULT/'full_profiling_report.md').open('x') as f:f.write(profile)
    closure=dict(experiment='EXP-0235',execution_state='completed',evidence_validity='valid',local_gate='pass',local_gate_scope='diagnostic_correctness_and_evidence_only_no_recipe_acceptance',adoption_status='not_applicable',source_head=head,actual_evaluation_source=actual['source_head'],
        final_ppl={v:conf[v]['statistics']['overall']['ppl'] for v in FAMILIES},restored_weights={v:conf[v]['restored_weight_count'] for v in FAMILIES},
        summary_sha256=sha(RESULT/'summary.json'),dataset_freeze_sha256=sha(RESULT/'dataset_freeze.json'),restoration_manifest_sha256=sha(RESULT/'restoration_manifest.json'),
        report_sha256=sha(RESULT/'REPORT.md'),full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),artifact_files=len(artifacts),
        PC052_three_phases_complete=True,baseline_promoted=False,other_recipes='frozen',device_speed='N/A',next_direction='discussion_only')
    write('closure.json',closure)
    write('evidence_sha256.json',{str(p.relative_to(RESULT)):stream_sha(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
    print('EXP235_CLOSED_DIAGNOSTIC_PASS_PC052_COMPLETE',flush=True)
if __name__=='__main__':main()
