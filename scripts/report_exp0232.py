#!/usr/bin/env python3
"""Close the final PC051 phase after every stage wrapper has finished."""
import hashlib,json,math,platform,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
import torch,transformers
from data_exp0232 import RESULT,OUTPUT,SOURCE,MODEL,write,sha,verified,preflight
from awq_exp0232 import frozen,inputs,C64_HASH

VARIANTS=['F','C8','C64','E64']

def stream_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def read(n):return json.loads((RESULT/n).read_text())

def check_summary(final):
    assert not final['reserve_trigger'] and final['independent_raw_token_reduction_count']==36
    dataset=read('dataset.json');lookup={r['id']:r for r in dataset['samples']}
    by_variant={v:[] for v in VARIANTS}
    for n,h in final['inputs'].items():
        assert sha(RESULT/n)==h
        v=Path(n).stem.split('_')[-1];r=read(n)
        assert r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
        by_variant[v].extend(r['samples'])
    for v,rows in by_variant.items():
        assert len(rows)==final['documents'] and len({r['id'] for r in rows})==len(rows)
        for group,g in final['statistics'].items():
            values=[x for r in rows if group=='overall' or group==lookup[r['id']]['cell']
                    or group in lookup[r['id']]['cell'].split('_') for x in r['nll']]
            assert abs(math.exp(math.fsum(values)/len(values))-g['ppl'][v])<1e-10,(group,v)

def audit_export():
    inputs()
    search=read('search.json');invariance=read('invariance.json');oracle=read('awq_oracle.json')
    assert search['pass_all'] and invariance['pass_all'] and oracle['pass_all']
    assert not search['final_or_development_data_used'] and len(search['records'])==28
    assert invariance['search_sha256']==sha(RESULT/'search.json')
    for m in invariance['checks'].values():assert m['finite'] and m['nrmse']<=.003 and m['cosine']>=.99999
    for i,r in enumerate(search['records']):
        assert r==read(f'search/layer{i}.json') and r['layer']==i
        assert sha(OUTPUT/'scales'/f'layer{i}.npz')==r['scales_sha256']
        for k in ['unquantized_FP16_equivalence','C64_GPU_CPU_calibration_concordance']:
            m=r[k];assert m['finite'] and m['nrmse']<=.003 and m['cosine']>=.99999
        for t in r['trials'].values():
            assert len(t['mse'])==20 and np.isfinite(t['mse']).all()
            assert t['choice']==int(np.argmin(t['mse'])) and t['selected_mse']==min(t['mse'])
            assert t['ratio']==t['choice']/20 and t['calibration_positions']==8192
    root=OUTPUT/'E64';p=read('E64/package.json');m=json.loads((root/'manifest.json').read_text())
    assert sha(root/'manifest.json')==p['manifest_sha256'] and p['control_manifest_sha256']==C64_HASH
    assert m['calibration_tokens']==65536 and m['coordinates']=='fresh_AWQ_input_diagonals_with_legal_norm_and_Up_compensation'
    assert m['search_sha256']==sha(RESULT/'search.json') and m['invariance_sha256']==sha(RESULT/'invariance.json')
    assert len(m['changed_files'])==449 and len(set(m['changed_files']))==449
    for n,e in m['files'].items():assert sha(root/n)==e['sha256'],n
    for n,h in read('E64/frozen_files.json').items():assert sha(root/n)==h,n
    stats=read('E64/weight_stats.json');assert len(stats)==196
    for name,s in stats.items():
        assert s['packed_roundtrip'] and s['tokens']==65536 and s['groupsize']==-1 and s['act_order']
        assert s['selection_positions']==65536 and sha(Path(s['clip_stats_path']))==s['clip_stats_sha256']
        assert len(s['clip_choice_histogram'])==3 and sum(s['clip_choice_histogram'])==s['rows']
    checks=read('E64/calibration_forward_checks.json');assert len(checks)==112
    assert all(c['finite'] and c['nrmse']<=.003 and c['cosine']>=.99999 for c in checks)
    hidden=list((OUTPUT/'checkpoints/E64').glob('layer*_hidden.npy'));assert len(hidden)==28
    for path in hidden:
        a=np.load(path,mmap_mode='r');assert a.shape==(512,128,2048) and a.dtype==np.float16
        assert all(np.isfinite(a[i:i+8]).all() for i in range(0,512,8)),path
    commands=[]
    for log in (RESULT/'commands').glob('*.log'):
        meta=json.loads(log.with_suffix('.json').read_text());assert sha(log)==meta['log_sha256']
        archive=OUTPUT/'artifacts'/meta['source_head']/'source.tar'
        assert stream_sha(archive)==meta['source_archive_sha256'];commands.append(meta)
    assert len(commands)==len(list((RESULT/'commands').glob('*.json')))
    exports=[c for c in commands if c['command'][1].endswith('/export_exp0232.py') and c['returncode']==0]
    assert len(exports)==1;execution=exports[0]
    with tarfile.open(OUTPUT/'artifacts'/execution['source_head']/'source.tar') as archive:
        for n,h in p['quantizer_source'].items():
            assert hashlib.sha256(archive.extractfile('scripts/'+n).read()).hexdigest()==h,n
            assert sha(SOURCE/'scripts'/n)==h,n
    control=json.loads(verified('exp0230','C64/package.json').read_text())
    for n in ['gptq_exp0221.py','output_scale_exp0224.py','experiment_exp0224.py']:
        assert p['quantizer_source'][n]==control['quantizer_source'][n],n
    write('export_source_audit.json',dict(pass_all=True,actual_source_head=execution['source_head'],
        unchanged_core_quantizer=True,all_five_quantizer_files_match_archive=True,manifest_source_head=p['source_head']))
    return p

def device_evidence():
    primary=read('device_primary_summary.json');combined=primary['reserve_required']
    final=read('device_combined_summary.json' if combined else 'device_primary_summary.json')
    assert not final['reserve_required'] and final['variant_mapping']==dict(F='F',A0='C64',A='E64')
    for n,h in final['inputs'].items():assert sha(RESULT/n)==h
    for name in ['development_device_check.json','before_gate.json','after_gate.json']:
        assert read(name)['pass_all']
    lookup={r['id']:r for r in read('dataset.json')['samples']}
    for key,v in final['variant_mapping'].items():
        rows=[]
        for phase in (['primary','reserve'] if combined else ['primary']):
            for p in sorted((RESULT/f'device/{phase}').glob('*_'+v+'.validated.json')):
                run=json.loads(p.read_text())
                assert sha(p.with_suffix('').with_suffix('.jsonl'))==run['raw_sha256']
                rows.extend(run['samples'])
        assert len(rows)==final['documents'] and len({r['id'] for r in rows})==len(rows)
        for group,g in final['statistics'].items():
            losses=[s['nll'] for r in rows if group=='overall' or group==lookup[r['id']]['cell']
                    or group in lookup[r['id']]['cell'].split('_') for s in r['steps']]
            assert abs(math.exp(math.fsum(losses)/len(losses))-g['ppl'][key])<1e-10
    assert (RESULT/'full_profiling_report.md').is_file()
    return final,read('speed_summary.json')

def unavailable_profile(head):
    prior=verified('exp0230','full_profiling_report.md');table=verified('exp0230','module_table.md').read_text()
    text=f'''# EXP0232 complete profiling record: software acceptance boundary

Source codex/exp-0232-w4f16-awq-input-equalization, closure commit {head}.
Evidence {RESULT}; artifacts {OUTPUT}. Direct control C64, candidate E64; F/C8 software references. Quality execution unit: packed FP16 GPU M64+16 teacher forcing. AWQ failed or was inconclusive at the frozen software PPL gate; no EXP232 device run was incurred. N/A is unavailable evidence, never measured zero.

|Required section|Repeat1 control/candidate/delta|Repeat10 control/candidate/delta|Reason|
|---|---|---|---|
|Complete Host wall / prefill / continuous decode|N/A|N/A|Software quality boundary|
|DSP invocation / setup / teardown / ledger / unattributed|N/A|N/A|No EXP232 device execution|
|All additive block stages|N/A|N/A|No EXP232 device execution|
|Projection DMA / HMX / unpack / waits / lifetime / workers|N/A|N/A|No EXP232 device execution|
|Attention QK / Softmax / AV / packing / waits / tasks|N/A|N/A|No EXP232 device execution|
|MLP GateUp / SwiGLU / Down / slots / staging / publication|N/A|N/A|No EXP232 device execution|
|DDR bytes / DMA descriptors / spill / HMX / VTCM / FastRPC|N/A|N/A|No physical performance claim|
|Device output hashes / mismatch / maximum LSB|N/A|N/A|Software correctness retained separately|

Five short / ten formal rotated pairs: N/A at this boundary. Engine counters may overlap and are not additive. Independent algebra, RTN/scale-choice, original FP16 invariance,196packing checks,112HF forward checks,28finite calibration checkpoints and repeat/causal/CE checks pass; see REPORT.md and immutable raw evidence.

## Stable historical three-recipe overview

M64 microseconds (% complete Host wall). Unchanged EXP230 evidence, not an E64 profile. W4F16 C64 is an experimental candidate without baseline promotion. F16F16/W4U8 are nonpaired EXP218 references. W4bytes differ; no activation-only attribution. Prior complete profile: {prior}, SHA256 {sha(prior)}.

{table}

## End-to-end speed

E64 prefill/decode: N/A. Historical EXP230 C64:64prompt tokens /63262.995us =1011.649859tok/s;15continuous-decode tokens /1389448.9595us =10.795647tok/s. These are directly measured prior per-channel results, never extrapolated from software evaluation time.
'''
    with (RESULT/'full_profiling_report.md').open('x') as f:f.write(text)

def main():
    preflight();frozen();assert not (RESULT/'closure.json').exists()
    primary=read('summary_primary.json');final=read('summary_combined.json' if primary['reserve_trigger'] else 'summary_primary.json')
    check_summary(final);package=audit_export()
    dev={v:read(f'software/development_{v}.json') for v in VARIANTS}
    for v,r in dev.items():
        assert r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
        if v!='E64':assert r['exact_prior_development_regression']
        assert abs(math.exp(math.fsum(x for row in r['samples'] for x in row['nll'])/2048)-r['ppl'])<1e-10
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    actual,speed=(device_evidence() if final['status']=='pass' else (None,None))
    if actual is None:unavailable_profile(head)
    artifacts={str(p.relative_to(OUTPUT)):dict(sha256=stream_sha(p),bytes=p.stat().st_size)
        for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write('artifacts_sha256.json',dict(root=str(OUTPUT),files=artifacts,all28_hidden_finite=True,
        all196_projection_pack_checks=True,all112_forward_checks=True))
    gpu=json.loads(subprocess.check_output(['/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python','-c',
        'import json,torch,transformers,numpy;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,gpu=torch.cuda.get_device_name(0))))'],text=True))
    environment=dict(cpu_python=sys.version,cpu_torch=torch.__version__,cpu_transformers=transformers.__version__,
        numpy=np.__version__,platform=platform.platform(),gpu=gpu,source_head=head)
    assert all(r['torch_version']==gpu['torch'] for r in dev.values())
    write('environment_at_closure.json',environment)
    status=actual['status']['A'] if actual else final['status']
    text=['# EXP0232 AWQ-style per-channel input equalization','',
        f"Software acceptance: **{final['status']}**, {final['documents']} documents / {final['targets']} targets; reserve used: {primary['reserve_trigger']}. Final experiment quality gate: **{status}**. No baseline promotion.",
        '', '## Descriptive development', '', '|Variant|PPL|','|---|---:|']
    for v,r in dev.items():text.append(f"|{v}|{r['ppl']:.6f}|")
    text+=['','F/C8/C64 reproduce EXP230 per-token development NLL/top1 exactly. Development was never used to select scales or a model; one E64 candidate only.',
        '', '## Independent final software PPL', '', '|Stratum|F|C8|C64|E64|E64/F ratio [95% CI]|Limit|',
        '|---|---:|---:|---:|---:|---|---:|']
    for name,g in final['statistics'].items():
        q=g['vs_F16']['E64'];p=g['ppl'];ci=q['ratio_ci95']
        text.append(f"|{name}|{p['F']:.6f}|{p['C8']:.6f}|{p['C64']:.6f}|{p['E64']:.6f}|{q['ppl_ratio']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]|{q['limit']}|")
    overall=final['statistics']['overall']
    text+=['',f"Matched E64/C64 ratio: {overall['E64_vs_C64']}. Descriptive E64/C8 ratio: {overall['E64_vs_C8']}.",
        '', 'Both E64 and C64 use65536calibration tokens/context128 and identical CPU GPTQ. C8 uses8192. E64 adds AWQ scale-search on the nested8192calibration subset; no final/development selection. Shared tokenizer/M64+16/masks/canonical FP16 GPU. Paired stratified document bootstrap5000seed232, overall5% and every language/domain/cell10%. All36PPL aggregates independently reproduced from raw token NLL. These are short-context conditional PPL, not long-context benchmark acceptance. Different experiments use different independent final documents; compare paired ratios, not raw PPL across experiments.',
        '', '## Method and correctness', '',
        'Official mit-han-lab/llm-awq commit d6e797a42b9ef7778de8ee2352116e0f48a78d61; source/license/provenance retained. Activation-only20ratio scale search on frozen C64 calibration trajectories with symmetric per-row RTN proxy; actual export uses unchanged EXP224 three-range output-aware GPTQ. RTN proxy and final GPTQ objectives differ. Fresh original FP32 folding: input gamma/QKV, post gamma/GateUp, linear Up rows/Down columns. V/O scaling skipped for GQA. No commuting through SiLU, rotation, grouping, LPBQ or upstream folded weights.',
        '', 'FP64 legal compensation and independent NumPy RTN/scale/choice oracles pass. All28real-layer equivalence and whole-model unquantized FP16 hidden/logit checks meet NRMSE<=.003 and cosine>=.99999. Global results: '+json.dumps(read('invariance.json')['checks'])+'. Original embedding/final norm/head/QK norms frozen. All196packed roundtrips,112HF forward comparisons and28finiteFP16(512,128,2048)hidden checkpoints verified. Actual export source archive and allfivequantizer files match; three core GPTQ files exactly match C64.',
        '', 'All1024new source windows independently reconstructed; document/text/32gram exclusions include all prior training/calibration/validation/final roles, EXP230/231 and allWikiTexttrain titles. Final data is now exposed and cannot be reused as new independent selection/acceptance data. All commands/source archives/models/scales/checkpoints have retained hash ledgers. Repeat/causal-mask/independent CE gates pass without numerical threshold changes.',
        '', '## Device and profiling boundary', '']
    if actual:
        text+=['Actual DSP used the same fixed data already scored by software; it remained disjoint from training and no backend scores changed the weights. Legacy230sentinel was copied solely for exact control regression, separately hashed; it was not final evaluation data. ABI108/runtime unchanged. Before/full/after, independent software/DSP reference and physical checks pass. All27DSP aggregates independently reproduced. Five short/ten formal rotated C64/E64 pairs and complete profiling retained.',
            json.dumps(actual,indent=2),json.dumps(speed['times'],indent=2)]
    else:text+=['Software acceptance failed or remained inconclusive, so EXP232 DSP correctness, physical gates,5short/10formal and E64 E2E speed are N/A. No device deployment or runtime change. full_profiling_report.md retains the complete unavailable sections and verified historical EXP230 references.']
    text+=['','## Ordered sequence closure','',
        'EXP230 calibration coverage/budget, EXP231 symmetric group128 software diagnostic, and EXP232 AWQ input equalization are complete. This ends PC051 authorized escalation. Discuss next direction using these results; do not automatically start another method or promote a baseline. F16F16 and W4U8 runtime/packages remain frozen. Group128 was tested only at8192calibration tokens; its result does not establish performance at65536 or on DSP.']
    with (RESULT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    closure=dict(experiment='EXP-0232',execution_state='completed',evidence_validity='valid',local_gate=status,
        adoption_status='pending',source_head=head,quality=final,device_quality=actual,
        development={v:r['ppl'] for v,r in dev.items()},manifest_sha256=package['manifest_sha256'],
        dataset_sha256=sha(RESULT/'dataset.json'),data_freeze_sha256=sha(RESULT/'dataset_freeze.json'),
        report_sha256=sha(RESULT/'REPORT.md'),full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),
        artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),artifact_files=len(artifacts),
        environment_sha256=sha(RESULT/'environment_at_closure.json'),device_speed=speed['times'] if speed else 'N/A_software_acceptance_boundary',
        other_recipes='frozen',baseline_promoted=False,next_direction='discuss_PC051_sequence_complete')
    write('closure.json',closure)
    write('evidence_sha256.json',{str(p.relative_to(RESULT)):stream_sha(p)
        for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
    print('EXP0232_CLOSED',status,'PC051_sequence_complete',flush=True)

if __name__=='__main__':main()
