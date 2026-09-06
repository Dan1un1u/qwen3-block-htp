#!/usr/bin/env python3
"""Close the bounded software diagnostic after all stage command logs close."""
import json,math,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch,transformers
from data_exp0231 import RESULT,OUTPUT,SOURCE,MEMORY,write,sha,verified,preflight
from export_exp0231 import frozen
from group_exp0231 import FORMAT,read_weight
import rotation_exp0219 as rot

def stream_sha(p):
    import hashlib
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    preflight();d=frozen();assert not (RESULT/'closure.json').exists()
    primary=json.loads((RESULT/'summary_primary.json').read_text())
    summary_name='summary_combined.json' if primary['reserve_trigger'] else 'summary_primary.json'
    final=json.loads((RESULT/summary_name).read_text())
    assert not final['reserve_trigger']
    for n,h in final['inputs'].items():assert sha(RESULT/n)==h,n
    assert final['independent_raw_token_reduction_count']==36
    dev={v:json.loads((RESULT/f'software/development_{v}.json').read_text()) for v in ['F','C8','C64','G128']}
    for v,r in dev.items():
        assert r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
        if v!='G128':assert r['exact_prior_development_regression']
        assert abs(math.exp(math.fsum(x for row in r['samples'] for x in row['nll'])/2048)-r['ppl'])<1e-10
    oracle=json.loads((RESULT/'group_oracle.json').read_text());assert oracle['pass_all']
    package=json.loads((RESULT/'G128/package.json').read_text());root=OUTPUT/'G128'
    assert sha(root/'manifest.json')==package['manifest_sha256']
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['format']==FORMAT
    stats=json.loads((RESULT/'G128/weight_stats.json').read_text());assert len(stats)==196
    for name,st in stats.items():
        assert st['packed_roundtrip'] and st['independent_numpy_export_equal']
        assert sha(st['clip_stats_path'])==st['clip_stats_sha256']
        layer,projection=name.split('/')
        assert sha(root/layer/(projection+'_group128_codes_hmx.bin'))==st['codes_sha256']
        assert sha(root/layer/(projection+'_group128_scales_f32.bin'))==st['scales_sha256']
        value=read_weight(root/layer,projection,st['rows'],st['width']);assert np.isfinite(value).all()
    for n,e in manifest['files'].items():assert sha(root/n)==e['sha256'],n
    for n,h in manifest['inherited_files'].items():assert sha(Path(manifest['frozen_base_package'])/n)==h,n
    assert sha(Path(manifest['frozen_base_package'])/'manifest.json')==manifest['frozen_base_manifest_sha256']
    forward=json.loads((RESULT/'G128/calibration_forward_checks.json').read_text());assert len(forward)==112
    assert all(r['finite'] and r['nrmse']<=.003 and r['cosine']>=.99999 for r in forward)
    hidden=list((OUTPUT/'checkpoints/G128').glob('*.npy'));assert len(hidden)==28
    for p in hidden:
        a=np.load(p,mmap_mode='r');assert a.shape==(64,128,2048) and a.dtype==np.float16
        for start in range(0,64,8):assert np.isfinite(a[start:start+8]).all(),p
    logs=list((RESULT/'commands').glob('*.log'))
    commands=[]
    for p in logs:
        m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['log_sha256']
        archive=OUTPUT/'artifacts'/m['source_head']/'source.tar'
        assert stream_sha(archive)==m['source_archive_sha256'];commands.append(m)
    assert len(commands)==len(list((RESULT/'commands').glob('*.json')))
    artifacts={str(p.relative_to(OUTPUT)):dict(sha256=stream_sha(p),bytes=p.stat().st_size) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write('artifacts_sha256.json',dict(root=str(OUTPUT),files=artifacts,all28_hidden_finite=True,all196_projection_pack_checks=True))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    gpu=subprocess.check_output(['/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python','-c',
        'import json,torch,transformers,numpy;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,gpu=torch.cuda.get_device_name(0))))'],text=True).strip()
    environment=dict(cpu_python=sys.version,cpu_torch=torch.__version__,cpu_transformers=transformers.__version__,
        numpy=np.__version__,platform=platform.platform(),gpu=json.loads(gpu),source_head=head)
    assert all(r['torch_version']==environment['gpu']['torch'] for r in dev.values())
    write('environment_at_closure.json',environment)
    next_direction='discuss_grouped_DSP_deployment_stop_accuracy_escalation' if final['status']=='pass' else 'separately_register_approved_AWQ_input_channel_equalization'
    text=['# EXP0231 static group128 software diagnostic','',
        f"Independent software acceptance: **{final['status']}**. {final['documents']} documents, {final['targets']} targets. Reserve used: {primary['reserve_trigger']}. No baseline promotion.",
        '', '## Matched development regression', '', '|Variant|PPL|','|---|---:|']
    for v,r in dev.items():text.append(f"|{v}|{r['ppl']:.6f}|")
    text+=['','F/C8/C64 reproduce all stored EXP230 development per-token NLL and top1 results exactly. Development is reused descriptive evidence, never independent final evaluation or a selection input. There is exactly one G128 candidate.',
        '', '## Independent final PPL (packed FP16 GPU software)', '',
        '|Stratum|F|C8|C64|G128|G128/F ratio [95% CI]|Limit|', '|---|---:|---:|---:|---:|---|---:|']
    for name,g in final['statistics'].items():
        q=g['vs_F16']['G128'];p=g['ppl'];ci=q['ratio_ci95']
        text.append(f"|{name}|{p['F']:.6f}|{p['C8']:.6f}|{p['C64']:.6f}|{p['G128']:.6f}|{q['ppl_ratio']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]|{q['limit']}|")
    overall=final['statistics']['overall']
    text+=['',f"G128/C8 matched-calibration ratio: {overall['G128_vs_C8']}. G128/C64 descriptive ratio: {overall['G128_vs_C64']}.",
        '', 'Both G128 and C8 use identical 8192 calibration tokens. C64 uses65536 and is a stronger descriptive control; do not attribute its difference to group size alone. Every final comparison shares original tokenizer, M64 prompt,16targets, masks and FP16 execution. Paired stratified document bootstrap5000seed231; overall5%, every language/domain/cell10%, fixed reserve rule. All36 aggregate PPL values independently reduced from raw token NLL with math.fsum. This is short-context conditional PPL, not published long-context benchmark acceptance.',
        '', '## Method and correctness', '',
        'Fresh original transformer weights, unchanged original gamma/embedding/norms and inherited per-channel W4 head. Symmetric[-7,7], FP32 static scales per128 original input columns; no rotation, LPBQ or second-level scale quantization. Global act-order uses original-column group lookup. Three full GPTQ trials, per-group absmax/midpoint/weight-L2.4 ranges, row-wise final projection output-SSE choice. Original CPU Gram/damping0.01 and staged quantized inputs retained.',
        '', 'Independent dense Schur-elimination codes match allthree trials; NumPy group clipping and output-SSE/choices match; dead/zero/multiple-group/permutation tests and one-group per-channel equivalence pass. All196 projections pass independent packed-file NumPy FP16 reconstruction;112 staged-vs-HF forwards pass;28retained hidden checkpoints are finiteFP16(64,128,2048). All model tensors, source archives, source-data windows and command logs retain verified SHA256. Repeat, causal-mask and independent CE checks pass for every reported run.',
        '', 'New primary/reserve sources exclude prior training/calibration/evaluation, including all EXP230 roles, by document identity, text hash and32-token windows;1024source windows independently reconstructed. Used final data is now exposed and cannot become new independent calibration/selection data. No scoring-based quantizer changes.',
        '', '## Profiling boundary and next action', '',
        'This is a software-only format diagnostic. Grouped DSP correctness, VTCM/DMA/HMX costs,5short/10formal timings and E2E token/s are N/A: no grouped DSP runtime was built or executed. Dequantized PyTorch forward time is not a group-W4 speed result. FP32group metadata is0.25bit/weight,6.25% of raw4bit payload before headers/alignment; future DSP deployment must measure its cost.',
        '',f'Next authorized action: {next_direction}. F16F16 and W4U8 runtime/packages remain frozen. No automatic baseline promotion. See full_profiling_report.md for all unavailable sections and exact prior EXP230 measured references.']
    with (RESULT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    table=verified('exp0230','module_table.md').read_text();prior_profile=verified('exp0230','full_profiling_report.md')
    profile=f'''# EXP0231 complete profiling record: host-only software diagnostic

Source branch codex/exp-0231-w4f16-group128-software; closure commit {head}.
Evidence {RESULT}; artifacts {OUTPUT}. Direct control C8; candidate G128; F and C64 descriptive controls. Quality execution unit is a packed-weight FP16 software M64+16 teacher-forcing forward. No target DSP runtime for this format exists in this experiment. Paired profiling rounds, repeat-one and repeat-ten: N/A.

|Required section|Repeat1 control/candidate/delta|Repeat10 control/candidate/delta|Reason|
|---|---|---|---|
|Complete Host wall / prefill / continuous decode|N/A|N/A|No grouped device execution|
|DSP invocation / setup / teardown / ledger / unattributed|N/A|N/A|No DSP timestamp recording|
|All additive block stages|N/A|N/A|No grouped runtime|
|Projection DMA / HMX / unpack / waits / lifetime / workers|N/A|N/A|No grouped runtime|
|Attention QK / Softmax / AV / packing / waits / tasks|N/A|N/A|No grouped runtime|
|MLP GateUp / SwiGLU / Down / slots / staging / publication|N/A|N/A|No grouped runtime|
|DDR bytes / DMA descriptors / spill / HMX / VTCM / FastRPC|N/A|N/A|No device use or physical performance claim|
|Device output hashes / mismatch / maximum LSB|N/A|N/A|Software implementation oracles documented in REPORT.md|

Engine/work counters can overlap and would not be additive; none are measured here. Software repeat/causal/CE checks, dense quantizer oracle, inherited tensor identity,196pack roundtrips,112forward checks and finite values pass. Quality scores and confidence intervals are in REPORT.md. N/A never means measured zero.

## Stable historical three-recipe overview

The following M64 table uses microseconds and percent of complete Host wall. It is unchanged EXP230 evidence, **not a G128 profile**. Its W4F16 C64 column is an experimental quality candidate, not a promoted baseline; F16F16/W4U8 are nonpaired EXP218 references. W4bytes differ, so no activation-only attribution. Full repeat1/repeat10 counters and provenance: {prior_profile}, SHA256 {sha(prior_profile)}.

{table}

## End-to-end speed

G128 prefill/decode: N/A (no grouped device implementation). Historical EXP230 C64:64prompt tokens /63262.995us =1011.649859tok/s;15continuous-decode tokens /1389448.9595us =10.795647tok/s. These are prior per-channel measurements only. No extrapolation from software evaluation time.
'''
    with (RESULT/'full_profiling_report.md').open('x') as f:f.write(profile)
    closure=dict(experiment='EXP-0231',execution_state='completed',evidence_validity='valid',local_gate=final['status'],
        adoption_status='pending',source_head=head,quality=final,development={v:r['ppl'] for v,r in dev.items()},
        dataset_sha256=sha(RESULT/'dataset.json'),data_freeze_sha256=sha(RESULT/'dataset_freeze.json'),
        manifest_sha256=package['manifest_sha256'],report_sha256=sha(RESULT/'REPORT.md'),
        full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),
        artifact_files=len(artifacts),environment_sha256=sha(RESULT/'environment_at_closure.json'),
        device_speed='N/A_software_only',other_recipes='frozen',baseline_promoted=False,next_direction=next_direction)
    write('closure.json',closure)
    write('evidence_sha256.json',{str(p.relative_to(RESULT)):stream_sha(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
    print('EXP0231_CLOSED',final['status'],next_direction,flush=True)

if __name__=='__main__':main()
