#!/usr/bin/env python3
"""Close the bounded software diagnostic after all stage command logs close."""
import json,math,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch,transformers
from data_exp0234 import RESULT,OUTPUT,SOURCE,MEMORY,write,sha,verified,preflight
from export_exp0234 import frozen
from group_exp0234 import FORMAT,read_weight
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
    dev={v:json.loads((RESULT/f'software/development_{v}.json').read_text()) for v in ['F','G8','C64','G64']}
    for v,r in dev.items():
        assert r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
        if v!='G64':assert r['exact_prior_development_regression']
        assert abs(math.exp(math.fsum(x for row in r['samples'] for x in row['nll'])/2048)-r['ppl'])<1e-10
    oracle=json.loads((RESULT/'group_oracle.json').read_text());assert oracle['pass_all']
    package=json.loads((RESULT/'G64/package.json').read_text());root=OUTPUT/'G64'
    assert sha(root/'manifest.json')==package['manifest_sha256']
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['format']==FORMAT
    stats=json.loads((RESULT/'G64/weight_stats.json').read_text());assert len(stats)==196
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
    forward=json.loads((RESULT/'G64/calibration_forward_checks.json').read_text());assert len(forward)==112
    assert all(r['finite'] and r['nrmse']<=.003 and r['cosine']>=.99999 for r in forward)
    hidden=list((OUTPUT/'checkpoints/G64').glob('*.npy'));assert len(hidden)==28
    for p in hidden:
        a=np.load(p,mmap_mode='r');assert a.shape==(512,128,2048) and a.dtype==np.float16
        for start in range(0,512,8):assert np.isfinite(a[start:start+8]).all(),p
    logs=list((RESULT/'commands').glob('*.log'))
    commands=[]
    for p in logs:
        m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['log_sha256']
        archive=OUTPUT/'artifacts'/m['source_head']/'source.tar'
        assert stream_sha(archive)==m['source_archive_sha256'];commands.append(m)
    assert len(commands)==len(list((RESULT/'commands').glob('*.json')))
    actual=next(c for c in commands if any('export_exp0234.py' in str(a) for a in c['command']) and c['returncode']==0)
    import tarfile,hashlib
    with tarfile.open(OUTPUT/'artifacts'/actual['source_head']/'source.tar') as archive:
        for n,h in package['quantizer_source'].items():
            assert hashlib.sha256(archive.extractfile('scripts/'+n).read()).hexdigest()==h,n
    write('export_execution_provenance.json',dict(actual_start_source=actual['source_head'],completion_source_label=manifest['source_head'],all_quantizer_files_identical=True,source_of_execution='retained stage command and source archive'))

    artifacts={str(p.relative_to(OUTPUT)):dict(sha256=stream_sha(p),bytes=p.stat().st_size) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write('artifacts_sha256.json',dict(root=str(OUTPUT),files=artifacts,all28_hidden_finite=True,all196_projection_pack_checks=True))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    gpu=subprocess.check_output(['/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python','-c',
        'import json,torch,transformers,numpy;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,gpu=torch.cuda.get_device_name(0))))'],text=True).strip()
    environment=dict(cpu_python=sys.version,cpu_torch=torch.__version__,cpu_transformers=transformers.__version__,
        numpy=np.__version__,platform=platform.platform(),gpu=json.loads(gpu),source_head=head)
    assert all(r['torch_version']==environment['gpu']['torch'] for r in dev.values())
    write('environment_at_closure.json',environment)
    next_direction='approved_EXP235_C64_sensitivity'
    text=['# EXP0234 static group128 software diagnostic','',
        f"Independent software acceptance: **{final['status']}**. {final['documents']} documents, {final['targets']} targets. Reserve used: {primary['reserve_trigger']}. No baseline promotion.",
        '', '## Matched development regression', '', '|Variant|PPL|','|---|---:|']
    for v,r in dev.items():text.append(f"|{v}|{r['ppl']:.6f}|")
    text+=['','F/C64 reproduce stored EXP230 development per-token NLL/top1 exactly; G8 reproduces EXP231 G128 exactly. Development is reused descriptive evidence, never independent final evaluation or a selection input. There is exactly one G64 candidate.',
        '', '## Independent final PPL (packed FP16 GPU software)', '',
        '|Stratum|F|G8|C64|G64|G64/F ratio [95% CI]|Limit|', '|---|---:|---:|---:|---:|---|---:|']
    for name,g in final['statistics'].items():
        q=g['vs_F16']['G64'];p=g['ppl'];ci=q['ratio_ci95']
        text.append(f"|{name}|{p['F']:.6f}|{p['G8']:.6f}|{p['C64']:.6f}|{p['G64']:.6f}|{q['ppl_ratio']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]|{q['limit']}|")
    overall=final['statistics']['overall']
    text+=['',f"G64/G8 matched-group calibration-budget ratio: {overall['G64_vs_G8']}. G64/C64 matched64K grouping ratio: {overall['G64_vs_C64']}.",
        '', 'G64 and C64 use identical65536 calibration tokens; G64/G8 compares64K versus8K atidenticalgroup128 format. G8 is frozenEXP231; no recalibration. Allthree use the same frozenhead/norms/embedding. Every final comparison shares original tokenizer, M64 prompt,16targets, masks and FP16 execution. Paired stratified document bootstrap5000seed234; overall5%, every language/domain/cell10%, fixed reserve rule. All36 aggregate PPL values independently reduced from raw token NLL with math.fsum. This is short-context conditional PPL, not published long-context benchmark acceptance.',
        '', '## Method and correctness', '',
        'Fresh original transformer weights, unchanged original gamma/embedding/norms and inherited per-channel W4 head. Symmetric[-7,7], FP32 static scales per128 original input columns; no rotation, LPBQ or second-level scale quantization. Global act-order uses original-column group lookup. Three full GPTQ trials, per-group absmax/midpoint/weight-L2.4 ranges, row-wise final projection output-SSE choice. Original CPU Gram/damping0.01 and staged quantized inputs retained.',
        '', 'Independent dense Schur-elimination codes match allthree trials; NumPy group clipping and output-SSE/choices match; dead/zero/multiple-group/permutation tests and one-group per-channel equivalence pass. All196 projections pass independent packed-file NumPy FP16 reconstruction;112 staged-vs-HF forwards pass;28retained hidden checkpoints are finiteFP16(512,128,2048). All model tensors, source archives, source-data windows and command logs retain verified SHA256. Repeat, causal-mask and independent CE checks pass for every reported run.',
        '', 'The shared PC052 panel is exactly EXP233 dataset.json, frozen and independently reconstructed before allthree phases, excluding prior roles throughEXP232 by document/text/32grams. This is disclosed paired reuse, not a newly independent dataset perphase. F/C64 controls are byte-identical verified EXP233 scoring evidence; G8/G64 are scored here. Unused inherited AWQ metadata is covered by the referenced EXP233 erratum; actualgroupcalibration is512x128. No scoring-based quantizer changes.',
        '', '## Profiling boundary and next action', '',
        'This is a software-only format diagnostic. Grouped DSP correctness, VTCM/DMA/HMX costs,5short/10formal timings and E2E token/s are N/A: no grouped DSP runtime was built or executed. Dequantized PyTorch forward time is not a group-W4 speed result. FP32group metadata is0.25bit/weight,6.25% of raw4bit payload before headers/alignment; future DSP deployment must measure its cost.',
        '',f'Next authorized action: {next_direction}. F16F16 and W4U8 runtime/packages remain frozen. No automatic baseline promotion. See full_profiling_report.md for all unavailable sections and exact prior EXP230 measured references.']
    with (RESULT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    table=verified('exp0230','module_table.md').read_text();prior_profile=verified('exp0230','full_profiling_report.md')
    profile=f'''# EXP0234 complete profiling record: host-only software diagnostic

Source branch codex/exp-0234-w4f16-group128-64k; closure commit {head}.
Evidence {RESULT}; artifacts {OUTPUT}. Controls C64 forgrouping andG8 forcalibrationbudget; candidateG64; floatingF reference. Quality execution unit is a packed-weight FP16 software M64+16 teacher-forcing forward. No target DSP runtime for this format exists in this experiment. Paired profiling rounds, repeat-one and repeat-ten: N/A.

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

The following M64 table uses microseconds and percent of complete Host wall. It is unchanged EXP230 evidence, **not a G64 profile**. Its W4F16 C64 column is an experimental quality candidate, not a promoted baseline; F16F16/W4U8 are nonpaired EXP218 references. W4bytes differ, so no activation-only attribution. Full repeat1/repeat10 counters and provenance: {prior_profile}, SHA256 {sha(prior_profile)}.

{table}

## End-to-end speed

G64 prefill/decode: N/A (no grouped device implementation). Historical EXP230 C64:64prompt tokens /63262.995us =1011.649859tok/s;15continuous-decode tokens /1389448.9595us =10.795647tok/s. These are prior per-channel measurements only. No extrapolation from software evaluation time.
'''
    with (RESULT/'full_profiling_report.md').open('x') as f:f.write(profile)
    closure=dict(experiment='EXP-0234',execution_state='completed',evidence_validity='valid',local_gate=final['status'],
        adoption_status='pending',source_head=head,quality=final,development={v:r['ppl'] for v,r in dev.items()},
        dataset_sha256=sha(RESULT/'dataset.json'),data_freeze_sha256=sha(RESULT/'dataset_freeze.json'),
        manifest_sha256=package['manifest_sha256'],export_actual_source=actual['source_head'],report_sha256=sha(RESULT/'REPORT.md'),
        full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),
        artifact_files=len(artifacts),environment_sha256=sha(RESULT/'environment_at_closure.json'),
        device_speed='N/A_software_only',other_recipes='frozen',baseline_promoted=False,next_direction=next_direction)
    write('closure.json',closure)
    write('evidence_sha256.json',{str(p.relative_to(RESULT)):stream_sha(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
    print('EXP0234_CLOSED',final['status'],next_direction,flush=True)

if __name__=='__main__':main()
