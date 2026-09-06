#!/usr/bin/env python3
"""Final evidence closure. Run directly after all stage wrappers have exited."""
import hashlib,json,math,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import transformers
from data_exp0230 import RESULT,OUTPUT,SOURCE,write,sha,preflight,verified
from export_exp0230 import frozen
from device_exp0230 import variants
from evaluate_exp0230 import package

def retain_artifact_ledger():
    """Hash all retained model/intermediate files, reusing only identical inodes."""
    cache={};files={}
    for p in sorted(OUTPUT.rglob('*')):
        if not p.is_file():continue
        st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns) if st.st_ino else str(p)
        if key not in cache:
            h=hashlib.sha256()
            with p.open('rb') as f:
                while chunk:=f.read(8*1024*1024):h.update(chunk)
            now=p.stat()
            assert (now.st_size,now.st_mtime_ns,now.st_ino)==(st.st_size,st.st_mtime_ns,st.st_ino)
            cache[key]=h.hexdigest()
        files[str(p.relative_to(OUTPUT))]=dict(sha256=cache[key],bytes=st.st_size)
    for v,batch in [('C8',64),('C64',512)]:
        paths=list((OUTPUT/'checkpoints'/v).glob('layer*_hidden.npy'));assert len(paths)==28
        for p in paths:
            x=np.load(p,mmap_mode='r')
            assert x.shape==(batch,128,2048) and x.dtype==np.float16
            assert all(np.isfinite(x[i:i+8]).all() for i in range(0,batch,8))
        manifest=json.loads((package(v)/'manifest.json').read_text())
        for n,d in manifest['files'].items():
            assert files[v+'/'+n.replace(chr(92),'/')]['sha256']==d['sha256']
        stats=json.loads((RESULT/v/'weight_stats.json').read_text())
        for d in stats.values():
            n=str(Path(d['clip_stats_path']).relative_to(OUTPUT))
            assert files[n]['sha256']==d['clip_stats_sha256']
    for p in (RESULT/'commands').glob('*.json'):
        d=json.loads(p.read_text());n='artifacts/'+d['source_head']+'/source.tar'
        assert files[n]['sha256']==d['source_archive_sha256']
    write('artifacts_sha256.json',dict(root=str(OUTPUT),files=files,
        all56_hidden_checkpoints_shape_dtype_finite=True,all_packages_clip_stats_source_archives_verified=True))
    gpu=json.loads(subprocess.check_output([
        '/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python','-c',
        'import json,sys,torch,numpy,transformers;print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,transformers=transformers.__version__)))'],text=True))
    for v in ['F','A0','C8','C64']:
        assert json.loads((RESULT/f'software/development_{v}.json').read_text())['torch_version']==gpu['torch']
    write('environment_at_closure.json',dict(captured_at_closure=True,platform=platform.platform(),
        cpu=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
                 transformers=transformers.__version__,torch_build=torch.__config__.show()),gpu=gpu,
        export_threads=16,export_device='CPU',quantizer_source_unchanged=True))
    return len(files)

def main():
    preflight();frozen();vs=variants();selection=json.loads((RESULT/'selection.json').read_text())
    primary=json.loads((RESULT/'primary_summary.json').read_text())
    combined=primary['reserve_required']
    summary=json.loads((RESULT/('combined_summary.json' if combined else 'primary_summary.json')).read_text())
    for n in ['independent_data_audit.json','development_device_check.json','before_gate.json','after_gate.json']:
        assert json.loads((RESULT/n).read_text())['pass_all']
    for log in (RESULT/'commands').glob('*.log'):
        assert log.with_suffix('.json').exists(),('stage still open',log)
        meta=json.loads(log.with_suffix('.json').read_text());assert sha(log)==meta['log_sha256']
    speed=json.loads((RESULT/'speed_summary.json').read_text())
    artifact_count=retain_artifact_ledger()
    for v in ['C8','C64']:
        d=json.loads((RESULT/v/'package.json').read_text())
        assert sha(package(v)/'manifest.json')==d['manifest_sha256']
        for n,h in json.loads((RESULT/v/'frozen_files.json').read_text()).items():assert sha(package(v)/n)==h
    # Independent unstratified token reduction reproduces every equally balanced stratum.
    dataset=json.loads((RESULT/'dataset.json').read_text())
    lookup={r['id']:r for r in dataset['samples']}
    for key,v in zip(['F','A0','A'],vs):
        rows=[]
        for split in (['primary','reserve'] if combined else ['primary']):
            for p in sorted((RESULT/f'device/{split}').glob('*_'+v+'.validated.json')):
                rows.extend(json.loads(p.read_text())['samples'])
        assert len(rows)==summary['documents']
        for group,g in summary['statistics'].items():
            losses=[s['nll'] for row in rows if group=='overall' or group==lookup[row['id']]['cell']
                        or group in lookup[row['id']]['cell'].split('_') for s in row['steps']]
            actual=math.exp(math.fsum(losses)/len(losses))
            assert abs(actual-g['ppl'][key])<1e-10,(group,key)
    status=summary['status']['A'];selected=selection['selected']
    text=['# EXP0230 GPTQ calibration coverage/budget','',
        f'Candidate selected before final testing: {selected}. Independent DSP acceptance: **{status}**. '
        f'{summary["documents"]} documents, {summary["targets"]} scored targets; reserve used: {combined}. No baseline promoted.','',
        '## Development (packed FP16 GPU software; not actual DSP)','',
        '|Variant|Calibration tokens|PPL|','|---|---:|---:|']
    for v in ['F','A0','C8','C64']:
        text.append(f'|{v}|{dict(F="N/A",A0=8192,C8=8192,C64=65536)[v]}|{selection["development"][v]["ppl"]:.6f}|')
    text+=['','### A0 module sensitivity (development software only)','',
        '|FP16-restored family|PPL|Change vs A0|','|---|---:|---:|']
    for family in ['attention','mlp','head']:
        d=json.loads((RESULT/f'diagnostics/{family}/software/development_A0.json').read_text())
        assert d['all_other_parameters_and_buffers_exact'] and not d['affects_selection']
        text.append(f'|{family}|{d["ppl"]:.6f}|{100*(d["ppl"]/selection["development"]["A0"]["ppl"]-1):+.3f}%|')
    text+=['','These hybrids are diagnostic only, not candidates or deployable mixed-precision claims. '
        'Each restores one family from the original checkpoint while every other A0 tensor and buffer remains exact. '
        'Effects interact and are not additive. No diagnostic uses final data or changes candidate selection.']
    text+=['','C8 and C64 use the unchanged original-coordinate CPU EXP224 GPTQ, act-order, staged quantized inputs '
        'and three output-aware row-scale candidates. C8 is a document-balanced subset of C64. '
        'Original norms/embedding/head are frozen; only transformer calibration differs. Selection uses development NLL, tie1e-6 favors C8.','',
        '## Untouched final (actual DSP, matched token/context/masks)','',
        '|Stratum|FP16 PPL|A0 PPL|Selected PPL|A0/F16 ratio [95% CI]|Selected/F16 ratio [95% CI]|Limit|',
        '|---|---:|---:|---:|---|---|---:|']
    for group,g in summary['statistics'].items():
        comps=[]
        for k in ['A0','A']:
            d=g['vs_F16'][k];comps.append(f'{d["ppl_ratio"]:.6f} [{d["ratio_ci95"][0]:.6f}, {d["ratio_ci95"][1]:.6f}]')
        text.append(f'|{group}|{g["ppl"]["F"]:.6f}|{g["ppl"]["A0"]:.6f}|{g["ppl"]["A"]:.6f}|{comps[0]}|{comps[1]}|{g["vs_F16"]["A"]["limit"]:.2f}|')
    text+=['', 'Fixed gates: overall<=1.05, each language/domain/cell<=1.10. Paired stratified document bootstrap5000seed230. '
        'Passing requires point and upperCI gates; point failure is fail, passing points with crossing intervals are inconclusive at bounded budget. '
        'This is M64+16 conditional short-context PPL, not a published full benchmark or long-context acceptance. '
        'All27DSP aggregate PPL values independently reproduced via math.fsum.','',
        '## Correctness and provenance','',
        'All1664 source windows independently reconstructed; document/text/32gram separation from prior calibration/training/'
        'validation/EXP229/qbh includes holdout. Both original qbh controls reproduce exactly; within-session and before/full/after '
        'sentinels match. Packed software/device development checks pass. All device targets finite,8MiB VTCM,no intermediateDDR/spill,'
        'valid self-computed cache progression. Original model and inherited tensor hashes verified. '
        'Export source commits may precede added evaluation/orchestration files; command source archives retain the precise executed '
        'export implementation and unchanged quantizer source hashes. See commands and per-layer export records. '
        f'All{artifact_count} retained model/intermediate files have an artifact ledger; all56 hidden checkpoints '
        'have verified FP16 shapes and finite values. Package tensors, clipping statistics and source archives match their recorded hashes.','',
        '## Profiling','',
        'One warmup,5short,10rotated selected/A0 pairs complete;320invocation and8960layer ledgers valid. '
        'See full_profiling_report.md and module_table.md for complete measurements and historical nonpaired other-recipe columns.',
        json.dumps(speed['times'],indent=2),'',
        '## Next authorized direction','',
        'Stop escalation if the fixed acceptance passes. Otherwise PC051 permits the next separately registered group128 software '
        'diagnostic, then AWQ input-channel equalization if still needed. W4A8 remains frozen. No automatic promotion or grouped DSP deployment.','']
    (RESULT/'REPORT.md').write_text('\n'.join(text))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    write('closure.json',dict(experiment='EXP-0230',execution_state='completed',evidence_validity='valid',
        local_gate=status if status in ['pass','fail'] else 'not_applicable',adoption_status='pending',
        selected=selected,selected_before_final=True,quality=summary,development=selection['development'],
        speed=speed['times'],source_head=head,runtime_unchanged=True,other_recipes='frozen',baseline_promoted=False,
        dataset_sha256=sha(RESULT/'dataset.json'),data_freeze_sha256=sha(RESULT/'dataset_freeze.json'),
        report_sha256=sha(RESULT/'REPORT.md'),summary_sha256=sha(RESULT/('combined_summary.json' if combined else 'primary_summary.json')),
        full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),
        artifact_files=artifact_count,artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),
        environment_at_closure_sha256=sha(RESULT/'environment_at_closure.json'),
        package_manifest_sha256={v:sha(package(v)/'manifest.json') for v in ['F','A0','C8','C64']},
        next_direction='stop_acceptance_pass' if status=='pass' else 'PC051_group128_software_diagnostic'))
    write('evidence_sha256.json',{str(p.relative_to(RESULT)):sha(p) for p in sorted(RESULT.rglob('*')) if p.is_file()})
    print('EXP0230_CLOSED',selected,status,flush=True)

if __name__=='__main__':main()
