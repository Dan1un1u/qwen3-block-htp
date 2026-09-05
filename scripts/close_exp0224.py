#!/usr/bin/env python3
"""Close the fixed transformer clipping trial without altering its outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from statistics import mean
import numpy as np
import torch
import transformers
import eval_exp0218 as ev
import measure_exp0218 as measure
from output_scale_exp0224 import RESULT,OUTPUT,PREVIOUS,PREVIOUS_MODELS,CLIPPED,control_package,CANDIDATES
from experiment_exp0224 import VARIANTS,verify_origin
from measure_exp0224 import ALL,package_path
from rotation_exp0219 import write_json
from summarize_exp0218 import table

SOURCE=Path(__file__).resolve().parents[1]


def report():
    assert not (RESULT/'REPORT.md').exists()
    q=json.loads((RESULT/'quality_summary.json').read_text());speed=json.loads((RESULT/'speed_summary.json').read_text())
    selection={};checks={};rows=[]
    for v in VARIANTS:
        stats=json.loads((RESULT/v/'weight_stats.json').read_text());assert len(stats)==196
        counts=np.zeros(3,dtype=np.int64);ratio_sum=0.;sse=np.zeros(3);selected=0.
        for value in stats.values():
            a=np.load(value['clip_stats_path']);scores=a['candidate_output_sse'];choice=a['choice'];best=a['selected_output_sse']
            assert scores.shape==(value['rows'],3)
            assert np.array_equal(best,scores[np.arange(len(choice)),choice])
            assert (best<=scores[:,0]).all() and (best<=scores[:,2]).all()
            assert value['selection_positions']==8192
            counts+=np.bincount(choice,minlength=3);ratio_sum+=float((a['scale']/a['absmax_scale']).astype(np.float64).sum())
            sse+=scores.sum(0);selected+=float(best.sum())
        selection[v]=dict(rows=int(counts.sum()),candidate_names=CANDIDATES,choice_histogram=counts.tolist(),mean_selected_ratio=ratio_sum/counts.sum(),summed_projection_sse=sse.tolist(),summed_selected_sse=selected)
        rows.append([v,int(counts.sum()),*[f'{100*n/counts.sum():.2f}%' for n in counts],f'{ratio_sum/counts.sum():.4f}'])
        forward=json.loads((RESULT/v/'calibration_forward_checks.json').read_text())
        assert len(forward)==56 and all(x['finite'] and x['nrmse']<=.003 and x['cosine']>=.99999 for x in forward)
        checks[v]=dict(FP32=json.loads((RESULT/v/'invariance.json').read_text()),forward_checks=len(forward),max_forward_nrmse=max(x['nrmse'] for x in forward),frozen_file_count=len(json.loads((RESULT/v/'frozen_files.json').read_text())),determinism=q['determinism'][v],execution=json.loads((RESULT/(v+'_execution_identity.json')).read_text()))
    write_json(RESULT/'scale_selection_summary.json',selection)
    body='# EXP-0224: post-GPTQ output-aware per-row scales\n\n'
    body+='Only the approved first direction: original-coordinate A and fresh gamma-fold/fixed H2048 R1/H128 R2 C. Controls are frozen EXP0221 A and EXP0223 C. All196transformer projections per candidate regenerated from verified Qwen3-origin. Head, embedding and all norms byte-identical to the corresponding control. No LPBQ/group scales, learned rotations, calibration expansion, runtime or other-recipe changes.\n\n'
    body+='## Fixed method\n\nThree candidates per output row: absmax/7, midpoint between absmax and EXP0223 L2.4 clipping scale, and that clipping scale. Each runs complete GPTQ before selection. Score=sum over all8192calibration positions of squared linear output error using FP16(codes*scale).float() minus original FP32 weights; FP32 GEMM in512token chunks, FP64 sum. Undamped actual-input error, not RTN or GPTQ damped proxy. Ties choose earlier/larger range. Per-row choices, all three scores/scales and selected scores retained.\n\n'
    body+='Frozen64x128 bilingual EXP0221 training calibration, same act-order,1% damping,FP64 factor/FP32 error compensation,block128 computational only. True sequential QKV->O->Gate/Up->Down and layer ordering uses the newly quantized predecessors. Included endpoints guarantee nonincreasing selected local score on these identical current inputs; this does not guarantee whole-model quality or imply that old/new projection inputs are identical. No evaluation/holdout selection or tuning.\n\n'
    body+='## Actual DSP quality\n\nFrozen qbh-lite-v1:512conditional targets,24strict tasks,4open prefixes; a lightweight diagnostic, not general quality certification. Software results remain separate.\n\n'+(RESULT/'quality_table.md').read_text()
    body+='\nPredeclared lower-NLL AND more-task effectiveness: '+json.dumps(q['effectiveness'])+'. Aggregate pass iff any effectiveness true. Candidate C versus candidate A both-metric diagnostic: '+str(q['C_incremental_rotation'])+'. No promotion.\n'
    body+='\n## Scale selection\n\n'+table(['Variant','Rows','Absmax','Midpoint','L2.4 clip','Mean range/absmax'],rows)
    body+='\nSummed projection SSEs are unnormalized local sums from different projection inputs/shapes, retained for diagnostics rather than whole-model comparison. All selected row scores no worse than both endpoints on identical inputs.\n\n```json\n'+json.dumps(selection,indent=2)+'\n```\n'
    body+='\n## Checks and provenance\n\nIndependent NumPy full-input scoring and choices, explicit-scale dense GPTQ parity, original absmax parity, zero-row ties and packing pass.392projection packing and112staged/HF forward checks pass. FP32 transformed equivalence, independent16speed tokens and all quick/full/repeat consistency pass.\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n'
    body+='\nSoftware diagnostic:\n\n```json\n'+json.dumps(q['software_diagnostic'],indent=2)+'\n```\n'
    body+='\n## Complete profiling\n\nA0/A/C0/C rotating four-way1warmup,5short,10formal;640formal invocation and17920layer ledgers. M64 prefill plus15feedback decodes, scoring off; direct complete28-layer Host wall includes final norm/head/greedy/persistent KV, excludes cold staging and separately logged WSL tokenizer/detokenizer. Fixed16tokens continue after EOS if encountered and are diagnostic throughput. No partial-model extrapolation.\n\nFrozen EXP0218 source d981072513d06ed61731c14743c76ac6bc81617f ABI108, embedded218 intentionally retained, outer224. Exact8MiB VTCM,zero timed intermediate hidden/logits DDR/spill,one FastRPC/one HMX owner,no QNN. Offline model work is not DSP timing. Scaffolds named layer14 execute all28layers; inherited replay references remain historical and unused.\n\n'
    body+='Overview candidate chosen by lowest new DSP NLL, ties more tasks then alphabetical; report selection does not promote. F16/U8 are frozen EXP0218 nonpaired references with different weights; no activation-only attribution. Full report retains every repeat1/repeat10 numeric control/candidate field, exclusive additive ledgers and overlapping engine/wait counters.\n\n'+(RESULT/'module_table.md').read_text()
    body+='\n## Direct E2E\n\n```json\n'+json.dumps(dict(times=speed['times'],paired_speed_percent=speed['paired_speed_percent']),indent=2)+'\n```\n'
    body+='\n## Retained artifacts and reproduction\n\nModels D:/llm_exp/models/qwen3-block-htp/exp0224; results D:/llm_exp/results/qwen3-block-htp/exp0224. experiment_exp0224.py A|C; measure_exp0224.py deploy/quick/full/repeat and warmup/short/formal; summarize_exp0224.py; close_exp0224.py report then commit/sync/archive. Fresh paths and registered experiment required.392row-search NPZs,56hidden checkpoints, source archive, runtime binaries, manifests and all logs bound by closure and evidence ledgers. No overwrite or automatic further optimization.\n'
    (RESULT/'REPORT.md').write_text(body);(SOURCE/'docs/EXP0224_OUTPUT_SCALE.md').write_text(body)
    with (RESULT/'full_profiling_report.md').open('a') as f:f.write('\n\n## Experiment context and checks\n\n'+body)


def archive():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    q=json.loads((RESULT/'quality_summary.json').read_text());speed=json.loads((RESULT/'speed_summary.json').read_text());checks={}
    for phase,count in [('warmup',1),('short',5),('formal',10)]:
        inv=0
        for v in ALL:
            measure.ROOT=RESULT/v;paths=sorted((RESULT/phase).glob(f'round_*_{v}.jsonl'));assert len(paths)==count
            for p in paths:inv+=len(measure.speed_validate(p,'w4f16')[0])
        checks[phase]=dict(sessions_per_variant=count,invocation_ledgers=inv,layer_ledgers=inv*28)
    for phase in ['quick','full','repeat']:
        for v in VARIANTS:ev.parse_device(RESULT/phase/f'{v}.jsonl',phase)
    origin=verify_origin();cache={}
    def digest(p):
        st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=ev.digest(p)
        return cache[key]
    a=OUTPUT/'artifacts'/head;a.mkdir(parents=True,exist_ok=False);packages={}
    for v in ALL:
        root=package_path(v);manifest=json.loads((root/'manifest.json').read_text())
        for name,value in manifest['files'].items():assert digest(root/name)==value['sha256'],(v,name)
        packages[v]=dict(path=str(root),manifest_sha256=digest(root/'manifest.json'),files=len(manifest['files']))
        shutil.copy2(root/'manifest.json',a/(v+'_manifest.json'));print('PACKAGE_VERIFIED',v,flush=True)
    intermediates={}
    for directory in ['clip_stats','checkpoints']:
        files=sorted(p for p in (OUTPUT/directory).rglob('*') if p.is_file())
        assert len(files)==(392 if directory=='clip_stats' else 56)
        intermediates.update({str(p):digest(p) for p in files})
    for v in VARIANTS:
        for value in json.loads((RESULT/v/'weight_stats.json').read_text()).values():
            assert intermediates[value['clip_stats_path']]==value['clip_stats_sha256']
        frozen=json.loads((RESULT/v/'frozen_files.json').read_text())
        for n,h in frozen.items():assert digest(OUTPUT/v/n)==digest(control_package(v)/n)==h
    write_json(RESULT/'intermediate_sha256.json',intermediates)
    subprocess.run(['git','archive','--format=tar.gz','-o',str(a/'source.tar.gz'),head],cwd=SOURCE,check=True)
    binaries=json.loads((ev.ROOT/'closure.json').read_text())['speed_runtime']['binaries']
    for n,h in binaries.items():
        p=ev.ROOT/'speed_runtime'/n;assert digest(p)==h;shutil.copy2(p,a/n)
    shutil.copy2(RESULT/'REPORT.md',a/'REPORT.md')
    dependencies=[PREVIOUS/n for n in ['calibration.json','quality_summary.json','closure.json','evidence_sha256.json']]+[PREVIOUS_MODELS/'calibration_ids_u32.bin']+[CLIPPED/n for n in ['quality_summary.json','closure.json','evidence_sha256.json']]+[ev.ROOT/n for n in ['dataset_v1.json','teacher_bf16.json','speed_summary.json','original_checkpoint_sha256.json','closure.json']]
    write_json(RESULT/'closure.json',dict(experiment='EXP-0224',source_head=head,source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip(),
        execution_state='completed',evidence_validity='valid',effectiveness=q['effectiveness'],C_incremental_rotation=q['C_incremental_rotation'],
        quality=q['summary'],software_diagnostic=q['software_diagnostic'],quality_determinism=q['determinism'],speed=speed['times'],paired_speed_percent=speed['paired_speed_percent'],
        scale_selection=json.loads((RESULT/'scale_selection_summary.json').read_text()),profile_candidate=q['profile_candidate'],checks=checks,
        original_checkpoint_only=True,original_shards=origin,lm_head_embedding_norms='byte_identical_A_EXP0221_A_C_EXP0223_C',other_recipes='frozen',lpbq=False,holdout_scored=False,selected_baseline_changed=False,
        physical_gate='pass_exact8MiB_zero_timed_intermediate_DDR_spill_one_FastRPC_no_QNN',runtime_source='d981072513d06ed61731c14743c76ac6bc81617f',runtime_abi=108,runtime_binaries=binaries,
        packages=packages,artifacts=str(a),intermediate_files=len(intermediates),frozen_dependencies={str(p):digest(p) for p in dependencies}))
    write_json(RESULT/'artifacts_sha256.json',{str(p):digest(p) for p in a.iterdir() if p.is_file()})
    files=sorted(p for p in RESULT.rglob('*') if p.is_file() and p.name!='evidence_sha256.json')
    write_json(RESULT/'evidence_sha256.json',{str(p.relative_to(RESULT)):digest(p) for p in files})
    print(json.dumps(dict(head=head,evidence_files=len(files),ledger_sha256=digest(RESULT/'evidence_sha256.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','archive']);a=p.parse_args()
    report() if a.phase=='report' else archive()
