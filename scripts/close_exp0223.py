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
from clipping_exp0223 import RESULT,OUTPUT,PREVIOUS,PREVIOUS_MODELS,RATIOS
from experiment_exp0223 import VARIANTS,verify_origin
from measure_exp0223 import ALL,package_path
from rotation_exp0219 import write_json
from summarize_exp0218 import table

SOURCE=Path(__file__).resolve().parents[1]


def report():
    assert not (RESULT/'REPORT.md').exists()
    q=json.loads((RESULT/'quality_summary.json').read_text())
    speed=json.loads((RESULT/'speed_summary.json').read_text())
    clip={};checks={};rows=[]
    for v in VARIANTS:
        stats=json.loads((RESULT/v/'weight_stats.json').read_text());assert len(stats)==196
        histogram=np.zeros(80,dtype=np.int64)
        for value in stats.values():histogram+=np.array(value['clip_choice_histogram'])
        clip[v]=dict(rows=int(histogram.sum()),clipped_rows=int(histogram[1:].sum()),
            mean_ratio=float(np.array(RATIOS)@histogram/histogram.sum()),ratio_histogram=histogram.tolist(),
            probes_lower_than_clipped_RTN=sum(x['gptq_probe_nrmse']<x['rtn_probe_nrmse'] for x in stats.values()),
            mean_clipped_RTN_probe_nrmse=mean(x['rtn_probe_nrmse'] for x in stats.values()),
            mean_clipped_GPTQ_probe_nrmse=mean(x['gptq_probe_nrmse'] for x in stats.values()))
        rows.append([v,clip[v]['rows'],f"{100*clip[v]['clipped_rows']/clip[v]['rows']:.2f}%",f"{clip[v]['mean_ratio']:.4f}"])
        forward=json.loads((RESULT/v/'calibration_forward_checks.json').read_text())
        assert len(forward)==56 and all(x['finite'] and x['nrmse']<=.003 and x['cosine']>=.99999 for x in forward)
        checks[v]=dict(FP32=json.loads((RESULT/v/'invariance.json').read_text()),
            forward_checks=len(forward),max_forward_nrmse=max(x['nrmse'] for x in forward),
            frozen_file_count=len(json.loads((RESULT/v/'frozen_files.json').read_text())),
            determinism=q['determinism'][v],execution=json.loads((RESULT/(v+'_execution_identity.json')).read_text()))
    write_json(RESULT/'clipping_summary.json',clip)
    text='''# EXP-0223: transformer per-channel clipping before GPTQ

User-approved bounded scale/clipping trial. A uses original coordinates/gamma, B fresh gamma folding, C fresh fixed H2048 R1/H128 R2. All196 transformer projections per variant are generated afresh from verified Qwen3-origin. Each variant retains byte-identical EXP0221 LM head, embedding and all norm files. No LPBQ/group quantization, new rotation, new calibration, other-recipe change or runtime change.

## Fixed quantization intervention

For each output row of transformed FP32 weights, evaluate80 ratios p=1-i/100 for i=0..79. Scale=p*absmax(row)/7, nearest-even signed[-7,7]. Select the minimum sum(abs(weight-reconstructed_weight)^2.4); ties choose the largest range. This adapts the weight-range search approach in SpinQuant's utils/quant_utils.py and its w_clip evaluation option to our fixed integer grid. It is not a full SpinQuant reproduction. No activation-weighted clipping search or post-GPTQ grid search is performed. Original unclipped range is a candidate, so selected weight-local objective cannot increase; this does not guarantee model quality.

GPTQ then keeps that chosen scale fixed while doing activation-weighted error compensation. Existing true-sequential QKV->O->Gate/Up->Down ordering,1% damping,FP64 factor/FP32 compensation, act-order/inverse export and computational block128 remain. Calibration is the frozen EXP0221 independent64x128 bilingual training tokens (8192 total); no evaluation or holdout tuning. Each later projection/layer sees the newly quantized predecessor. The old head stays frozen even though transformer outputs change, isolating the transformer intervention.

## Actual DSP quality

Frozen qbh-lite-v1:512 conditional targets,24 strict tasks,4 open prefixes. EXP0221 absmax A/B/C DSP quality is a frozen historical control. All new candidates complete actual DSP quick/full/repeat and independent software quality. Software scores are separate and not assumed bit-exact DSP. This lightweight diagnostic does not certify general model quality.

'''+(RESULT/'quality_table.md').read_text()
    text+='\nPredeclared lower-NLL AND more-task effectiveness per variant: '+json.dumps(q['effectiveness'])+'. Incremental clipped C versus clipped A on BOTH metrics: '+str(q['C_incremental_rotation'])+'. No automatic baseline promotion.\n'
    text+='\n## Scale selection\n\n'+table(['Variant','Output rows','Rows clipped','Mean selected range / absmax'],rows)
    text+='\nPer-row choices, scales and before/after objectives are retained as hashed NPZ artifacts. Local probe errors use every16th calibration input (512 positions) on identical candidate inputs, comparing clipped RTN and clipped GPTQ against unquantized transformed weights. They are unweighted projection summaries, not whole-model errors, and are not used to select settings. Input Gram hashes inherited from the GPTQ utility identify diagonals, not full Gram matrices.\n'
    text+='\n## Independent checks and identities\n\nNumPy exhaustive scale selection, dense explicit-scale GPTQ elimination, original absmax parity and integer packing oracles pass. All588projection packing checks and168staged/HF forward checks pass. Nonprojection package files remain byte-identical to each EXP0221 variant.\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n'
    text+='\nSoftware diagnostic:\n\n```json\n'+json.dumps(q['software_diagnostic'],indent=2)+'\n```\n'
    text+='''
## Complete profiling scope

All six variants A0/A/B0/B/C0/C complete one warmup,5short and10formal sessions with rotating order. A0/B0/C0 are the corresponding frozen EXP0221 packages, freshly timed here.960formal invocation ledgers and26880layer ledgers close. All16tokens per speed session match independent per-package references. Scope is one M64 prefill plus15feedback decode steps; quality scoring is off. Host wall is complete28-layer token-in/token-out execution with final norm/head/greedy and persistent KV, excluding cold staging and separately logged WSL tokenizer/detokenizer. No per-layer throughput extrapolation. Fixed speed sequences continue after EOS if present; such throughput describes diagnostic execution.

Runtime remains EXP0218 d981072513d06ed61731c14743c76ac6bc81617f ABI108 (embedded218 intentionally retained, outer experiment223).8MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC/one HMX owner, no QNN. Offline quantization elapsed time is not DSP inference time. Package scaffolds named layer14 execute all28layers. Historical inherited replay references are not valid new replay references and are not consumed in generation/evaluation.

The overview W4A16 column uses the lowest-NLL clipped candidate under the predeclared tie rule. Other recipe columns are frozen EXP0218 nonpaired references, with different W4 weights, so no activation-only attribution. Full report retains every numeric repeat-one/repeat-ten control/candidate field; additive accounting fields are exclusive, engine/wait counters overlap. Detailed raw evidence is retained.

'''+(RESULT/'module_table.md').read_text()
    text+='\n## Direct E2E\n\n```json\n'+json.dumps(dict(times=speed['times'],paired_speed_percent=speed['paired_speed_percent']),indent=2)+'\n```\n'
    text+='''
## Reproduction and retained evidence

Models: D:/llm_exp/models/qwen3-block-htp/exp0223. Results: D:/llm_exp/results/qwen3-block-htp/exp0223. Frozen dataset/calibration/source/package hashes, source archive, runtime binaries, per-row NPZs,84hidden checkpoints, manifests and full logs are bound by closure/evidence ledgers. experiment_exp0223.py A|B|C generates fresh models; measure_exp0223.py deploy/quick/full/repeat and warmup/short/formal collects results; summarize_exp0223.py and close_exp0223.py retain reports. Fresh output paths and a registered experiment are required; do not overwrite or tune retained outputs. No baseline promotion or automatic further direction.

'''
    text+='\nEnvironment:\n\n```json\n'+json.dumps(dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,threads=16),indent=2)+'\n```\n'
    (RESULT/'REPORT.md').write_text(text);(SOURCE/'docs/EXP0223_CHANNEL_CLIPPING.md').write_text(text)
    with (RESULT/'full_profiling_report.md').open('a') as f:f.write('\n\n## Experiment context and checks\n\n'+text)


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
        assert len(files)==(588 if directory=='clip_stats' else 84)
        intermediates.update({str(p):digest(p) for p in files})
    for v in VARIANTS:
        for value in json.loads((RESULT/v/'weight_stats.json').read_text()).values():
            assert intermediates[value['clip_stats_path']]==value['clip_stats_sha256']
        frozen=json.loads((RESULT/v/'frozen_files.json').read_text())
        for n,h in frozen.items():assert digest(OUTPUT/v/n)==digest(PREVIOUS_MODELS/v/n)==h
    write_json(RESULT/'intermediate_sha256.json',intermediates)
    subprocess.run(['git','archive','--format=tar.gz','-o',str(a/'source.tar.gz'),head],cwd=SOURCE,check=True)
    binaries=json.loads((ev.ROOT/'closure.json').read_text())['speed_runtime']['binaries']
    for n,h in binaries.items():
        p=ev.ROOT/'speed_runtime'/n;assert digest(p)==h;shutil.copy2(p,a/n)
    shutil.copy2(RESULT/'REPORT.md',a/'REPORT.md')
    dependencies=[PREVIOUS/n for n in ['calibration.json','quality_summary.json','closure.json','evidence_sha256.json']]+[PREVIOUS_MODELS/'calibration_ids_u32.bin']+[ev.ROOT/n for n in ['dataset_v1.json','teacher_bf16.json','speed_summary.json','original_checkpoint_sha256.json','closure.json']]
    write_json(RESULT/'closure.json',dict(experiment='EXP-0223',source_head=head,source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip(),
        execution_state='completed',evidence_validity='valid',effectiveness=q['effectiveness'],C_incremental_rotation=q['C_incremental_rotation'],
        quality=q['summary'],software_diagnostic=q['software_diagnostic'],quality_determinism=q['determinism'],speed=speed['times'],paired_speed_percent=speed['paired_speed_percent'],
        clipping=json.loads((RESULT/'clipping_summary.json').read_text()),profile_candidate=q['profile_candidate'],checks=checks,
        original_checkpoint_only=True,original_shards=origin,lm_head_embedding_norms='byte_identical_per_variant_EXP0221',other_recipes='frozen',lpbq=False,holdout_scored=False,selected_baseline_changed=False,
        physical_gate='pass_exact8MiB_zero_timed_intermediate_DDR_spill_one_FastRPC_no_QNN',runtime_source='d981072513d06ed61731c14743c76ac6bc81617f',runtime_abi=108,runtime_binaries=binaries,
        packages=packages,artifacts=str(a),intermediate_files=len(intermediates),frozen_dependencies={str(p):digest(p) for p in dependencies}))
    write_json(RESULT/'artifacts_sha256.json',{str(p):digest(p) for p in a.iterdir() if p.is_file()})
    files=sorted(p for p in RESULT.rglob('*') if p.is_file() and p.name!='evidence_sha256.json')
    write_json(RESULT/'evidence_sha256.json',{str(p.relative_to(RESULT)):digest(p) for p in files})
    print(json.dumps(dict(head=head,evidence_files=len(files),ledger_sha256=digest(RESULT/'evidence_sha256.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','archive']);a=p.parse_args()
    report() if a.phase=='report' else archive()
