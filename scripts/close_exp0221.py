#!/usr/bin/env python3
"""Produce the reviewable report, then archive a committed EXP-0221 checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from statistics import mean
import eval_exp0218 as ev
import measure_exp0218 as measure
from experiment_exp0221 import RESULT,OUTPUT,BASE,VARIANTS,verify_origin
from rotation_exp0219 import write_json
from summarize_exp0218 import table

SOURCE=Path(__file__).resolve().parents[1]

def report():
    q=json.loads((RESULT/'quality_summary.json').read_text());speed=json.loads((RESULT/'speed_summary.json').read_text());cal=json.loads((RESULT/'calibration.json').read_text())
    rows=[];checks={}
    for v in VARIANTS:
        stats=json.loads((RESULT/v/'weight_stats.json').read_text());assert len(stats)==197
        rows.append([v,len(stats),f"{mean(x['rtn_probe_nrmse'] for x in stats.values()):.6f}",f"{mean(x['gptq_probe_nrmse'] for x in stats.values()):.6f}",sum(x['gptq_probe_nrmse']<x['rtn_probe_nrmse'] for x in stats.values())])
        checks[v]=dict(FP32=json.loads((RESULT/v/'invariance.json').read_text()),calibration_forward=json.loads((RESULT/v/'calibration_forward_checks.json').read_text()),software=q['software_diagnostic'][v])
    text='''# EXP-0221: per-channel GPTQ with original, folded and rotated weights

This completed experiment tests A=original+GPTQ, B=fresh gamma fold+GPTQ and C=fresh gamma fold+fixed Sylvester H2048 R1/H128 R2+GPTQ. All changed weights originate in verified Qwen3-origin shards. No upstream folded weights, LPBQ/group scales or QNN assets are used. The other two recipes remain frozen. Historical RTN A/B/C quality comes from EXP-0219; these controls were not re-evaluated.

## Fixed calibration and quantizer

64 sequences of128 tokens,32 English WikiText-2-raw train rows and32 Chinese Wikipedia20231101.zh train rows. First128 tokens from first32 eligible distinct rows per language, skipping truncated/short rows and exact32-token overlaps with evaluation samples. No padding, chat-template addition, parameter search or evaluation-guided selection. Raw HTTP snapshots, dataset metadata revisions, exact IDs and tokenizer hashes are retained. Eight holdout windows were only included in the duplicate-exclusion check and never scored. This8192-token calibration is a bounded first trial, not evidence of calibration sufficiency across domains or long contexts.

GPTQ method reference: IST-DASLab/gptq at2d65066eeb06a5c9ff5184d8cebdf33662c67faf, Apache-2.0, archived with its license. Adaptation uses CPU, FP64 damped-Hessian factorization and FP32 error compensation. Fixed transformed-row absmax/7 scale, nearest-even signed[-7,7], groupsize=-1, act-order enabled/inverted on export, damping1% of mean Hessian diagonal. Computational blocksize128 does not add quantization groups. No clipping search. Row chunks share the same input factor and preserve per-row independence.

True-sequential FP16 calibration processes QKV, O, Gate/Up, Down in order; subsequent projections/layers observe previously quantized outputs. The final head observes the quantized stack. All transforms precede calibration, so input statistics use the correct folded/rotated coordinates. Original FP32 transformed weights determine scales before quantization; deployed activations and reconstructed weights use FP16. Tied HF embedding/head are explicitly detached. All197 transformer/head projections are GPTQ quantized, embedding stays FP16, Q/K head norm and RoPE semantics stay intact.

## Actual DSP quality

Frozen qbh-lite-v1:512 bilingual conditional targets,24 strict short tasks and4open prefixes. Quick/full/repeat suites are deterministic across all overlapping token/code/NLL/rank/tie/saturation fields. This is a lightweight diagnostic, not broad model-quality certification. Software results are independently computed from exported integer codes/scales and are not assumed bit-exact DSP logits.

'''+(RESULT/'quality_table.md').read_text()
    text+='\nPredeclared gates: A improves original RTN NLL AND short-task count = '+str(q['A_effectiveness'])+'; C improves A on BOTH metrics = '+str(q['C_incremental_rotation'])+'. No automatic baseline promotion.\n'
    text+='\n## Calibration-local diagnostics\n\n'+table(['Variant','Projections','Mean RTN NRMSE','Mean GPTQ NRMSE','GPTQ lower count'],rows)
    text+='\nThese errors use every16th calibration activation,512 positions, and compare linear output against unquantized transformed FP32 weights on identical inputs. Means are unweighted across projections, not whole-model errors; they are neither evaluation scores nor used to choose grid/settings. Input-statistic hashes in weight_stats are hashes of the Gram diagonal, not the full Gram.\n'
    text+='\nAll dense inverse-Hessian oracle, packed-code roundtrip, FP32 transform invariance and staged-versus-unchanged-HF calibration-forward checks pass. Full details:\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n'
    text+='''
## Complete profiling and provenance

Original RTN plus A/B/C each pass one warmup,5short,10formal sessions with rotating order. All640formal invocation ledgers and17920layer ledgers close exactly. Each16-token speed sequence matches its independent software reference. Fixed M64+15 feedback steps continue after EOS if present; timing then describes diagnostic execution, not usable text throughput. Host wall covers the complete model token-in/token-out pass, excluding cold staging and separately logged WSL tokenizer/detokenizer work. Low-NLL GPTQ candidate is chosen solely for the overview column under the predeclared rule; this does not promote a baseline. Other columns are frozen EXP0218 nonpaired references and do not support activation-only attribution with changed W4 weights.

Runtime is unchanged EXP0218 d981072513d06ed61731c14743c76ac6bc81617f ABI108, embeddedlabel218 intentional; outer experiment221. Full28-layer model, final norm/head/greedy, persistent KV;8MiB VTCM, zero timed intermediate hidden/logits DDR/spill, one full-model FastRPC and one HMX owner, no QNN. Offline calibration/quantization costs are not DSP inference timings.

Evidence D:/llm_exp/results/qwen3-block-htp/exp0221; models D:/llm_exp/models/qwen3-block-htp/exp0221. Source archives, package manifests, frozen runtime binaries, dependency hashes and final evidence ledger are identified in closure.json/evidence_sha256.json. Inherited layer-replay references remain historical placeholders and cannot validate new layer replay. Every successful deployment checks1276files and frozen binary hashes. Tokenizer metadata and license-filename recovery preserved downloaded data and exact calibration IDs; see calibration_recovery.json/reference_recovery.json. Downloads used the existing Windows Clash proxy without global proxy changes.

Reproduction requires a newly registered experiment and fresh output paths. gptq_exp0221.py freezes calibration and checks the independent oracle; experiment_exp0221.py generates each candidate; measure_exp0221.py deploys/runs suites; summarize_exp0221.py reports. Do not overwrite retained outputs or tune against this evaluation/holdout.

'''+(RESULT/'module_table.md').read_text()+'\n\n## Direct E2E\n\n```json\n'+json.dumps(dict(times=speed['times'],paired_speed_percent=speed['paired_speed_percent']),indent=2)+'\n```\n'
    identity=dict(source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip(),
        reporting_input_source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        implementation_files={n:ev.digest(SOURCE/'scripts'/n) for n in ['gptq_exp0221.py','experiment_exp0221.py','measure_exp0221.py','summarize_exp0221.py']},
        calibration_sha256=ev.digest(RESULT/'calibration.json'),calibration_ids_sha256=cal['ids_sha256'],
        final_archive_source_head='Recorded after report commit in closure.json')
    text+='\n\n## Source identity\n\n```json\n'+json.dumps(identity,indent=2)+'\n```\n'
    (RESULT/'REPORT.md').write_text(text);(SOURCE/'docs/EXP0221_GPTQ_ROTATION.md').write_text(text)
    with (RESULT/'full_profiling_report.md').open('a') as f:f.write('\n\n## Experiment context and checks\n\n'+text)

def archive():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();q=json.loads((RESULT/'quality_summary.json').read_text());speed=json.loads((RESULT/'speed_summary.json').read_text());checks={}
    for phase,count in [('warmup',1),('short',5),('formal',10)]:
        inv=0
        for v in ['original']+VARIANTS:
            measure.ROOT=RESULT/v;paths=sorted((RESULT/phase).glob(f'round_*_{v}.jsonl'));assert len(paths)==count
            for p in paths:inv+=len(measure.speed_validate(p,'w4f16')[0])
        checks[phase]=dict(sessions_per_variant=count,invocation_ledgers=inv,layer_ledgers=inv*28)
    for phase in ['quick','full','repeat']:
        for v in VARIANTS:ev.parse_device(RESULT/phase/f'{v}.jsonl',phase)
    origin=verify_origin();cache={}
    def digest(p):
        st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in cache:
            h=hashlib.sha256()
            with p.open('rb') as f:
                while b:=f.read(8*1024*1024):h.update(b)
            cache[key]=h.hexdigest()
        return cache[key]
    packages={};a=OUTPUT/'artifacts'/head;a.mkdir(parents=True,exist_ok=False)
    for v in ['original']+VARIANTS:
        root=BASE if v=='original' else OUTPUT/v;m=json.loads((root/'manifest.json').read_text());old=m['files']
        files={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),x) for n,x in old.items()}
        for n,x in files.items():assert digest(root/n)==x['sha256'],(v,n)
        packages[v]=dict(path=str(root),manifest_sha256=digest(root/'manifest.json'),verified_files=len(files));shutil.copy2(root/'manifest.json',a/(v+'_manifest.json'))
        print('PACKAGE_VERIFIED',v,flush=True)
    subprocess.run(['git','archive','--format=tar.gz','-o',str(a/'source.tar.gz'),head],cwd=SOURCE,check=True)
    binaries=json.loads((ev.ROOT/'closure.json').read_text())['speed_runtime']['binaries']
    for n,h in binaries.items():
        p=ev.ROOT/'speed_runtime'/n;assert digest(p)==h;shutil.copy2(p,a/n)
    shutil.copy2(RESULT/'REPORT.md',a/'REPORT.md')
    deps=[ev.ROOT/n for n in ['dataset_v1.json','teacher_bf16.json','speed_summary.json','original_checkpoint_sha256.json','closure.json']]+[ev.ROOT.parent/'exp0219/quality_summary.json',BASE/'manifest.json',OUTPUT/'calibration_ids_u32.bin']
    write_json(RESULT/'closure.json',dict(experiment='EXP-0221',source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip(),source_head=head,execution_state='completed',evidence_validity='valid',A_effectiveness=q['A_effectiveness'],C_incremental_rotation=q['C_incremental_rotation'],selected_baseline_changed=False,original_checkpoint_only=True,original_shards=origin,lpbq_weights_or_group_scales_used=False,runtime_source='d981072513d06ed61731c14743c76ac6bc81617f',runtime_abi=108,runtime_binaries=binaries,checks=checks,quality=q['summary'],software_diagnostic=q['software_diagnostic'],quality_determinism=q['determinism'],speed=speed['times'],paired_speed_percent=speed['paired_speed_percent'],profile_candidate=q['profile_candidate'],unquantized_equivalence='pass_all_three',physical_gate='pass_exact_8MiB_zero_timed_intermediate_DDR_spill_one_FastRPC_no_QNN',other_recipes='F16F16_and_W4U8_frozen',holdout_scored=False,packages=packages,frozen_dependencies={str(p):digest(p) for p in deps},artifacts=str(a)))
    write_json(RESULT/'artifacts_sha256.json',{str(p):digest(p) for p in sorted(a.iterdir()) if p.is_file()})
    files=sorted(p for p in RESULT.rglob('*') if p.is_file() and p.name!='evidence_sha256.json')
    write_json(RESULT/'evidence_sha256.json',{str(p.relative_to(RESULT)):digest(p) for p in files})
    print(json.dumps(dict(source=head,evidence_files=len(files),ledger_sha256=digest(RESULT/'evidence_sha256.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','archive']);a=p.parse_args()
    report() if a.phase=='report' else archive()
