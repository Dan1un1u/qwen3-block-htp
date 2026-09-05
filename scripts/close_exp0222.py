#!/usr/bin/env python3
"""Summarize head-only effects and archive the committed host diagnostic."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
from statistics import mean
import torch
import transformers
import numpy as np
import eval_exp0218 as ev
from experiment_exp0222 import RESULT, OUTPUT, PREVIOUS, SOURCE
from rotation_exp0219 import write_json
from summarize_exp0218 import table


def report():
    assert not (RESULT/'summary.json').exists()
    results = {}; rows = []; task_changes = {}; checks = {}
    data = {s['id']:s for s in ev.dataset()['samples']}
    for v in ['A','B','C']:
        results[v] = json.loads((RESULT/v/'summary.json').read_text())
        a,b = (results[v][h] for h in ['W4','FP16'])
        results[v]['delta'] = dict(nll=b['nll']-a['nll'],tasks=b['tasks_correct']-a['tasks_correct'],
            teacher_top1_agreement=b['teacher_top1_agreement']-a['teacher_top1_agreement'],
            language_nll={k:b['language_nll'][k]-a['language_nll'][k] for k in ['zh','en']})
        rows.append([v,f"{a['nll']:.6f}",f"{b['nll']:.6f}",f"{b['nll']-a['nll']:+.6f}",
            f"{a['tasks_correct']}/24",f"{b['tasks_correct']}/24",f"{a['teacher_top1_agreement']*100:.2f}%",f"{b['teacher_top1_agreement']*100:.2f}%"])
        old = {s['id']:s for s in json.loads((RESULT/v/'W4_quality.json').read_text())['samples']}
        new = {s['id']:s for s in json.loads((RESULT/v/'FP16_quality.json').read_text())['samples']}
        task_changes[v] = [dict(id=i,language=data[i]['language'],old_correct=s['correct'],new_correct=new[i]['correct'],
            old_text=s['text'],new_text=new[i]['text']) for i,s in old.items()
            if s['kind']=='task' and s['correct']!=new[i]['correct']]
        checks[v] = dict(control=json.loads((RESULT/v/'control_reproduction.json').read_text()),
            repeats={h:json.loads((RESULT/v/(h+'_repeat.json')).read_text())['check'] for h in ['W4','FP16']},
            provenance=json.loads((RESULT/v/'provenance.json').read_text()))
        assert checks[v]['provenance']['non_head_unchanged']
    gaps = {h:{v:results[v][h]['nll']-results['A'][h]['nll'] for v in ['B','C']} for h in ['W4','FP16']}
    environment = dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,threads=16)
    summary = dict(experiment='EXP-0222',results=results,nll_gap_to_A=gaps,task_changes=task_changes,
        execution_state='completed',evidence_validity='valid',local_gate='pass',adoption_status='not_applicable',
        quality_role='host_FP16_software_ablation_not_DSP',device_work=False,profiling='N/A_host_only',
        frozen_other_recipes=True,selected_baseline_changed=False,holdout_scored=False,environment=environment)
    write_json(RESULT/'summary.json',summary)
    heading = table(['Transformer','W4 head NLL','FP16 head NLL','NLL delta','W4 tasks','FP16 tasks','W4 teacher top1','FP16 teacher top1'],rows)
    text = '''# EXP-0222: LM head precision ablation

User-approved host-only diagnostic on frozen EXP-0221 GPTQ A/B/C. Each row compares identical transformer weights, embedding, normalization and arithmetic, changing only LM head precision. A retains original gamma and uses original FP16 head; B retains identity final norm and uses freshly folded W*gamma; C retains rotated coordinates and uses freshly folded/rotated (W*gamma)*H2048. Original checkpoint only; no LPBQ, clipping, calibration, rotation learning or other-recipe changes.

## Paired software quality

These are software-versus-software pairs. They must not be substituted for DSP results. Fixed qbh-lite-v1:512 conditional targets,24 strict tasks,4 open prefixes; eight holdout samples unscored. Conditional scores use the same teacher-forced inputs; greedy samples use each head's own generated feedback. NLL lower is better. This lightweight set is diagnostic, not a broad quality certification.

'''+heading
    text += '\n## Gaps to unrotated A\n\n```json\n'+json.dumps(gaps,indent=2)+'\n```\n'
    text += '\nA gap that persists with FP16 heads cannot be attributed solely to W4 head quantization. Differences in paired head gains identify representation-dependent head sensitivity; these are finite interventions and need not add linearly to independent transformer ablations. No hypothesis about clipping or learned rotation was tested here.\n'
    text += '\n## Task changes\n\n```json\n'+json.dumps(task_changes,ensure_ascii=False,indent=2)+'\n```\n'
    text += '''
## Validation and evidence

All consumed package files are checked against frozen manifest identities from EXP-0221 closure. Original checkpoint shards are rehashed. All non-head parameter hashes match before and after both quality evaluations. W4 full controls reproduce previous software token/top1/text/grades exactly, NLL tolerance1e-5; four fixed bilingual NLL/task cases repeat exactly for both heads. A dense explicit Hadamard head-row oracle checks construction independently of the exporter butterfly. Fresh FP16 head tensors and hashes are retained in models/exp0222/A|B|C. No inherited replay caches are consumed.

'''
    for v in ['A','B','C']:
        p=checks[v]['provenance']
        compact=dict(control=checks[v]['control'],repeats=checks[v]['repeats'],consumed_files=len(p['consumed_files']),
            non_head_unchanged=p['non_head_unchanged'],head_construction=p['head_construction'],head_formula=p['head_formula'],
            head_path=p['fp16_head_path'],head_sha256=p['fp16_head_sha256'],execution_source_head=p['source_head'])
        text+='\n### '+v+'\n\n```json\n'+json.dumps(compact,indent=2)+'\n```\n'
    text+='''
## Profiling boundary

Host-only semantic attribution under PC-042. No DSP runtime integration, device execution, warmup/short/formal measurements or module profiling were performed. All repeat-one/repeat-ten module, engine, physical runtime ledger and complete accelerator Host-wall sections are N/A. E2E token/s is N/A. CPU evaluation elapsed time is retained only for reproducibility and is not accelerator throughput. Frozen EXP-0221 performance is not relabeled as mixed-head performance. No automatic baseline promotion. Discuss these results with the user before choosing another direction.

## Reproduction

scripts/experiment_exp0222.py A|B|C reconstructs frozen packed transformer packages, verifies controls and performs each head swap. scripts/close_exp0222.py report then a committed source archive binds the result. Output paths refuse overwrite; future work needs a newly registered experiment. Source and result copies of this report match. Evidence ledger and closure.json record final source HEAD; provenance.json records actual execution source HEAD per variant.

'''
    text+='\nEnvironment:\n\n```json\n'+json.dumps(environment,indent=2)+'\n```\n'
    (RESULT/'REPORT.md').write_text(text)
    (SOURCE/'docs/EXP0222_LM_HEAD_ABLATION.md').write_text(text)
    print(heading,flush=True)


def archive():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    a=OUTPUT/'artifacts'/head;a.mkdir(parents=True,exist_ok=False)
    subprocess.run(['git','archive','--format=tar.gz','-o',str(a/'source.tar.gz'),head],cwd=SOURCE,check=True)
    shutil.copy2(RESULT/'REPORT.md',a/'REPORT.md')
    heads={}
    for v in ['A','B','C']:
        p=json.loads((RESULT/v/'provenance.json').read_text())
        assert ev.digest(Path(p['fp16_head_path']))==p['fp16_head_sha256']
        heads[v]=dict(path=p['fp16_head_path'],sha256=p['fp16_head_sha256'])
    summary=json.loads((RESULT/'summary.json').read_text())
    write_json(RESULT/'closure.json',dict(source_head=head,source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip(),
        **summary,fp16_heads=heads,artifacts=str(a),source_archive_sha256=ev.digest(a/'source.tar.gz'),
        report_sha256=ev.digest(RESULT/'REPORT.md'),previous_closure_sha256=ev.digest(PREVIOUS/'closure.json'),
        dataset_sha256=ev.digest(ev.ROOT/'dataset_v1.json')))
    files=sorted(p for p in RESULT.rglob('*') if p.is_file() and p.name!='evidence_sha256.json')
    write_json(RESULT/'evidence_sha256.json',{str(p.relative_to(RESULT)):ev.digest(p) for p in files})
    print(json.dumps(dict(head=head,evidence_files=len(files),ledger_sha256=ev.digest(RESULT/'evidence_sha256.json'))))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','archive']);a=p.parse_args()
    report() if a.phase=='report' else archive()
