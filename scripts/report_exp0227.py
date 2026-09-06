#!/usr/bin/env python3
"""Select presentation by independent validation, then close full evidence."""
import argparse,json,math,subprocess
from pathlib import Path
import torch
from block_reconstruction_exp0227 import RESULT,OUTPUT,CONTROLS,PLAN,ROOT,digest,preflight
from rotation_exp0219 import write_json


def select():
    assert digest(RESULT.parent/'exp0226/closure.json')=='21db2339632d67b76778ed649f01499c165cecab085c7fa2574df8fb9786e724'
    old=json.loads((RESULT.parent/'exp0226/validation_summary.json').read_text())
    scores={v:json.loads((RESULT/v/'validation.json').read_text()) for v in ['A','R']}
    assert all(len(s['samples'])==64 and s['dataset_sha256']==digest(RESULT/'learning_data.json') for s in scores.values())
    chosen=min(scores,key=lambda v:(scores[v]['nll'],v))
    write_json(RESULT/'selection.json',dict(selected=chosen,rule='presentation_only_minimum_actual_independent_validation_NLL_A_R_tie_A',
        scores={v:dict(nll=s['nll'],ppl=math.exp(s['nll']),language_nll=s['language_nll']) for v,s in scores.items()},
        controls={v:old[k] for v,k in [('A0','control_A'),('R0','old100')]},evaluation_used=False,
        both_candidates_evaluated=True))
    print('PRESENTATION_SELECTED',chosen,flush=True)


def report():
    sel=json.loads((RESULT/'selection.json').read_text());quality=json.loads((RESULT/'quality_summary.json').read_text())
    speed=json.loads((RESULT/'speed_summary.json').read_text());records={}
    lines=['# EXP0227 fixed-rotation final-format block reconstruction',
        'A0=EXP0224 A; R0=EXP0225 step100. A/R learn only transformer row scales; GPTQ codes, rotations, norms, embedding and head stay fixed. Frozen8192 calibration and independent64-window Wiki/news validation; qbh/holdout never selects parameters. This is an OmniQuant-inspired scale-only block reconstruction adaptation, not full OmniQuant.',
        '## Independent actual-package validation',
        '| Variant | NLL | PPL | English NLL | Chinese NLL |','|---|---:|---:|---:|---:|']
    for v,s in {**sel['controls'],**sel['scores']}.items():
        lines.append(f"| {v} | {s['nll']:.6f} | {s['ppl']:.4f} | {s['language_nll']['en']:.6f} | {s['language_nll']['zh']:.6f} |")
    lines += ['## Frozen DSP qbh-lite-v1', (RESULT/'quality_table.md').read_text(),
        'These are512 conditional targets and24 strict short tasks, not a broad capability certification. Software metrics are retained separately. PPL is conditional on the fixed short benchmark. Both content and format failures count under unchanged scoring.',
        '## Reconstruction diagnostics', '| Coordinate | selected0/50/100 blocks | final train relative MSE | final validation relative MSE | elapsed seconds |', '|---|---|---:|---:|---:|']
    for v in ['A','R']:
        r=json.loads((RESULT/'training'/v/'complete.json').read_text());records[v]=r
        counts=[r['selected_steps'].count(i) for i in [0,50,100]]
        lines.append(f"| {v} | {counts} | {r['final_train_relative_mse']:.8g} | {r['final_validation_relative_mse']:.8g} | {r['elapsed_s']:.2f} |")
    lines += ['Each local selection compares checkpoints on the same incoming student stream at that block. Improvements are not additive causal contributions. Full-model quality determines whether reconstruction transfers.',
        '## Effectiveness', '```json',json.dumps(dict(per_coordinate=quality['effectiveness'],incremental_rotation=quality['incremental_rotation'],baseline_promoted=False),indent=2),'```',
        '## Complete profiling', (RESULT/'module_table.md').read_text(),
        'The F16A16 and W4A8 columns are frozen nonpaired EXP0218 references. Current A0/R0/A/R use1warmup5short10four-way rotating formal rounds with identical ABI108 runtime. Full additive and overlapping counters are retained in full_profiling_report.md.',
        '## Direct E2E throughput','```json',json.dumps(speed['times'],indent=2),'```']
    (RESULT/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    write_json(RESULT/'reconstruction_summary.json',records)


def close():
    assert not (RESULT/'closure.json').exists()
    quality=json.loads((RESULT/'quality_summary.json').read_text());speed=json.loads((RESULT/'speed_summary.json').read_text())
    assert speed['formal_invocations']==640 and speed['formal_layer_ledgers']==17920
    assert len(list((RESULT/'short').glob('round_*.jsonl')))==20
    assert len(list((RESULT/'formal').glob('round_*.jsonl')))==40
    assert json.loads((RESULT/'unit_oracle.json').read_text())['passed']
    controls=json.loads((RESULT/'device_controls_verified.json').read_text())
    assert all(r['verified_files']==1276 for r in controls['controls'].values())
    packages={};blocks={}
    for v in ['A','R']:
        root=OUTPUT/v;manifest=json.loads((root/'manifest.json').read_text())
        for n,r in manifest['files'].items():assert digest(root/n)==r['sha256'],n
        assert len(manifest['changed_files'])==197
        checks=json.loads((RESULT/v/'export_oracle.json').read_text())
        assert checks['passed'] and len(checks['projections'])==196
        assert all(r['train_export_weight_exact'] and r['codes_exact'] for r in checks['projections'])
        done=json.loads((RESULT/'training'/v/'complete.json').read_text())
        assert done['layers']==28 and done['trainable_parameters']==573440 and not done['smoke']
        for i in range(28):
            r=json.loads((RESULT/'training'/v/f'layer{i:02d}_selection.json').read_text())
            assert [x['step'] for x in r['candidates']]==[0,50,100]
            assert r['selected_step']==min(r['candidates'],key=lambda x:(x['validation_relative_mse'],x['step']))['step']
            assert digest(Path(r['selected_checkpoint']))==r['selected_sha256']
            assert not r['qbh_used']
            assert all(c['zero_step_exact'] and c['finite'] and c['nrmse']<=.003 and c['cosine']>=.99999 for c in r['checks'])
        packages[v]=dict(path=str(root),manifest_sha256=digest(root/'manifest.json'),files=len(manifest['files']))
        blocks[v]=done
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff','--name-only','b90d9a7309fe470fb7dc7b8de83b483621ad481c',head,'--','src','include','CMakeLists.txt'],cwd=ROOT,text=True).strip()
    archives={str(p.relative_to(OUTPUT)):digest(p) for folder in ['training','smoke','artifacts'] for p in sorted((OUTPUT/folder).rglob('*')) if p.is_file()}
    write_json(RESULT/'intermediate_sha256.json',archives)
    ledger={str(p.relative_to(RESULT)):digest(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name not in ['evidence_sha256.json','closure.json']}
    write_json(RESULT/'evidence_sha256.json',ledger)
    write_json(RESULT/'closure.json',dict(experiment='EXP-0227',source_head=head,execution_state='completed',evidence_validity='valid',
        local_gate='pass' if any(quality['effectiveness'].values()) else 'fail',adoption_status='pending',
        effectiveness=quality['effectiveness'],incremental_rotation=quality['incremental_rotation'],
        quality=quality['summary'],software_diagnostic=quality['software_diagnostic'],speed=speed['times'],
        validation=json.loads((RESULT/'selection.json').read_text()),reconstruction=blocks,packages=packages,
        evidence_files=len(ledger),evidence_ledger_sha256=digest(RESULT/'evidence_sha256.json'),intermediate_files=len(archives),
        protocol_sha256=digest(RESULT/'protocol.json'),full_profiling_report_sha256=digest(RESULT/'full_profiling_report.md'),
        results_report_sha256=digest(RESULT/'REPORT.md'),implementation_gate='pass',physical_gate='pass',
        other_recipes='frozen',runtime_unchanged=True,selected_baseline_changed=False,holdout_scored=False))
    print((RESULT/'closure.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','report','close']);a=p.parse_args();preflight()
    if a.phase=='select':select()
    elif a.phase=='report':report()
    else:close()
