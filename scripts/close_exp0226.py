#!/usr/bin/env python3
"""Archive completed learned-rotation evidence; no automatic baseline promotion."""
import hashlib,json,subprocess,math
import torch
from pathlib import Path
from learned_rotation_exp0226 import RESULT,OUTPUT,PLAN
from rotation_exp0219 import write_json
from prepare_exp0164_generation_package import sha256_file

def close():
    assert not (RESULT/'closure.json').exists()
    selection=json.loads((RESULT/'selection.json').read_text());selected=selection['selected']
    assert selection['evaluation_used'] is False and selection['rule']==PLAN['checkpoint_selection']
    assert selected==min(selection['scores'],key=lambda k:(selection['scores'][k]['nll'],int(k[4:])))
    assert json.loads((RESULT/'reload_oracle.json').read_text())['passed']
    device_controls=json.loads((RESULT/'device_controls_verified.json').read_text())
    assert all(r['verified_files']==1276 for r in device_controls['controls'].values())
    identity=json.loads((RESULT/'initial_rotation_identity.json').read_text())
    assert identity['R1_exact'] and identity['R2_exact']
    old_ledger_path=RESULT.parent/'exp0225/evidence_sha256.json'
    assert sha256_file(old_ledger_path)=='1d73989c7a6f4d5d8b91db553db32ee22fa023011f125c4ebe29718234ed95d8'
    old_ledger=json.loads(old_ledger_path.read_text())
    for name in ['invariance.json','calibration_forward_checks.json','weight_stats.json','software_generation.json']:
        assert sha256_file(RESULT/'step000'/name)==old_ledger['step000/'+name]
    actual_initial=torch.load(OUTPUT/'training_exact/step000.pt',map_location='cpu',weights_only=False)
    smoke_initial=torch.load(OUTPUT/'smoke_exact/step000.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(actual_initial['rotations'][k],smoke_initial['rotations'][k]) for k in ['R1','R2'])
    data_sha=sha256_file(RESULT/'learning_data.json')
    assert identity['data_sha256']==data_sha
    train_rows=[json.loads(line) for line in (RESULT/'training_exact.jsonl').read_text().splitlines()]
    assert [r['step'] for r in train_rows]==list(range(1,101))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in train_rows)
    assert max(train_rows[-1]['orthogonality'].values())<1e-4
    validation={}

    quality=json.loads((RESULT/'quality_summary.json').read_text())
    speed=json.loads((RESULT/'speed_summary.json').read_text())
    assert quality['profile_candidate']==speed['profile_candidate']==selected
    assert speed['formal_invocations']==480 and speed['formal_layer_ledgers']==13440
    assert len(list((RESULT/'short').glob('round_*.jsonl')))==15
    assert len(list((RESULT/'formal').glob('round_*.jsonl')))==30
    train=json.loads((RESULT/'training_exact_complete.json').read_text());assert train['steps']==100
    packages={}
    for step in PLAN['checkpoints']:
        v=f'step{step:03d}';manifest=OUTPUT/v/'manifest.json';record=json.loads(manifest.read_text())
        for n,r in record['files'].items():assert sha256_file(OUTPUT/v/n.replace(chr(92),'/'))==r['sha256'],n
        checkpoint=(OUTPUT.parent/'exp0225/training/step000.pt') if v=='step000' else OUTPUT/'training_exact'/f'{v}.pt'
        assert record['rotation_checkpoint_sha256']==sha256_file(checkpoint)
        validation[v]=json.loads((RESULT/v/'validation.json').read_text())
        assert validation[v]['dataset_sha256']==data_sha
        assert validation[v]['nll']==selection['scores'][v]['nll']
        assert len(validation[v]['samples'])==64 and all(r['tokens']==127 for r in validation[v]['samples'])
        invariance=json.loads((RESULT/v/'invariance.json').read_text())['checks']
        assert len(invariance)==2 and all(c['passed'] for c in invariance)
        forward=json.loads((RESULT/v/'calibration_forward_checks.json').read_text())
        assert len(forward)==56 and all(c['finite'] and c['nrmse']<=.003 and c['cosine']>=.99999 for c in forward)
        stats=json.loads((RESULT/v/'weight_stats.json').read_text())
        assert len(stats)==197 and all(c['packed_roundtrip'] for c in stats.values())

        packages[v]=dict(manifest_sha256=sha256_file(manifest),rotation_checkpoint_sha256=sha256_file(checkpoint))
    source=Path(__file__).resolve().parents[1];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    runtime_changes=subprocess.check_output(['git','diff','--name-only','cf8a239a636bb33dd9bf49c7aa03760df9af6c81',head,'--','src','include','CMakeLists.txt'],cwd=source,text=True).strip();assert not runtime_changes
    control=json.loads((RESULT/'control_A/validation.json').read_text());assert control['dataset_sha256']==data_sha
    old100=json.loads((RESULT/'old100/validation.json').read_text());assert old100['dataset_sha256']==data_sha
    for variant in ['step000','old100']:
        historical=json.loads((RESULT.parent/'exp0225'/('step100' if variant=='old100' else variant)/'validation.json').read_text())
        current=json.loads((RESULT/variant/'validation.json').read_text())
        assert all(a['nll']==b['nll'] for a,b in zip(current['samples'][:32],historical['samples']))
    validation_summary={k:dict(nll=v['nll'],ppl=math.exp(v['nll']),language_nll=v['language_nll']) for k,v in {'control_A':control,'old100':old100,**validation}.items()}
    write_json(RESULT/'validation_summary.json',validation_summary)
    intermediates={str(p.relative_to(OUTPUT)):sha256_file(p) for folder in ['smoke_exact','training_exact','checkpoints','clip_stats','artifacts'] for p in sorted((OUTPUT/folder).rglob('*')) if p.is_file()}
    write_json(RESULT/'intermediate_sha256.json',intermediates)
    ledger={str(p.relative_to(RESULT)):sha256_file(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name not in ['evidence_sha256.json','closure.json']}
    write_json(RESULT/'evidence_sha256.json',ledger)
    record=dict(experiment='EXP-0226',source_head=head,execution_state='completed',evidence_validity='valid',
        implementation_gate='pass',quality_determinism='pass',effectiveness=quality['incremental_rotation'],
        selected_checkpoint=selected,packages=packages,quality=quality['summary'],speed=speed['times'],
        evidence_files=len(ledger),evidence_ledger_sha256=sha256_file(RESULT/'evidence_sha256.json'),
        intermediate_files=len(intermediates),other_recipes='frozen',runtime_unchanged=True,selected_baseline_changed=False,
        sampling_summary_sha256=sha256_file(RESULT/'sampling_summary.json'),results_report_sha256=sha256_file(RESULT/'REPORT.md'),validation=validation_summary,training_final_orthogonality=train_rows[-1]['orthogonality'],training_data_sha256=train['data_sha256'],full_profiling_report_sha256=sha256_file(RESULT/'full_profiling_report.md'))
    write_json(RESULT/'closure.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':close()
