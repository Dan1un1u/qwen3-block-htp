#!/usr/bin/env python3
"""Archive completed learned-rotation evidence; no automatic baseline promotion."""
import hashlib,json,subprocess,math
from pathlib import Path
from learned_rotation_exp0225 import RESULT,OUTPUT,PLAN
from rotation_exp0219 import write_json
from prepare_exp0164_generation_package import sha256_file

def close():
    assert not (RESULT/'closure.json').exists()
    selection=json.loads((RESULT/'selection.json').read_text());selected=selection['selected']
    assert selection['evaluation_used'] is False and selection['rule']==PLAN['checkpoint_selection']
    assert selected==min(selection['scores'],key=lambda k:(selection['scores'][k]['nll'],int(k[4:])))
    identity=json.loads((RESULT/'initial_rotation_identity.json').read_text())
    assert identity['R1_exact'] and identity['R2_exact']
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
    assert speed['formal_invocations']==320 and speed['formal_layer_ledgers']==8960
    assert len(list((RESULT/'short').glob('round_*.jsonl')))==10
    assert len(list((RESULT/'formal').glob('round_*.jsonl')))==20
    train=json.loads((RESULT/'training_exact_complete.json').read_text());assert train['steps']==100
    packages={}
    for step in PLAN['checkpoints']:
        v=f'step{step:03d}';manifest=OUTPUT/v/'manifest.json';record=json.loads(manifest.read_text())
        for n,r in record['files'].items():assert sha256_file(OUTPUT/v/n.replace(chr(92),'/'))==r['sha256'],n
        checkpoint=OUTPUT/('training' if v=='step000' else 'training_exact')/f'{v}.pt'
        assert record['rotation_checkpoint_sha256']==sha256_file(checkpoint)
        validation[v]=json.loads((RESULT/v/'validation.json').read_text())
        assert validation[v]['dataset_sha256']==data_sha
        assert validation[v]['nll']==selection['scores'][v]['nll']
        assert len(validation[v]['samples'])==32 and all(r['tokens']==127 for r in validation[v]['samples'])
        invariance=json.loads((RESULT/v/'invariance.json').read_text())['checks']
        assert len(invariance)==2 and all(c['passed'] for c in invariance)
        forward=json.loads((RESULT/v/'calibration_forward_checks.json').read_text())
        assert len(forward)==56 and all(c['finite'] and c['nrmse']<=.003 and c['cosine']>=.99999 for c in forward)
        stats=json.loads((RESULT/v/'weight_stats.json').read_text())
        assert len(stats)==197 and all(c['packed_roundtrip'] for c in stats.values())

        packages[v]=dict(manifest_sha256=sha256_file(manifest),rotation_checkpoint_sha256=sha256_file(OUTPUT/('training' if v=='step000' else 'training_exact')/f'{v}.pt'))
    source=Path(__file__).resolve().parents[1];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    runtime_changes=subprocess.check_output(['git','diff','--name-only','cf8a239a636bb33dd9bf49c7aa03760df9af6c81',head,'--','src','include','CMakeLists.txt'],cwd=source,text=True).strip();assert not runtime_changes
    control=json.loads((RESULT/'control_A/validation.json').read_text());assert control['dataset_sha256']==data_sha
    validation_summary={k:dict(nll=v['nll'],ppl=math.exp(v['nll']),language_nll=v['language_nll']) for k,v in {'control_A':control,**validation}.items()}
    write_json(RESULT/'validation_summary.json',validation_summary)
    intermediates={str(p.relative_to(OUTPUT)):sha256_file(p) for folder in ['training','training_exact','checkpoints','clip_stats'] for p in sorted((OUTPUT/folder).rglob('*')) if p.is_file()}
    write_json(RESULT/'intermediate_sha256.json',intermediates)
    ledger={str(p.relative_to(RESULT)):sha256_file(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name not in ['evidence_sha256.json','closure.json']}
    write_json(RESULT/'evidence_sha256.json',ledger)
    record=dict(experiment='EXP-0225',source_head=head,execution_state='completed',evidence_validity='valid',
        implementation_gate='pass',quality_determinism='pass',effectiveness=quality['incremental_rotation'],
        selected_checkpoint=selected,packages=packages,quality=quality['summary'],speed=speed['times'],
        evidence_files=len(ledger),evidence_ledger_sha256=sha256_file(RESULT/'evidence_sha256.json'),
        intermediate_files=len(intermediates),other_recipes='frozen',runtime_unchanged=True,selected_baseline_changed=False,
        validation=validation_summary,training_final_orthogonality=train_rows[-1]['orthogonality'],training_data_sha256=train['data_sha256'],full_profiling_report_sha256=sha256_file(RESULT/'full_profiling_report.md'))
    write_json(RESULT/'closure.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':close()
