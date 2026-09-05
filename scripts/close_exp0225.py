#!/usr/bin/env python3
"""Archive completed learned-rotation evidence; no automatic baseline promotion."""
import hashlib,json,subprocess
from pathlib import Path
from learned_rotation_exp0225 import RESULT,OUTPUT,PLAN
from rotation_exp0219 import write_json
from prepare_exp0164_generation_package import sha256_file

def close():
    assert not (RESULT/'closure.json').exists()
    selected=json.loads((RESULT/'selection.json').read_text())['selected']
    quality=json.loads((RESULT/'quality_summary.json').read_text())
    speed=json.loads((RESULT/'speed_summary.json').read_text())
    assert quality['profile_candidate']==speed['profile_candidate']==selected
    assert speed['formal_invocations']==320 and speed['formal_layer_ledgers']==8960
    assert len(list((RESULT/'short').glob('round_*.jsonl')))==10
    assert len(list((RESULT/'formal').glob('round_*.jsonl')))==20
    train=json.loads((RESULT/'training_complete.json').read_text());assert train['steps']==100
    packages={}
    for step in PLAN['checkpoints']:
        v=f'step{step:03d}';manifest=OUTPUT/v/'manifest.json';record=json.loads(manifest.read_text())
        for n,r in record['files'].items():assert sha256_file(OUTPUT/v/n.replace(chr(92),'/'))==r['sha256'],n
        assert (RESULT/v/'validation.json').exists()
        packages[v]=dict(manifest_sha256=sha256_file(manifest),rotation_checkpoint_sha256=sha256_file(OUTPUT/'training'/f'{v}.pt'))
    source=Path(__file__).resolve().parents[1];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    runtime_changes=subprocess.check_output(['git','diff','--name-only','cf8a239a636bb33dd9bf49c7aa03760df9af6c81',head,'--','src','include','CMakeLists.txt'],cwd=source,text=True).strip();assert not runtime_changes
    intermediates={str(p.relative_to(OUTPUT)):sha256_file(p) for folder in ['training','checkpoints','clip_stats'] for p in sorted((OUTPUT/folder).rglob('*')) if p.is_file()}
    write_json(RESULT/'intermediate_sha256.json',intermediates)
    ledger={str(p.relative_to(RESULT)):sha256_file(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name not in ['evidence_sha256.json','closure.json']}
    write_json(RESULT/'evidence_sha256.json',ledger)
    record=dict(experiment='EXP-0225',source_head=head,execution_state='completed',evidence_validity='valid',
        implementation_gate='pass',quality_determinism='pass',effectiveness=quality['incremental_rotation'],
        selected_checkpoint=selected,packages=packages,quality=quality['summary'],speed=speed['times'],
        evidence_files=len(ledger),evidence_ledger_sha256=sha256_file(RESULT/'evidence_sha256.json'),
        intermediate_files=len(intermediates),other_recipes='frozen',runtime_unchanged=True,selected_baseline_changed=False,
        training_data_sha256=train['data_sha256'],full_profiling_report_sha256=sha256_file(RESULT/'full_profiling_report.md'))
    write_json(RESULT/'closure.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':close()
