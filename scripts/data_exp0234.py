#!/usr/bin/env python3
"""References for the predeclared PC052 paired panel and64K group calibration."""
import json,shutil
from pathlib import Path
from data_exp0229 import BASE,SOURCE,MEMORY,MODEL,CELLS,sha,verified,preflight
RESULT=BASE/'exp0234'
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0234')
def write(n,x):
 p=RESULT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def prepare():
 preflight();assert not (RESULT/'dataset_freeze.json').exists();RESULT.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
 panel=verified('exp0233','dataset.json');assert json.loads(verified('exp0233','independent_data_audit.json').read_text())['pass_all']
 with (RESULT/'dataset.json').open('xb') as f:f.write(panel.read_bytes())
 refs=[panel,verified('exp0233','dataset_freeze.json'),verified('exp0233','independent_data_audit.json'),verified('exp0233','recovery/dataset_metadata_erratum.json'),verified('exp0230','dataset.json'),verified('exp0230','inputs/C64_calibration_u32.bin'),verified('exp0231','G128/package.json')]
 write('dataset_freeze.json',dict(files={'dataset.json':sha(RESULT/'dataset.json')},references={str(p):sha(p) for p in refs},protocol_sha256=sha(MEMORY/'docs/experiments/EXP-0234.md'),frozen_before_export=True,
  evaluation_role='disclosed_shared_PC052_panel_frozen_before_all_three_phases',calibration_tokens=65536,calibration_context=128,unused_metadata_erratum_reference=str(refs[3])))
 print('EXP234_INPUTS_FROZEN',sha(RESULT/'dataset_freeze.json'),flush=True)
def frozen():
 f=json.loads((RESULT/'dataset_freeze.json').read_text());assert sha(MEMORY/'docs/experiments/EXP-0234.md')==f['protocol_sha256']
 for n,h in f['files'].items():assert sha(RESULT/n)==h
 for n,h in f['references'].items():assert sha(n)==h,n
 return json.loads((RESULT/'dataset.json').read_text())
if __name__=='__main__':prepare()
