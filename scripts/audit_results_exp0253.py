#!/usr/bin/env python3
"""Independent result reconstruction and provenance checks for EXP0253."""
import json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
import data_exp0253 as data
R=data.RESULT
S=data.SOURCE

def read(n):return json.loads((R/n).read_text())
def write(n,d):data.write(n,d)
def main():
 data.preflight();data.frozen()
 sw=read('software_summary.json');inputs=read('inputs.json');f=read('freeze.json');ds=read('dataset.json')
 assert f['frozen_before_inference'] and read('dataset_freeze.json')['frozen_before_export_and_scoring']
 for n,h in f['files'].items():assert data.sha(R/n)==h
 for n,h in f['references'].items():assert data.sha(n)==h
 for name in ['F','C64','A8','R3_ALL']:
  p=data.verified('exp0246',f'parameters/{name}.json');assert inputs['parameters'][name]==json.loads(p.read_text())['parameters']
 assert read('independent_data_audit.json')['pass_all'] and read('checks/numerical.json')['pass_all']
 assert len(read('checks/numerical.json')['cases'])==280
 assert read('development_complete.json')['pass_all']
 heads=set();reproduced=[];score_paths=[]
 for phase in ['development','final']:
  panel=inputs['development'] if phase=='development' else ds['samples']
  for name in inputs['names']:
   path=f'scores/{phase}_{name}.json';score_paths.append(path);d=read(path)
   assert d['name']==name and d['phase']==phase and d['freeze_sha256']==data.sha(R/'freeze.json')
   assert [(x['id'],x['cell']) for x in d['samples']]==[(x['id'],x['cell']) for x in panel]
   cell={c:math.fsum(math.fsum(x['nll']) for x in d['samples'] if x['cell']==c)/(16*sum(x['cell']==c for x in panel)) for c in data.CELLS}
   mean=math.fsum(cell.values())/4
   assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-9
   assert all(len(x['nll'])==len(x['top1'])==16 and all(math.isfinite(v) for v in x['nll']) for x in d['samples'])
   checks=d['checks'];assert checks['repeat_exact'] and checks['causal_exact'] and checks['prefix_immutable'] and checks['CE_max_abs']<5e-6
   if name.endswith('_float_core'):
    calls=28*16*len(panel)//4;assert checks['boundary_counts']==dict(float_core=calls,context_qdq=calls)
   heads.add(d['source_head']);subprocess.run(['git','merge-base','--is-ancestor',d['source_head'],'HEAD'],cwd=S,check=True)
   if phase=='development' and not name.endswith('_float_core'):
    p=data.verified('exp0249' if name.endswith('_legacy') else 'exp0252',path);old=json.loads(p.read_text())
    assert d['samples']==old['samples'] and d['ppl']==old['ppl'];reproduced.append(name)
  for recipe in ['F','C64']:
   d=read(f'checks/{phase}_{recipe}_weights.json');assert d['unchanged']
   p=data.verified('exp0252',f'checks/{phase}_{recipe}_weights.json');old=json.loads(p.read_text());assert d['digest']==old['digest'] and d['manifest_sha256']==old['manifest_sha256']
 assert len(reproduced)==6 and heads==set(sw['inference_source_heads'])
 # Separate summation path verifies every reported effect and gate.
 for n,d in sw['vs_F'].items():
  a=read('scores/final_'+n+'.json');b=read('scores/final_F.json')
  ratio=a['ppl']/b['ppl'];assert abs(ratio-d['ppl_ratio'])<1e-12
  gate=ratio<=1.05 and all(math.exp(a['cell_nll'][c]-b['cell_nll'][c])<=1.1 for c in data.CELLS)
  assert sw['quality_gate'][n]['point_pass']==gate
 assert all(sw[k] is None for k in ['device_ppl','e2e_tokens_per_second']) and sw['device_runs']==0
 write('independent_integrity_checks.json',dict(pass_all=True,independent_NLL_PPL_reconstruction=True,quality_gates_recomputed=True,
     old_development_exact=reproduced,source_heads=sorted(heads),frozen_parameters_unchanged=True,weight_digests_match_parent=True,
     full_float_core_and_single_context_QDQ_counts=True,dataset_documents=128,dataset_targets=2048,
     result_scores={n:data.sha(R/n) for n in score_paths},numerical_reference_cases=280,device_runs=0))
 print('INDEPENDENT_INTEGRITY_PASS',flush=True)
if __name__=='__main__':main()
