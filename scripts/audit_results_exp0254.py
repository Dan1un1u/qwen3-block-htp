#!/usr/bin/env python3
import json,math,subprocess
from pathlib import Path
import data_exp0254 as data
R=data.RESULT

def read(n):return json.loads((R/n).read_text())
def main():
 data.preflight();data.frozen();f=read('freeze.json');inputs=read('inputs.json');ds=read('dataset.json');z=read('summary.json');route=read('route.json')
 for n,h in f['files'].items():assert data.sha(R/n)==h
 for n,h in f['references'].items():assert data.sha(n)==h
 parent=json.loads(data.verified('exp0253','inputs.json').read_text());assert inputs['parameters']==parent['parameters'] and inputs['development']==parent['development'] and inputs['prefix']==[151645]
 assert read('independent_data_audit.json')['pass_all'] and read('independent_data_audit.json')['documents']==256
 ids={p:{x['id'] for x in ds['samples'] if x['split']==p} for p in ['triage','confirmation']};assert len(ids['triage'])==len(ids['confirmation'])==128 and not ids['triage']&ids['confirmation']
 assert read('checks/numerical.json')['pass_all'] and len(read('checks/numerical.json')['new_FP16_cases'])==12
 assert route['freeze_sha256']==data.sha(R/'freeze.json') and route['frozen_before_confirmation']
 for n,h in route['triage_sha256'].items():assert data.sha(R/f'scores/triage_{n}.json')==h
 c=read('scores/triage_C64.json');j=read('scores/triage_J.json');close=j['ppl']/c['ppl']<=1.05 and all(math.exp(j['cell_nll'][k]-c['cell_nll'][k])<=1.1 for k in data.CELLS)
 assert route['branch']==('attention' if close else 'downstream') and route['close_point']==close
 expected=inputs['branches'][route['branch']];assert route['names']==expected
 assert {p.stem.removeprefix('confirmation_') for p in (R/'scores').glob('confirmation_*.json')}==set(expected)
 evaluated=[];heads=set();phase_expected={'development':inputs['base_names'],'triage':inputs['base_names'],'branch_development':[n for n in expected if n not in inputs['base_names']],'confirmation':expected}
 for phase,names in phase_expected.items():
  done=read(phase+'_complete.json');assert done['pass_all'] and done['names']==names
  panel=inputs['development'] if 'development' in phase else [x for x in ds['samples'] if x['split']==phase]
  for name in names:
   path=f'scores/{phase}_{name}.json';d=read(path);checks=d['checks'];mask=set(inputs['masks'][name]);assert d['mask']==sorted(mask)
   assert d['freeze_sha256']==data.sha(R/'freeze.json') and [(x['id'],x['cell']) for x in d['samples']]==[(x['id'],x['cell']) for x in panel]
   if phase in ['branch_development','confirmation']:assert d['route_sha256']==data.sha(R/'route.json')
   cells={k:math.fsum(math.fsum(x['nll']) for x in d['samples'] if x['cell']==k)/(16*sum(x['cell']==k for x in panel)) for k in data.CELLS}
   mean=math.fsum(cells.values())/4;assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-9
   assert all(len(x['nll'])==len(x['top1'])==16 and all(math.isfinite(n) for n in x['nll']) for x in d['samples'])
   assert all(checks[k] for k in ['repeat_exact','causal_exact','cache_oracle_exact','hook_sentinel_exact','prefix_immutable']) and checks['CE_max_abs']<5e-6
   if name not in ['F','C64']:
    steps=len(panel)//4*16;calls=28*steps
    assert checks['rotation_counts']==dict(prefix_K=28*len(panel)//4,body_K=calls,Q=calls)
    expected_counts={'core':calls}
    for site,quant,restored in [('q_rope','q_qdq','q_restored'),('k_cache','k_cache=qdq','k_cache=restored'),('v_out','v_out=qdq','v_out=restored')]:expected_counts[restored if 'L00.'+site in mask else quant]=calls
    assert checks['boundary_counts']==expected_counts
    hooks={f'L{i:02d}.{site}'+('=restored' if f'L{i:02d}.{site}' in mask else '=qdq'):steps for i in range(28) for site in ['q_out','k_out','v_out','attn_context','residual_mid','residual_out','swiglu']}
    assert checks['hook_counts']==hooks
   if phase=='development' and name in ['F','C64','B']:
    oldname='R3_float_core' if name=='B' else name;p=data.verified('exp0253',f'scores/development_{oldname}.json');old=json.loads(p.read_text());assert d['samples']==old['samples'] and d['ppl']==old['ppl']
   heads.add(d['source_head']);subprocess.run(['git','merge-base','--is-ancestor',d['source_head'],'HEAD'],cwd=data.SOURCE,check=True);evaluated.append(path)
  for recipe in ['F','C64']:
   if not any((n=='F')==(recipe=='F') for n in names):continue
   w=read(f'checks/{phase}_{recipe}_weights.json');old=json.loads(data.verified('exp0253',f'checks/development_{recipe}_weights.json').read_text())
   assert w['unchanged'] and w['digest']==old['digest'] and w['manifest_sha256']==old['manifest_sha256']
 assert heads==set(z['inference_source_heads'])
 for phase in ['triage','confirmation']:
  st=z[phase]
  for n,g in st['quality_gate'].items():
   d=read(f'scores/{phase}_{n}.json');ref=read(f'scores/{phase}_F.json')
   gate=d['ppl']/ref['ppl']<=1.05 and all(math.exp(d['cell_nll'][k]-ref['cell_nll'][k])<=1.1 for k in data.CELLS);assert gate==g['point_pass']
  # Recompute reported interactions directly from independently summed NLLs.
  for value in st['interactions_positive_extra_joint_benefit'].values():
   expected_value=math.fsum(w*read(f'scores/{phase}_{n}.json')['mean_nll'] for n,w in value['weights'].items());assert abs(expected_value-value['mean_nll'])<1e-12
 assert z['device_runs']==0 and z['device_ppl'] is None and z['e2e_tokens_per_second'] is None
 data.write('independent_integrity_checks.json',dict(pass_all=True,evaluated_scores=len(evaluated),route_recomputed=True,heldout_confirmation=True,model_and_parameters_unchanged=True,hooks_and_cache_oracles=True,all_scores_NLL_PPL_reconstructed=True,all_interactions_reconstructed=True,quality_gates_recomputed=True,development_reproductions=['F','C64','B'],source_heads=sorted(heads),files={n:data.sha(R/n) for n in evaluated}))
 print('INDEPENDENT_INTEGRITY_PASS',len(evaluated),flush=True)
if __name__=='__main__':main()
