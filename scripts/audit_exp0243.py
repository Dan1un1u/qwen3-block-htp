from pathlib import Path
import hashlib,json,math
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0243')
def read(p):return json.loads((R/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inp=read('inputs.json');fr=read('inputs_freeze.json');selection=read('selection.json')
assert sha(R/'inputs.json')==fr['inputs_sha256']
assert sha(R/'dataset.json')==fr['final_dataset_sha256']
scored=[]
for p in sorted((R/'scores').glob('*.json')):
 d=json.loads(p.read_text());expected=read('dataset.json')['samples'] if d['phase']=='final' else inp['development']
 assert [(s['id'],s['cell']) for s in d['samples']]==[(s['id'],s['cell']) for s in expected]
 assert all(len(s['nll'])==len(s['top1'])==16 for s in d['samples'])
 mean=sum(sum(s['nll']) for s in d['samples'])/(16*len(d['samples']))
 assert abs(mean-d['mean_nll'])<1e-12 and abs(math.log(d['ppl'])-mean)<1e-12
 assert d['repeat_exact'] and d['causal_exact'] and d['independent_CE_error']<5e-6
 assert d['cache_length']==79+len(d['prefix']['ids'])
 assert d['cache_append_calls']==28*(16 if d['mode']=='sequential' else 1)
 assert d['cache_storage']==('uint8' if d['policy'] else 'float16')
 if d['policy']:assert d['parameters_sha256']==sha(R/f'calibration/{d["prefix"]["name"]}.json')
 scored.append(p.name)
for c in selection['candidates']:
 p=R/f'scores/development_C64_{c["prefix"]["name"]}_{c["policy"]}_bulk.json'
 assert sha(p)==c['score_sha256']
assert selection['final_not_used']
assert selection['selected']['mean_nll']<=min(c['mean_nll'] for c in selection['candidates'])+1e-6
seedfiles=0
for prefix in inp['prefixes']:
 if not prefix['eligible']:continue
 name=prefix['name'];cal=read(f'calibration/{name}.json');paths=read(f'checks/{name}_cache_paths.json')
 assert len(cal['sites'])==478 and cal['V_cache_scale_shared'] and cal['prefix_KV_extrema_included']
 assert sha(R/f'calibration/{name}_arrays.npz')==cal['arrays_sha256']
 for pol in ['minmax','mse','percentile']:
  p=R/f'calibration/{name}_{pol}_prefix_u8.npz';assert sha(p)==paths[pol]['prefix_artifact_sha256']
  with np.load(p) as z:
   assert len(z.files)==(56 if prefix['ids'] else 0)
   for key in z.files:assert z[key].dtype==np.uint8 and z[key].shape[-2]==len(prefix['ids'])
  seedfiles+=1
for v in ['F','C64']:
 dig=[read(f'checks/{v}_{phase}_weights.json') for phase in ['development','final']]
 diag=read(f'diagnostics/{v}.json')
 assert all(d['unchanged'] and d['digest']==diag['weight_digest'] for d in dig)
 assert diag['unchanged'] and diag['disabled_exact']
tail=read('checks/remaining_tail_localization.json')
with np.load(R/'calibration/eos_arrays.npz') as z:
 for name,d in tail.items():
  a=z[name+'/token_absmax'].reshape(128,128)
  assert d['tokens_over_1000']==int((a>1000).sum())
  assert d['positions_with_over_1000']==np.where((a>1000).any(0))[0].tolist()
assert read('checks/cache_model_oracle.json')['same_shape_logits_exact']
with (R/'independent_integrity_checks.json').open('x') as f:json.dump(dict(pass_all=True,score_files=len(scored),U8_prefix_artifacts=seedfiles,score_arithmetic_and_ids=True,cache_layout_and_storage=True,weights_unchanged=True,selection_development_only=True,inputs_frozen=True),f,indent=2)
print('INDEPENDENT_INTEGRITY_PASS',len(scored),seedfiles)
