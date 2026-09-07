#!/usr/bin/env python3
"""Fixed paired document uncertainty and independent raw-token PPL reductions."""
import json,math
import numpy as np
from data_exp0236 import RESULT,BASE,CELLS,write,sha,verified,preflight,frozen
from summarize_exp0229 import statistics,test_statistics
VARIANTS=['F','G64','P64','H64']
GROUPS={'overall':CELLS,'en':['en_wiki','en_news'],'zh':['zh_wiki','zh_news'],'wiki':['en_wiki','zh_wiki'],'news':['en_news','zh_news'],**{c:[c] for c in CELLS}}
def main():
 preflight();d=frozen();test_statistics();old=json.loads(verified('exp0233','dataset.json').read_text());dev=json.loads(verified('exp0230','dataset.json').read_text())
 datasets={'development':[r for r in dev['samples'] if r['split']=='development'],'final':d['samples'],'PC052':old['samples']}
 hashes={'development':sha(BASE/'exp0230/dataset.json'),'final':sha(RESULT/'dataset.json'),'PC052':sha(BASE/'exp0233/dataset.json')}
 for phase,rows in datasets.items():
  losses={};runs={};inputs={};ids=[r['id'] for r in rows]
  for v in VARIANTS:
   p=RESULT/f'software/{phase}_{v}.json';r=json.loads(p.read_text());assert r['dataset_sha256']==hashes[phase] and r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
   assert [s['id'] for s in r['samples']]==ids and [s['cell'] for s in r['samples']]==[s['cell'] for s in rows]
   assert all(len(s['nll'])==16 and np.isfinite(s['nll']).all() for s in r['samples'])
   runs[v]=r;losses[v]=np.array([math.fsum(s['nll'])/16 for s in r['samples']]);inputs[str(p.relative_to(RESULT))]=sha(p)
  result={};computed={v:statistics(rows,dict(F=losses['F'],A0=losses['G64'],A=losses[v]),seed=236) for v in ['P64','H64']}
  for name,cells in GROUPS.items():
   chosen=[i for i,r in enumerate(rows) if r['cell'] in cells];nll={v:math.fsum(x for i in chosen for x in runs[v]['samples'][i]['nll'])/(16*len(chosen)) for v in VARIANTS}
   g=computed['P64'][name];h=computed['H64'][name]
   for key,v in [('F','F'),('A0','G64'),('A','P64')]:assert abs(math.exp(nll[v])-g['ppl'][key])<1e-10
   assert abs(math.exp(nll['H64'])-h['ppl']['A'])<1e-10
   result[name]=dict(nll=nll,ppl={v:math.exp(x) for v,x in nll.items()},vs_F16={'G64':g['vs_F16']['A0'],'P64':g['vs_F16']['A'],'H64':h['vs_F16']['A']},vs_G64={'P64':g['A_vs_A0'],'H64':h['A_vs_A0']})
  status={}
  for v in ['G64','P64','H64']:
   gates=[r['vs_F16'][v] for r in result.values()];status[v]='fail' if not all(x['point_pass'] for x in gates) else 'pass' if all(x['confident_pass'] for x in gates) else 'inconclusive'
  out=dict(experiment='EXP-0236',phase=phase,role='fresh_independent_fixed_final' if phase=='final' else 'exposed_regression_no_selection',documents=len(rows),targets=16*len(rows),statistics=result,status=status,inputs=inputs,dataset_sha256=hashes[phase],independent_raw_PPL_reductions=36,bootstrap_seed=236,bootstrap_replicates=5000,bootstrap_unit='paired_document_four_equal_cells',baseline_promoted=False)
  write('summary_'+phase+'.json',out);print('HEAD_SUMMARY',phase,json.dumps(dict(status=status,overall=result['overall'])),flush=True)
if __name__=='__main__':main()
