#!/usr/bin/env python3
"""Predeclared paired acceptance for both independently trained references."""
import argparse,json,math
import numpy as np
from data_exp0233 import RESULT,CELLS,write,sha,preflight
from autoround_exp0233 import frozen
from summarize_exp0229 import statistics,test_statistics
VARIANTS=['F','C64','AR-P','AR-G']
def summarize(combined=False):
 preflight();d=frozen();phases=['primary','reserve'] if combined else ['primary']
 if combined:assert json.loads((RESULT/'summary_primary.json').read_text())['reserve_trigger']
 rows=[r for r in d['samples'] if r['split'] in phases];ids=[r['id'] for r in rows];losses={};inputs={};full={}
 for v in VARIANTS:
  samples={}
  for phase in phases:
   p=RESULT/f'software/{phase}_{v}.json';r=json.loads(p.read_text())
   assert r['dataset_sha256']==sha(RESULT/'dataset.json') and r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
   assert [s['id'] for s in r['samples']]==[s['id'] for s in rows if s['split']==phase]
   for s in r['samples']:
    assert s['id'] not in samples and len(s['nll'])==16 and np.isfinite(s['nll']).all();samples[s['id']]=s
   inputs[str(p.relative_to(RESULT))]=sha(p)
  full[v]=samples;losses[v]=np.array([math.fsum(samples[i]['nll'])/16 for i in ids])
 result={};status={};trigger=False
 groups={'overall':CELLS,'en':['en_wiki','en_news'],'zh':['zh_wiki','zh_news'],'wiki':['en_wiki','zh_wiki'],'news':['en_news','zh_news'],**{c:[c] for c in CELLS}}
 for v in ['AR-P','AR-G']:
  st=statistics(rows,dict(F=losses['F'],A0=losses['C64'],A=losses[v]),seed=233);gates=[]
  for name,g in st.items():
   target=result.setdefault(name,dict(nll={},ppl={},vs_F16={},vs_C64={}))
   for old,new in [('F','F'),('A0','C64'),('A',v)]:target['nll'][new]=g['nll'][old];target['ppl'][new]=g['ppl'][old]
   target['vs_F16']['C64']=g['vs_F16']['A0'];target['vs_F16'][v]=g['vs_F16']['A'];target['vs_C64'][v]=g['A_vs_A0'];gates.append(g['vs_F16']['A'])
  status[v]='fail' if not all(g['point_pass'] for g in gates) else 'pass' if all(g['confident_pass'] for g in gates) else 'inconclusive'
  trigger|=any(g['straddles_gate'] for g in gates)
 for name,cells in groups.items():
  chosen=[r['id'] for r in rows if r['cell'] in cells]
  for v in VARIANTS:
   value=math.exp(math.fsum(x for i in chosen for x in full[v][i]['nll'])/(16*len(chosen)))
   assert abs(value-result[name]['ppl'][v])<1e-10
 out=dict(experiment='EXP-0233',phase='combined' if combined else 'primary',documents=len(rows),targets=16*len(rows),statistics=result,status=status,
  reserve_trigger=bool(trigger and not combined),bootstrap=dict(unit='paired_document_stratified_four_cells',replicates=5000,seed=233),independent_raw_token_reduction_count=36,
  baseline_promoted=False,device_speed='N/A',dataset_sha256=sha(RESULT/'dataset.json'),inputs=inputs)
 write('summary_combined.json' if combined else 'summary_primary.json',out)
 print(json.dumps(dict(status=status,reserve_trigger=out['reserve_trigger'],overall=result['overall']),indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--combined',action='store_true');a=p.parse_args();test_statistics();summarize(a.combined)
