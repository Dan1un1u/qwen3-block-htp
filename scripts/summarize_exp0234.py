#!/usr/bin/env python3
"""Fixed group12864K software PPL acceptance, matched controls and bounded reserve."""
import argparse,json,math
import numpy as np
from data_exp0234 import RESULT,CELLS,write,sha,preflight
from data_exp0234 import frozen
from summarize_exp0229 import statistics, test_statistics

VARIANTS=['F','G8','C64','G64']

def summarize(combined=False):
    preflight();d=frozen();phases=['primary','reserve'] if combined else ['primary']
    if combined:assert json.loads((RESULT/'summary_primary.json').read_text())['reserve_trigger']
    rows=[r for r in d['samples'] if r['split'] in phases];ids=[r['id'] for r in rows]
    losses={};inputs={};full={}
    for v in VARIANTS:
        samples={}
        for phase in phases:
            p=RESULT/f'software/{phase}_{v}.json';run=json.loads(p.read_text())
            assert run['dataset_sha256']==sha(RESULT/'dataset.json')
            assert run['repeat_exact'] and run['causal_mask_exact'] and run['independent_CE_max_abs']<5e-6
            expected=[r['id'] for r in rows if r['split']==phase]
            assert [r['id'] for r in run['samples']]==expected
            for row in run['samples']:
                assert row['id'] not in samples and len(row['nll'])==16
                assert np.isfinite(row['nll']).all();samples[row['id']]=row
            inputs[str(p.relative_to(RESULT))]=sha(p)
        assert set(samples)==set(ids);full[v]=samples
        losses[v]=np.array([math.fsum(samples[i]['nll'])/16 for i in ids])
    base=statistics(rows,dict(F=losses['F'],A0=losses['C64'],A=losses['G64'],G8=losses['G8']),seed=234)
    extra=statistics(rows,dict(F=losses['F'],A0=losses['G8'],A=losses['G64']),seed=234)
    groups={'overall':CELLS,'en':['en_wiki','en_news'],'zh':['zh_wiki','zh_news'],
            'wiki':['en_wiki','zh_wiki'],'news':['en_news','zh_news'],**{c:[c] for c in CELLS}}
    result={}
    for name,g in base.items():
        e=extra[name];mapping={'F':'F','A0':'C64','A':'G64','G8':'G8'}
        result[name]=dict(nll={mapping[k]:n for k,n in g['nll'].items()},ppl={mapping[k]:n for k,n in g['ppl'].items()},
            vs_F16={'G8':e['vs_F16']['A0'],'C64':g['vs_F16']['A0'],'G64':g['vs_F16']['A']},
            G64_vs_C64=g['A_vs_A0'],G64_vs_G8=e['A_vs_A0'])
        # Independent raw-token reduction, not a mean of document/cell PPL.
        chosen=[r['id'] for r in rows if r['cell'] in groups[name]]
        for v in VARIANTS:
            ppl=math.exp(math.fsum(x for i in chosen for x in full[v][i]['nll'])/(16*len(chosen)))
            assert abs(ppl-result[name]['ppl'][v])<1e-10,(name,v)
    gates=[g['vs_F16']['G64'] for g in result.values()]
    status='fail' if not all(g['point_pass'] for g in gates) else 'pass' if all(g['confident_pass'] for g in gates) else 'inconclusive'
    trigger=not combined and any(g['straddles_gate'] for g in gates)
    out=dict(experiment='EXP-0234',phase='combined' if combined else 'primary',role='packed_FP16_GPU_software_only',
        documents=len(rows),targets=len(rows)*16,statistics=result,status=status,reserve_trigger=trigger,
        bootstrap=dict(unit='paired_document_stratified_four_cells',replicates=5000,seed=234),
        independent_raw_token_reduction_count=36,baseline_promoted=False,device_speed='N/A',
        dataset_sha256=sha(RESULT/'dataset.json'),inputs=inputs)
    write('summary_combined.json' if combined else 'summary_primary.json',out)
    print(json.dumps({k:out[k] for k in ['status','reserve_trigger','documents','targets']},indent=2),flush=True)
    print(json.dumps(result['overall'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--combined',action='store_true');a=p.parse_args()
    test_statistics();summarize(a.combined)
