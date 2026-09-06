#!/usr/bin/env python3
"""Paired document-bootstrap PPL acceptance; declared gates never tuned."""
import argparse,json,math
import numpy as np
from data_exp0229 import RESULT,CELLS,sha,write
from measure_exp0229 import frozen,VARIANTS

def statistics(rows,losses,seed=229):
    """Each row is a distinct document with 16 scored tokens."""
    groups={'overall':CELLS,'en':['en_wiki','en_news'],'zh':['zh_wiki','zh_news'],
            'wiki':['en_wiki','zh_wiki'],'news':['en_news','zh_news'],
            **{c:[c] for c in CELLS}}
    rng=np.random.default_rng(seed);means={};replicates={}
    for cell in CELLS:
        indices=np.array([i for i,r in enumerate(rows) if r['cell']==cell])
        assert len(indices)>0
        boot=rng.integers(0,len(indices),size=(5000,len(indices)))
        means[cell]={v:float(a[indices].mean()) for v,a in losses.items()}
        replicates[cell]={v:a[indices][boot].mean(1) for v,a in losses.items()}
    result={}
    for name,cells in groups.items():
        values={v:float(np.mean([means[c][v] for c in cells])) for v in losses}
        boot={v:np.mean([replicates[c][v] for c in cells],axis=0) for v in losses}
        threshold=1.05 if name=='overall' else 1.10
        comparisons={}
        for v in ['A0','A']:
            delta=values[v]-values['F'];ci=np.exp(np.quantile(boot[v]-boot['F'],[.025,.975]))
            comparisons[v]=dict(delta_nll=delta,ppl_ratio=math.exp(delta),ratio_ci95=ci.tolist(),
                limit=threshold,point_pass=math.exp(delta)<=threshold,
                confident_pass=float(ci[1])<=threshold,straddles_gate=float(ci[0])<=threshold<=float(ci[1]))
        ci=np.exp(np.quantile(boot['A']-boot['A0'],[.025,.975]))
        result[name]=dict(nll=values,ppl={v:math.exp(n) for v,n in values.items()},vs_F16=comparisons,
            A_vs_A0=dict(ppl_ratio=math.exp(values['A']-values['A0']),ratio_ci95=ci.tolist()))
    return result

def summarize(combined=False):
    frozen();dataset=json.loads((RESULT/'dataset.json').read_text())
    rows=[r for r in dataset['samples'] if combined or r['split']=='primary']
    expected=[r['id'] for r in rows];losses={}
    input_hashes={}
    for v in VARIANTS:
        samples={}
        for phase in (['primary','reserve'] if combined else ['primary']):
            for i in range(16):
                p=RESULT/'device'/phase/(f'{phase}_{i:02d}_{v}.validated.json')
                d=json.loads(p.read_text());input_hashes[str(p.relative_to(RESULT))]=sha(p)
                assert d['raw_sha256']==sha(p.with_suffix('').with_suffix('.jsonl'))
                for row in d['samples']:
                    assert row['id'] not in samples
                    samples[row['id']]=float(np.mean([s['nll'] for s in row['steps']]))
        assert set(samples)==set(expected)
        losses[v]=np.array([samples[i] for i in expected])
    teacher=json.loads((RESULT/'teacher_bf16.json').read_text());assert teacher['dataset_sha256']==sha(RESULT/'dataset.json')
    by_id={r['id']:r for r in teacher['samples']}
    losses['BF16_GPU']=np.array([np.mean(by_id[i]['nll']) for i in expected])
    assert all(np.isfinite(a).all() for a in losses.values())
    result=statistics(rows,losses)
    need_reserve=not combined and any(d['straddles_gate'] for g in result.values() for d in g['vs_F16'].values())
    status={}
    for v in ['A0','A']:
        metrics=[g['vs_F16'][v] for g in result.values()]
        status[v]='fail' if not all(d['point_pass'] for d in metrics) else ('pass' if all(d['confident_pass'] for d in metrics) else 'inconclusive')
    eligible=[v for v in ['A0','A'] if status[v]=='pass']
    recommended=None
    if eligible:
        recommended='A' if 'A' in eligible and ('A0' not in eligible or result['overall']['A_vs_A0']['ratio_ci95'][1]<1) else 'A0'
    summary=dict(experiment='EXP-0229',phase='combined' if combined else 'primary',documents=len(rows),
        scored_targets=len(rows)*16,statistics=result,reserve_required=need_reserve,status=status,
        recommended_candidate= recommended,baseline_promoted=False,bootstrap_replicates=5000,
        bootstrap_seed=229,bootstrap_unit='distinct_document_stratified_by_four_equal_cells',
        dataset_sha256=sha(RESULT/'dataset.json'),inputs=input_hashes,
        scope='M64 context plus16 conditional targets; short-context acceptance only')
    name='combined_summary.json' if combined else 'primary_summary.json'
    write(name,summary)
    print(json.dumps(dict(status=status,reserve_required=need_reserve,targets=len(rows)*16,overall=result['overall']),indent=2),flush=True)
    return summary

def test_statistics():
    rows=[dict(cell=c) for c in CELLS for _ in range(10)]
    f=np.linspace(1,5,40);a=f+math.log(1.02)
    result=statistics(rows,dict(F=f,A0=a,A=a,BF16_GPU=f))
    for name,g in result.items():
        assert abs(g['vs_F16']['A']['ppl_ratio']-1.02)<1e-12
        assert np.max(np.abs(np.array(g['vs_F16']['A']['ratio_ci95'])-1.02))<1e-12
        assert g['vs_F16']['A']['confident_pass']
        assert abs(g['A_vs_A0']['ppl_ratio']-1)<1e-12
    # Mean-NLL aggregation must not accidentally become arithmetic-mean PPL.
    assert abs(result['overall']['ppl']['F']-math.exp(f.mean()))<1e-12
    bad=statistics(rows,dict(F=f,A0=f+.2,A=f+.2))
    assert not bad['overall']['vs_F16']['A']['point_pass']
    print('STATISTICS_ORACLE_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--combined',action='store_true');p.add_argument('--test',action='store_true');a=p.parse_args()
    test_statistics() if a.test else summarize(a.combined)
