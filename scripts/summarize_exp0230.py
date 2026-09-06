#!/usr/bin/env python3
"""Predeclared paired document bootstrap and acceptance decision."""
import argparse,json
import numpy as np
from data_exp0230 import RESULT,write,sha,preflight
from export_exp0230 import frozen
from summarize_exp0229 import statistics
from device_exp0230 import variants

def summarize(combined=False):
    preflight();dataset=frozen();vs=variants()
    rows=[r for r in dataset['samples'] if r['split'] in (['primary','reserve'] if combined else ['primary'])]
    expected=[r['id'] for r in rows];losses={};inputs={}
    for key,v in zip(['F','A0','A'],vs):
        records={}
        for split in (['primary','reserve'] if combined else ['primary']):
            for i in range(16):
                p=RESULT/f'device/{split}/{split}_{i:02d}_{v}.validated.json';d=json.loads(p.read_text())
                assert sha(p.with_suffix('').with_suffix('.jsonl'))==d['raw_sha256'];inputs[str(p.relative_to(RESULT))]=sha(p)
                for row in d['samples']:
                    assert row['id'] not in records;records[row['id']]=float(np.mean([s['nll'] for s in row['steps']]))
        assert set(records)==set(expected)
        losses[key]=np.array([records[i] for i in expected])
    st=statistics(rows,losses,seed=230);status={}
    for key in ['A0','A']:
        metrics=[g['vs_F16'][key] for g in st.values()]
        status[key]='fail' if not all(m['point_pass'] for m in metrics) else 'pass' if all(m['confident_pass'] for m in metrics) else 'inconclusive'
    reserve=not combined and any(m['straddles_gate'] for g in st.values() for m in g['vs_F16'].values())
    result=dict(experiment='EXP-0230',phase='combined' if combined else 'primary',variant_mapping=dict(zip(['F','A0','A'],vs)),
        documents=len(rows),targets=len(rows)*16,statistics=st,status=status,reserve_required=reserve,
        bootstrap=dict(unit='paired_document_stratified_four_cells',replicates=5000,seed=230),
        baseline_promoted=False,selected_before_final=True,selection_sha256=sha(RESULT/'selection.json'),
        dataset_sha256=sha(RESULT/'dataset.json'),inputs=inputs)
    write('combined_summary.json' if combined else 'primary_summary.json',result)
    print(json.dumps(dict(status=status,reserve_required=reserve,overall=st['overall']),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--combined',action='store_true');a=p.parse_args();summarize(a.combined)
