#!/usr/bin/env python3
"""Fixed paired sensitivity statistics; no selected hybrid or additive attribution."""
import json,math
import numpy as np
from data_exp0235 import RESULT,CELLS,FAMILIES,specs,write,sha,preflight,frozen
from summarize_exp0229 import statistics,test_statistics
GROUPS={'overall':CELLS,'en':['en_wiki','en_news'],'zh':['zh_wiki','zh_news'],'wiki':['en_wiki','zh_wiki'],'news':['en_news','zh_news'],**{c:[c] for c in CELLS}}

def main():
    preflight();data=frozen();audit=json.loads((RESULT/'restoration_audits.json').read_text());assert audit['pass_all'] and audit['immutable_snapshots_exact'] and audit['final_C64_sentinel_exact']
    manifest=json.loads((RESULT/'restoration_manifest.json').read_text());assert len(audit['audits'])==65
    reused=json.loads((RESULT/'reused_controls.json').read_text());inputs={};reduction_count=0
    def analyze(phases,variants):
        nonlocal reduction_count
        rows=[r for r in data['samples'] if r['split'] in phases];ids=[r['id'] for r in rows];losses={};samples={}
        for v in variants:
            merged=[]
            for phase in phases:
                name=f'software/{phase}_{v}.json';p=RESULT/name;run=json.loads(p.read_text())
                assert run['dataset_sha256']==sha(RESULT/'dataset.json') and run['torch_version']==json.loads((RESULT/'environment.json').read_text())['torch']
                assert run['repeat_exact'] and run['causal_mask_exact'] and run['independent_CE_max_abs']<5e-6
                if name in reused:assert sha(p)==reused[name]['sha256']
                else:assert run['restoration_manifest_sha256']==sha(RESULT/'restoration_manifest.json') and run['all_logits_finite']
                expected=[r for r in rows if r['split']==phase]
                assert [(r['id'],r['cell']) for r in run['samples']]==[(r['id'],r['cell']) for r in expected]
                assert abs(math.exp(math.fsum(x for r in run['samples'] for x in r['nll'])/(len(expected)*16))-run['ppl'])<1e-10
                merged+=run['samples'];inputs[name]=sha(p)
            assert len(merged)==len(rows);byid={r['id']:r for r in merged};assert len(byid)==len(rows) and set(byid)==set(ids)
            assert all(len(r['nll'])==16 and np.isfinite(r['nll']).all() for r in merged)
            samples[v]=byid;losses[v]=np.array([math.fsum(byid[i]['nll'])/16 for i in ids])
        result={}
        for v in variants:
            stats=statistics(rows,dict(F=losses['F'],A0=losses['C64'],A=losses[v]),seed=235);groups={}
            for name,g in stats.items():
                nll=g['nll']['A'];den=g['nll']['A0']-g['nll']['F'];delta=nll-g['nll']['A0']
                chosen=[r['id'] for r in rows if r['cell'] in GROUPS[name]]
                raw=math.fsum(x for i in chosen for x in samples[v][i]['nll'])/(16*len(chosen))
                assert abs(raw-nll)<1e-12 and abs(math.exp(raw)-g['ppl']['A'])<1e-10
                reduction_count+=1
                groups[name]=dict(nll=nll,ppl=g['ppl']['A'],vs_F16=g['vs_F16']['A'],vs_C64=dict(delta_nll=delta,delta_nll_ci95=[math.log(x) for x in g['A_vs_A0']['ratio_ci95']],**g['A_vs_A0']),
                    fraction_C64_excess_NLL_removed=(-delta/den if den>0 else None))
            thresholds=[g['vs_F16'] for g in groups.values()]
            status='fail' if not all(x['point_pass'] for x in thresholds) else 'pass' if all(x['confident_pass'] for x in thresholds) else 'inconclusive'
            result[v]=dict(statistics=groups,restored_weight_count=manifest['variants'][v]['restored_weight_count'],restored_fraction_quantizable_weights=manifest['variants'][v]['restored_weight_count']/manifest['quantizable_weight_count'],diagnostic_reference_threshold_status=status)
        return dict(documents=len(rows),targets=16*len(rows),variants=result)
    dev=analyze(['development'],list(specs()));confirmation=analyze(['primary','reserve'],FAMILIES)
    assert reduction_count==594
    for v in ['F','C64']:
        run=json.loads((RESULT/f'software/development_{v}.json').read_text());assert run['exact_prior_development_regression']
    sentinel=json.loads((RESULT/'C64_final_sentinel.json').read_text());assert sentinel['samples']==json.loads((RESULT/'software/development_C64.json').read_text())['samples']
    inputs['C64_final_sentinel.json']=sha(RESULT/'C64_final_sentinel.json')
    ranking=sorted([v for v in specs() if '_L' in v],key=lambda v:dev['variants'][v]['statistics']['overall']['vs_C64']['delta_nll'])
    write('summary.json',dict(experiment='EXP-0235',role='fixed_precision_sensitivity_diagnostic_no_selected_recipe',development=dev,confirmation=confirmation,
        development_layer_ranking=ranking,bootstrap=dict(replicates=5000,seed=235,unit='paired_document_stratified_four_cells'),independent_raw_token_PPL_reductions=reduction_count,inputs=inputs,
        restoration_manifest_sha256=sha(RESULT/'restoration_manifest.json'),restoration_audit_sha256=sha(RESULT/'restoration_audits.json'),
        limitations='Layer effects conditional and nonadditive; exploratory intervals are pointwise, not simultaneous significance; all five families confirmed regardless of ranking; shared exposed PC052 panel independent of calibration.',baseline_promoted=False,device_speed='N/A_software_only'))
    print('EXP235_SUMMARY',flush=True)
    for v,x in confirmation['variants'].items():print(v,json.dumps(x['statistics']['overall']),flush=True)
    print('DEVELOPMENT_TOP10',ranking[:10],flush=True)
if __name__=='__main__':test_statistics();main()
