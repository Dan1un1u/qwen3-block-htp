#!/usr/bin/env python3
"""PC060: split post-RoPE Q/K and tied V; reuse verified PC059 arithmetic."""
import argparse,json,subprocess
from functools import lru_cache
import torch
import ablate_exp0244 as base
import data_exp0245 as data

ROOT=data.RESULT
SCOPES={'L0':range(1),'EARLY':range(1,7),'LATE':range(7,28),'ALL':range(28)}
FIELDS={'Q':['q_rope'],'K':['k_cache'],'V':['v_out','v_cache'],'QK':['q_rope','k_cache'],'QKV':['q_rope','k_cache','v_out','v_cache']}
@lru_cache(maxsize=1)
def masks():
    out={scope+'_'+kind:sorted(f'L{i:02d}.{n}' for i in layers for n in FIELDS[kind])
         for scope,layers in SCOPES.items() for kind in ['Q','K','V','QK']}
    for scope in ['L0','ALL']:out[scope+'_QKV']=sorted(f'L{i:02d}.{n}' for i in SCOPES[scope] for n in FIELDS['QKV'])
    out['L0_FULL']=sorted(n for n in base.params() if n.startswith('L00.') and not n.endswith('embedding_out'))
    assert len(out['L0_FULL'])==17
    for scope in SCOPES:assert set(out[scope+'_QK'])==set(out[scope+'_Q'])|set(out[scope+'_K'])
    return out
def final_names():return ['F','C64','A8']+[scope+'_'+k for scope in ['L0','ALL'] for k in ['Q','K','V','QK','QKV']]
def bind():
    # Rebind only experiment configuration of the imported module; no old files change.
    base.ROOT=ROOT;base.data=data;base.masks=masks
def read(n):return json.loads((ROOT/n).read_text())
def once(n,x):
    if (ROOT/n).exists():assert read(n)==x,n
    else:data.write(n,x)
def preflight():
    data.preflight()
    out=subprocess.check_output(['python3','-c','import yaml;print(yaml.safe_load(open("'+str(data.MEMORY/'PROJECT_STATUS.yaml')+'"))["governance"]["active_experiment"])'],text=True)
    assert out.strip()=='EXP-0245'
def freeze():
    parent=json.loads(data.verified('exp0244','dataset.json').read_text())
    samples=[r for i in range(8) for c in base.a.CELLS for r in [[v for v in parent['samples'] if v['split']=='development' and v['cell']==c][i]]]
    data.write('development.json',dict(role='exposed_EXP0244_first8_per_cell_sequential_development',samples=samples,parent_sha256=data.sha(data.verified('exp0244','dataset.json'))))
    data.write('masks.json',dict(masks=masks(),development_names=['F','C64','A8']+list(masks()),final_names=final_names(),
        development_sha256=data.sha(ROOT/'development.json'),dataset_sha256=data.sha(ROOT/'dataset.json'),
        protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0245.md'),parameters_sha256=data.sha(data.verified('exp0243','calibration/eos.json')),
        source_head=base.head(),final_list_frozen_before_inference=True))
def verify_freeze():
    d=read('masks.json');assert d['masks']==masks() and d['final_names']==final_names()
    assert d['protocol_sha256']==data.sha(data.MEMORY/'docs/experiments/EXP-0245.md')
    assert d['development_sha256']==data.sha(ROOT/'development.json') and d['dataset_sha256']==data.sha(ROOT/'dataset.json')
    assert d['parameters_sha256']==data.sha(data.verified('exp0243','calibration/eos.json'))
def mask_oracles(model,ins,samples):
    result={}
    for name in ['L0_Q','L0_K','L0_V','L0_QK','ALL_QKV']:
        ins.select(name)
        assert ins.restored==set(masks()[name])
        ins.cache_type=base.SelectiveCache;x,_,cache=base.forward(model,ins,samples,'sequential')
        dtypes=[dict(K=str(k.dtype),V=str(v.dtype)) for k,v in zip(cache.key_cache,cache.value_cache)]
        ins.cache_type=base.FloatOracleCache;y,_,_=base.forward(model,ins,samples,'sequential')
        assert torch.equal(x,y),name
        result[name]=dict(float_cache_exact=True,cache_length=80,append_calls=len(cache.appended),dtypes=dtypes,mask=sorted(ins.restored))
    ins.cache_type=base.SelectiveCache
    once('checks/mask_oracles.json',dict(pass_all=True,variants=result,scalar=base.a.scalar_oracle()))
def run(phase):
    frozen=read('masks.json');samples=read('development.json')['samples'] if phase=='development' else read('dataset.json')['samples']
    names=frozen[phase+'_names']
    for recipe in ['C64','F']:
        proof=f'checks/{recipe}_{phase}_weights.json'
        if (ROOT/proof).exists():
            assert all((ROOT/f'scores/{phase}_{n}_sequential.json').exists() for n in names if (n=='F')==(recipe=='F'))
            print('PHASE_RETAINED',recipe,phase,flush=True);continue
        model,before,mh=base.model_session(recipe);ins=base.Instrument(model);ins.build_prefix([151645])
        if phase=='development' and recipe=='C64':mask_oracles(model,ins,samples[:4])
        for n in names:
            if (n=='F')!=(recipe=='F'):continue
            result=base.score(model,ins,n,samples,phase,'sequential')
            mapped={'F':'F','C64':'C64','A8':'A8','L0_FULL':'layer_L00','ALL_QKV':'global_kv_and_qrope'}
            if phase=='development' and n in mapped:
                old=data.verified('exp0244',f'scores/rerank_{mapped[n]}_sequential.json')
                oldscore=json.loads(old.read_text());assert result['samples']==oldscore['samples'] and result['ppl']==oldscore['ppl']
                once(f'checks/reproduction_{n}.json',dict(exact=True,parent_score_sha256=data.sha(old),current_score_sha256=data.sha(ROOT/f'scores/{phase}_{n}_sequential.json')))
        assert before==base.a.state_digest(model)
        once(proof,dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=base.head()))
        ins.close();del ins,model;torch.cuda.empty_cache()
    if phase=='development':once('development_complete.json',dict(pass_all=True,configurations=len(names),final_not_used=True))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','development','final']);args=parser.parse_args()
    preflight();data.frozen();base.a.settings();bind()
    with torch.inference_mode():
        if args.phase=='freeze':freeze()
        else:
            verify_freeze()
            if args.phase=='final':assert read('development_complete.json')['pass_all']
            run(args.phase)
if __name__=='__main__':main()
