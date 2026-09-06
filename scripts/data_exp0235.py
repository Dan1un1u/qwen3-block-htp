#!/usr/bin/env python3
"""Frozen C64 restoration sets and shared PC052 evaluation inputs."""
import json
from pathlib import Path
from data_exp0229 import BASE,SOURCE,MEMORY,MODEL,CELLS,sha,verified,preflight
import rotation_exp0219 as rot
RESULT=BASE/'exp0235';OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0235')
FAMILIES=['F','C64','ATT_ALL','MLP_ALL','HEAD']
def specs():
    attention={f'model.layers.{i}.{rot.PROJECTIONS[n]}.weight' for i in range(28) for n in ['q','k','v','o']}
    mlp={f'model.layers.{i}.{rot.PROJECTIONS[n]}.weight' for i in range(28) for n in ['gate','up','down']}
    out={'F':sorted(attention|mlp|{'lm_head.weight'}),'C64':[],'ATT_ALL':sorted(attention),'MLP_ALL':sorted(mlp),'HEAD':['lm_head.weight']}
    for fam,names in [('ATT',attention),('MLP',mlp)]:
        for i in range(28):out[f'{fam}_L{i:02d}']=sorted(n for n in names if n.startswith(f'model.layers.{i}.'))
    assert len(out)==61 and len(out['F'])==197 and not attention&mlp
    return out

def write(n,x):
    p=RESULT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')

def prepare():
    preflight();assert not (RESULT/'dataset_freeze.json').exists();RESULT.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    panel=verified('exp0233','dataset.json');dev=verified('exp0230','dataset.json')
    refs=[panel,dev,verified('exp0233','dataset_freeze.json'),verified('exp0233','independent_data_audit.json'),verified('exp0218','original_checkpoint_sha256.json'),verified('exp0230','C64/package.json')]
    with (RESULT/'dataset.json').open('xb') as f:f.write(panel.read_bytes())
    write('variants.json',dict(development=specs(),confirmation_families=FAMILIES,selection=False,confirmation_both_halves_unconditional=True))
    reused={}
    for phase in ['primary','reserve']:
        for v in ['F','C64']:
            name=f'software/{phase}_{v}.json';p=verified('exp0233',name);dest=RESULT/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(p.read_bytes())
            refs.append(p);reused[name]=dict(source=str(p),sha256=sha(p),new_measurement=False)
    write('reused_controls.json',reused)
    write('dataset_freeze.json',dict(files={n:sha(RESULT/n) for n in ['dataset.json','variants.json','reused_controls.json']},references={str(p):sha(p) for p in refs},protocol_sha256=sha(MEMORY/'docs/experiments/EXP-0235.md'),role='shared_PC052_paired_panel_independent_of_calibration_not_new_per_phase',development_role='exposed_EXP230_descriptive_only',frozen_before_scoring=True))
    print('EXP235_INPUTS_FROZEN',sha(RESULT/'dataset_freeze.json'),flush=True)

def frozen():
    f=json.loads((RESULT/'dataset_freeze.json').read_text());assert sha(MEMORY/'docs/experiments/EXP-0235.md')==f['protocol_sha256']
    for n,h in f['files'].items():assert sha(RESULT/n)==h,n
    for n,h in f['references'].items():assert sha(n)==h,n
    assert json.loads((RESULT/'variants.json').read_text())['development']==specs()
    d=json.loads((RESULT/'dataset.json').read_text());old=json.loads((BASE/'exp0230/dataset.json').read_text())
    d['samples'] += [r for r in old['samples'] if r['split']=='development']
    return d
if __name__=='__main__':prepare()
