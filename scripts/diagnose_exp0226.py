#!/usr/bin/env python3
"""Frozen-control provenance, identical initialization and quantizer diagnostics."""
import argparse,gc,json,shutil,subprocess
import numpy as np
import torch
from learned_rotation_exp0226 import RESULT,OUTPUT,PLAN,LearnedModel
from rotation_exp0219 import write_json
from prepare_exp0164_generation_package import sha256_file as sha
from run_exp0164_semantic_gate import load_model
import eval_exp0218 as ev
OLD=RESULT.parent/'exp0225'

def inherit():
    ledger=OLD/'evidence_sha256.json';assert sha(ledger)=='1d73989c7a6f4d5d8b91db553db32ee22fa023011f125c4ebe29718234ed95d8'
    evidence=json.loads(ledger.read_text());files=['algebra_oracle.json','output_scale_oracle.json','full_model_oracle.json','exact_cayley_oracle.json'];record={}
    for name in files:
        assert sha(OLD/name)==evidence[name];assert not (RESULT/name).exists()
        shutil.copyfile(OLD/name,RESULT/name);record[name]=dict(source=str(OLD/name),sha256=evidence[name])
    paths=['scripts/learned_rotation_exp0225.py','scripts/exact_cayley_exp0225.py','scripts/output_scale_exp0224.py','scripts/gptq_exp0221.py']
    diff=subprocess.check_output(['git','diff','0900322f71b8cbc6071a3a34e3f23679464bbfeb','--',*paths],text=True);assert not diff
    write_json(RESULT/'inherited_oracles.json',dict(files=record,unchanged_math_sources=paths,parent_head='0900322f71b8cbc6071a3a34e3f23679464bbfeb',meaning='prior independent mathematical oracles reused unchanged; current data smoke and full export invariance still mandatory'))

def identity():
    new=OUTPUT/'smoke_exact/step000.pt';old=OUTPUT.parent/'exp0225/training_exact/step000.pt'
    a=torch.load(new,map_location='cpu',weights_only=False);b=torch.load(old,map_location='cpu',weights_only=False)
    record={k+'_exact':torch.equal(a['rotations'][k],b['rotations'][k]) for k in ['R1','R2']};assert all(record.values())
    assert a['plan']==PLAN and a['data_sha256']==sha(RESULT/'learning_data.json')
    record.update(new_checkpoint_sha256=sha(new),old_checkpoint_sha256=sha(old),data_sha256=a['data_sha256'])
    write_json(RESULT/'initial_rotation_identity.json',record)

@torch.no_grad()
def controls():
    from export_exp0226 import load_package,validate_model
    torch.set_num_threads(8)
    assert json.loads((RESULT/'initial_rotation_identity.json').read_text())['R1_exact']
    for variant,relative,expected in [('step000','exp0225/step000','de788d23a754e1de635268fff1e0caebd21e61344b838cd242b2a3b74139c77b'),('old100','exp0225/step100','44ac941297f1aab82ec4daff2ca8191379def73f6796a1c9c5d7d700820c5673'),('control_A','exp0224/A',None)]:
        root=OUTPUT.parent/relative
        if expected:assert sha(root/'manifest.json')==expected
        dest=RESULT/variant;dest.mkdir(parents=True,exist_ok=True)
        assert not (dest/'validation.json').exists()
        if variant=='step000':
            if not (OUTPUT/variant).exists():(OUTPUT/variant).symlink_to(root,target_is_directory=True)
            assert (OUTPUT/variant).resolve()==root.resolve()
            for name in ['software_generation.json','invariance.json','calibration_forward_checks.json','weight_stats.json']:shutil.copyfile(OLD/variant/name,dest/name)
        record=dict(path=str(root),manifest_sha256=sha(root/'manifest.json'),immutable_control=True,initial_rotation_identity_sha256=sha(RESULT/'initial_rotation_identity.json'))
        if (dest/'reused_package.json').exists():assert json.loads((dest/'reused_package.json').read_text())==record
        else:write_json(dest/'reused_package.json',record)
        model=load_package(root);validate_model(model,variant);del model;gc.collect();print('CONTROL_VALIDATED',variant,flush=True)

@torch.no_grad()
def surrogate():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=json.loads((RESULT/'learning_data.json').read_text());base=load_model(ev.MODEL,torch.bfloat16);model=LearnedModel(base);del base;gc.collect()
    model.recompute=False;model.eval()
    for variant,path in [('step000',OUTPUT/'training_exact/step000.pt'),('old100',OUTPUT.parent/'exp0225/training_exact/step100.pt'),('step050',OUTPUT/'training_exact/step050.pt'),('step100',OUTPUT/'training_exact/step100.pt')]:
        state=torch.load(path,map_location='cpu',weights_only=False);model.r1.copy_(state['rotations']['R1'])
        for p,v in zip(model.r2,state['rotations']['R2']):p.copy_(v)
        rows=[]
        for i,s in enumerate(data['validation']):
            nll=float(model.loss(torch.tensor([s['token_ids']],device='cuda')));assert np.isfinite(nll)
            rows.append(dict(language=s['language'],domain=s['domain'],row_index=s['row_index'],nll=nll,tokens=127))
        write_json(RESULT/f'surrogate_{variant}.json',dict(samples=rows,nll=float(np.mean([r['nll'] for r in rows])),language_nll={lang:float(np.mean([r['nll'] for r in rows if r['language']==lang])) for lang in ['en','zh']},domain_language_nll={d+'_'+l:float(np.mean([r['nll'] for r in rows if r['domain']==d and r['language']==l])) for d in ['wiki','news'] for l in ['en','zh']},dataset_sha256=sha(RESULT/'learning_data.json'),rotation_checkpoint_sha256=sha(path),quantizer=PLAN['quantizer'],execution='GPU unchanged training path; actual GPTQ software uses CPU FP16, no bitwise cross-device claim'))
        print('SURROGATE',variant,float(np.mean([r['nll'] for r in rows])),flush=True)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inherit','identity','controls','surrogate']);a=p.parse_args();globals()[a.phase]()
