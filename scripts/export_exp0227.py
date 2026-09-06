#!/usr/bin/env python3
"""Export trained FP32 row scales directly, preserving all existing GPTQ codes."""
import argparse
import json
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer
import eval_exp0218 as ev
import rotation_exp0219 as rot
import prepare_exp0164_generation_package as pack
import export_exp0226 as oracle
from experiment_exp0220 import exact_generation
from block_reconstruction_exp0227 import RESULT,OUTPUT,CONTROLS,PLAN,read_codes,verify_package,preflight,digest


@torch.no_grad()
def export(v):
    complete=json.loads((RESULT/'training'/v/'complete.json').read_text())
    assert complete['layers']==28 and complete['trainable_parameters']==573440 and not complete['smoke']
    original=verify_package(v);root=OUTPUT/v;result=RESULT/v
    assert not root.exists()
    result.mkdir(parents=True,exist_ok=True)
    shutil.copytree(CONTROLS[v],root,copy_function=pack.copy_link)
    changed=[];checks=[]
    for i in range(28):
        scales=torch.load(OUTPUT/'training'/v/f'layer{i:02d}_scales.pt',weights_only=True)
        for short,s in scales.items():
            prefix=f'layer{i}/{short}'
            n=len(s);byte_count=(root/(prefix+'_weight_w4_hmx.bin')).stat().st_size;k=byte_count*2//n
            codes,old_scale=read_codes(root/f'layer{i}',short,(n,k))
            s=s.numpy().astype('<f4');assert np.isfinite(s).all() and (s>0).all()
            name=prefix+'_weight_w4_scale_f32.bin';pack.atomic_write_bytes(root/name,s.tobytes());changed.append(name)
            # Independent NumPy expansion vs training's Torch FP32 product/FP16 cast.
            actual=(codes.numpy().astype(np.float32)*s[:,None]).astype(np.float16)
            expected=(codes.float()*torch.from_numpy(s)[:,None]).half().numpy()
            assert np.array_equal(actual,expected) and np.isfinite(actual).all()
            assert digest(root/(prefix+'_weight_w4_hmx.bin'))==digest(CONTROLS[v]/(prefix+'_weight_w4_hmx.bin'))
            ratio=s/old_scale.numpy();assert ratio.min()>=.5-1e-6 and ratio.max()<=2+1e-6
            checks.append(dict(layer=i,projection=short,train_export_weight_exact=True,codes_exact=True,
                scale_ratio_min=float(ratio.min()),scale_ratio_max=float(ratio.max()),
                changed_rows=int(np.count_nonzero(s!=old_scale.numpy()))))
    manifest=original.copy();old=original['files']
    manifest['files']={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),r) for n,r in old.items()}
    for n in changed:manifest['files'][n]=pack.file_record(root/n)
    manifest.update(experiment='EXP-0227',variant=v,transform='fixed_coordinate_block_reconstruction_scale_only',
        quantizer='frozen_GPTQ_codes_learned_FP32_row_scales_no_requantization',
        source_control=str(CONTROLS[v]),source_control_manifest_sha256=digest(CONTROLS[v]/'manifest.json'),
        reconstruction_protocol_sha256=digest(RESULT/'protocol.json'),changed_files=changed,
        inherited_replay_references='historical placeholders; not valid for new layer replay')
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    model=oracle.load_package(root)
    tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    generation=exact_generation(model,tok);rot.write_json(result/'software_generation.json',generation)
    name='generation_expected_token_ids_u32.bin'
    pack.atomic_write_bytes(root/name,np.asarray(generation['token_ids'],dtype='<u4').tobytes());changed.append(name)
    manifest['files'][name]=pack.file_record(root/name)
    manifest['generation']=dict(manifest['generation'],independent_expected_token_ids=generation['token_ids'],
        independent_expected_text=generation['text'],semantic_reference_sha256=digest(result/'software_generation.json'))
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    nonchanged=[n for n in manifest['files'] if n not in changed]
    assert all(digest(root/n)==digest(CONTROLS[v]/n) for n in nonchanged)
    rot.write_json(result/'export_oracle.json',dict(passed=True,projections=checks,nonchanged_files_exact=len(nonchanged),
        changed_files=changed,all_codes_and_nontransformer_tensors_frozen=True,manifest_sha256=digest(root/'manifest.json')))
    print('EXPORT_COMPLETE',v,flush=True)


@torch.no_grad()
def validate(v):
    model=oracle.load_package(OUTPUT/v)
    # Reuse canonical CPU FP16 scoring and independent validation rows exactly.
    oracle.RESULT=RESULT
    oracle.validate_model(model,v)


@torch.no_grad()
def quality(v):
    model=oracle.load_package(OUTPUT/v);tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    rot.write_json(RESULT/v/'software_quality.json',dict(samples=rot.quality(model,tok,ev.dataset()),
        role='final packed scale-reconstructed W4; software separate from actual DSP'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['export','validate','quality']);p.add_argument('variant',choices=['A','R']);a=p.parse_args()
    preflight();torch.set_grad_enabled(False);torch.set_num_threads(8)
    if a.phase=='export':export(a.variant)
    elif a.phase=='validate':validate(a.variant)
    else:quality(a.variant)
