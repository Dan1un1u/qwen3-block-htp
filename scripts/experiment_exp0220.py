#!/usr/bin/env python3
"""Original-checkpoint gamma attribution and gamma-preserving R2-only."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import eval_exp0218 as ev
import rotation_exp0219 as rot
import prepare_exp0164_generation_package as pack
from run_exp0164_semantic_gate import load_model

RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0220')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0220')
BASE=rot.BASE
VARIANTS=['qkv_fold','mlp_fold','head_fold','r2_only']

def verify_origin():
    ev.dataset()
    hashes=json.loads((ev.ROOT/'original_checkpoint_sha256.json').read_text())
    assert all(pack.sha256_file(ev.MODEL/n)==h for n,h in hashes.items())
    return hashes

def metrics(a,b):
    x,y=a.double().flatten(),b.double().flatten()
    return dict(nrmse=float((x-y).norm()/y.norm().clamp_min(1e-30)),
        cosine=float(F.cosine_similarity(x,y,dim=0)),max_abs=float((x-y).abs().max()),
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))

def quantized(w):
    w=w.float();scale=w.abs().amax(1).clamp_min(1e-30)/7
    return ((w/scale[:,None]).round().clamp(-7,7)*scale[:,None]).half()

def fp16_control():
    assert not (RESULT/'fp16_control.json').exists()
    hashes=verify_origin();data=ev.dataset();tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    probes=[next(s for s in data['samples'] if s['id']==i) for i in [0,20]]
    start=time.monotonic();model=load_model(ev.MODEL,torch.float16)
    original_layers={};conditionals=[];handles=[];sample_id=None
    def layer_hook(i):
        def hook(module,args,out):
            x=out[0] if isinstance(out,tuple) else out
            original_layers[(sample_id,i)]=x.clone()
        return hook
    def norm_hook(layer_index,norm_name,projections):
        def hook(module,args,out):
            x=args[0];unit=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+module.variance_epsilon)).to(x.dtype)
            # Last 16 positions of 79-token real input, enough to localize error
            # without creating a synthetic or gamma-divided activation proxy.
            unit=unit[:,-16:];actual=out[:,-16:];gamma=module.weight.float()
            for name,linear in projections:
                w=linear.weight.float();ref=F.linear(actual,w.half())
                baseline=F.linear(actual,quantized(w))
                folded=F.linear(unit,quantized(w*gamma[None,:]))
                folded_fp16=F.linear(unit,(w*gamma[None,:]).half())
                conditionals.append(dict(sample=sample_id,layer=layer_index,norm=norm_name,projection=name,
                    original_W4=metrics(baseline,ref),folded_W4=metrics(folded,ref),
                    folded_FP16=metrics(folded_fp16,ref)))
        return hook
    for i,layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
        handles.append(layer.input_layernorm.register_forward_hook(norm_hook(i,'input',[
            (n,layer.get_submodule(rot.PROJECTIONS[n])) for n in ['q','k','v']])))
        handles.append(layer.post_attention_layernorm.register_forward_hook(norm_hook(i,'post',[
            (n,layer.get_submodule(rot.PROJECTIONS[n])) for n in ['gate','up']])))
    handles.append(model.model.norm.register_forward_hook(norm_hook(28,'final',[('head',model.lm_head)])))
    probe_logits={}
    for s in probes:
        sample_id=s['id'];probe_logits[sample_id]=rot.logits(model,s)
        print(json.dumps(dict(phase='conditional_projection_diagnostic',sample=sample_id)),flush=True)
    for h in handles:h.remove()
    original=rot.quality(model,tok,data)
    del model;gc.collect()
    model=load_model(ev.MODEL,torch.float32);rot.transform(model,False);model.half()
    drift=[];handles=[]
    def folded_hook(i):
        def hook(module,args,out):
            x=out[0] if isinstance(out,tuple) else out
            drift.append(dict(sample=sample_id,layer=i,**metrics(x,original_layers[(sample_id,i)])))
        return hook
    for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(folded_hook(i)))
    logit_checks=[]
    for s in probes:
        sample_id=s['id'];actual=rot.logits(model,s)
        logit_checks.append(dict(sample=sample_id,**metrics(actual,probe_logits[sample_id]),
            top1_equal=int((actual.argmax(-1)==probe_logits[sample_id].argmax(-1)).sum())))
    for h in handles:h.remove()
    assert all(x['finite'] for x in drift+logit_checks)
    folded=rot.quality(model,tok,data)
    rot.write_json(RESULT/'fp16_control.json',dict(original_shards=hashes,
        role='Host-only diagnostic; frozen F16F16 device recipe/package unchanged',
        original=original,folded=folded,probe_logits=logit_checks,layer_drift=drift,
        conditional_projection=conditionals,elapsed_s=time.monotonic()-start))
    print('FP16_CONTROL_COMPLETE',flush=True)

def transform(model,variant):
    changed=[]
    # Deployment always has independent embedding/head tensors.
    if model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings=False
    for i,layer in enumerate(model.model.layers):
        if variant in ['qkv_fold','mlp_fold']:
            norm=layer.input_layernorm if variant=='qkv_fold' else layer.post_attention_layernorm
            for n in (['q','k','v'] if variant=='qkv_fold' else ['gate','up']):
                layer.get_submodule(rot.PROJECTIONS[n]).weight.mul_(norm.weight[None,:])
                changed.append((i,n))
            norm.weight.fill_(1)
        elif variant=='r2_only':
            w=layer.self_attn.v_proj.weight
            w.copy_(rot.fwht(w.reshape(8,128,2048),1).reshape_as(w))
            w=layer.self_attn.o_proj.weight
            w.copy_(rot.fwht(w.reshape(2048,16,128),2).reshape_as(w))
            changed.extend([(i,'v'),(i,'o')])
    if variant=='head_fold':model.lm_head.weight.mul_(model.model.norm.weight[None,:]);model.model.norm.weight.fill_(1)
    return changed

def load_packed(param,root,prefix):
    n,k=param.shape
    carrier=np.memmap(root/(prefix+'_weight_w4_hmx.bin'),dtype=np.uint8,mode='r',shape=(n//32,k//32,512))
    scales=np.fromfile(root/(prefix+'_weight_w4_scale_f32.bin'),dtype='<f4')
    assert len(scales)==n
    for first in range(0,n,1024):
        last=min(first+1024,n)
        q=rot.unpack(np.asarray(carrier[first//32:last//32]),last-first,k).float()
        param[first:last].copy_((q*torch.from_numpy(scales[first:last])[:,None]).half().float())

def exact_generation(model,tok):
    ids=torch.tensor([np.fromfile(BASE/'generation_prompt_token_ids_u32.bin',dtype='<u4').tolist()])
    tokens=[];past=None
    for step in range(16):
        result=model(ids,past_key_values=past,use_cache=True)
        token=int(result.logits[0,-1].float().argmax());tokens.append(token)
        past=result.past_key_values;ids=torch.tensor([[token]])
    return dict(token_ids=tokens,text=tok.decode(tokens,skip_special_tokens=False),
        stopping_rule='exactly16 greedy steps, same as runtime; EOS does not terminate speed scope')

def prepare(variant):
    root=OUTPUT/variant;directory=RESULT/variant
    assert not root.exists() and not directory.exists()
    hashes=verify_origin();directory.mkdir(parents=True);start=time.monotonic()
    data=ev.dataset();model=load_model(ev.MODEL,torch.float32)
    probes=[next(s for s in data['samples'] if s['id']==i) for i in [0,20]]
    refs=[rot.logits(model,s) for s in probes]
    changed=transform(model,variant)
    checks=[dict(sample=s['id'],**rot.difference(rot.logits(model,s),ref)) for s,ref in zip(probes,refs)]
    rot.write_json(directory/'invariance.json',dict(checks=checks,original_shards=hashes))
    assert all(c['passed'] for c in checks),checks
    root.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(BASE,root,copy_function=pack.copy_link)
    old_manifest=json.loads((BASE/'manifest.json').read_text());files=old_manifest['files']
    manifest=dict(old_manifest);manifest['files']={n.replace(chr(92),'/'):files.get(n.replace(chr(92),'/'),v) for n,v in files.items()}
    changed_files=[];stats={}
    def record(name):
        changed_files.append(name);manifest['files'][name]=pack.file_record(root/name)
    for i,layer in enumerate(model.model.layers):
        for short,long in rot.PROJECTIONS.items():
            param=layer.get_submodule(long).weight
            if (i,short) in changed:
                local={};rot.export_quantized(param,root/f'layer{i}',short,local)
                stats[f'layer{i}/{short}']=local[short]
                for suffix in ['_weight_w4_hmx.bin','_weight_w4_scale_f32.bin']:record(f'layer{i}/{short}'+suffix)
            else:load_packed(param,BASE/f'layer{i}',short)
        if variant in ['qkv_fold','mlp_fold']:
            n='input' if variant=='qkv_fold' else 'post';name=f'layer{i}/{n}_norm_weight_f16.bin'
            pack.atomic_write_bytes(root/name,np.ones(2048,dtype='<f2').tobytes());record(name)
        print(json.dumps(dict(variant=variant,layer=i)),flush=True)
    if variant=='head_fold':
        rot.export_quantized(model.lm_head.weight,root,'generation_lm_head',stats)
        for suffix in ['_weight_w4_hmx.bin','_weight_w4_scale_f32.bin']:record('generation_lm_head'+suffix)
        name='generation_final_norm_weight_f16.bin';pack.atomic_write_bytes(root/name,np.ones(2048,dtype='<f2').tobytes());record(name)
    else:load_packed(model.lm_head.weight,BASE,'generation_lm_head')
    model.half();gc.collect();tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    generation=exact_generation(model,tok);rot.write_json(directory/'software_generation.json',generation)
    name='generation_expected_token_ids_u32.bin';pack.atomic_write_bytes(root/name,np.asarray(generation['token_ids'],dtype='<u4').tobytes());record(name)
    manifest.update(experiment='EXP-0220',variant=variant,original_checkpoint_shards=hashes,
        transform='R2=H128, all original gamma retained, no R1' if variant=='r2_only' else variant,
        quantization='single_FP32_scale_per_output_channel_W4_minus7_to7',lpbq=False,
        changed_files=changed_files,inherited_replay_references='historical placeholders, not valid for new layer replay')
    manifest['generation'].update(independent_expected_token_ids=generation['token_ids'],independent_expected_text=generation['text'],semantic_reference_sha256=pack.sha256_file(directory/'software_generation.json'))
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    rot.write_json(directory/'weight_stats.json',stats)
    rot.write_json(directory/'package.json',dict(manifest_sha256=pack.sha256_file(root/'manifest.json'),changed_files=changed_files,original_shards=hashes))
    if variant=='r2_only':rot.write_json(directory/'software_quality.json',dict(samples=rot.quality(model,tok,data),role='Independent FP16 from actual packed codes/scales, not DSP score'))
    print(json.dumps(dict(variant=variant,completed=True,elapsed_s=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fp16_control']+VARIANTS);args=p.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    if args.phase=='fp16_control':fp16_control()
    else:prepare(args.phase)
