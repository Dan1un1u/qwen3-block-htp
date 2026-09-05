#!/usr/bin/env python3
"""Transformer-only clipped GPTQ A/B/C export with unchanged W4F16 DSP runtime."""
import argparse
import gc
import json
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import eval_exp0218 as ev
import rotation_exp0219 as rot
import prepare_exp0164_generation_package as pack
from experiment_exp0220 import BASE,verify_origin,exact_generation,metrics,load_packed
from run_exp0164_semantic_gate import load_model
from gptq_exp0221 import factor,quantize,pack_codes,sha
from clipping_exp0223 import RESULT,OUTPUT,PREVIOUS,PREVIOUS_MODELS,select_scale

VARIANTS=['A','B','C']

def export(w,x,f,root,prefix):
    started=time.monotonic();n,k=w.shape;dequant=torch.empty((n,k),dtype=torch.float16)
    probe=x.reshape(-1,k)[::16].float();rtn_sse=gptq_sse=energy=0.;endpoints=0;row_stats=[]
    cp=root/(prefix+'_weight_w4_hmx.new');sp=root/(prefix+'_weight_w4_scale_f32.new')
    with cp.open('xb') as out,sp.open('xb') as scales:
        for first in range(0,n,1024):
            original=w[first:first+1024].float();scale,rs=select_scale(original);row_stats.append(rs)
            codes,scale=quantize(original,f,explicit_scale=scale)
            assert torch.isfinite(scale).all() and (scale>0).all()
            packed=pack_codes(codes);q=codes.float()*scale[:,None]
            assert torch.isfinite(q.half()).all()
            out.write(packed.tobytes());scales.write(scale.numpy().astype('<f4').tobytes())
            dequant[first:first+1024]=q.half();endpoints+=int((codes.abs()==7).sum())
            target=F.linear(probe,original);actual=F.linear(probe,q.half().float())
            rtn=((original/scale[:,None]).round().clamp(-7,7)*scale[:,None]).half().float()
            rtn_sse+=float((F.linear(probe,rtn)-target).double().square().sum())
            gptq_sse+=float((actual-target).double().square().sum());energy+=float(target.double().square().sum())
    cp.replace(root/(prefix+'_weight_w4_hmx.bin'));sp.replace(root/(prefix+'_weight_w4_scale_f32.bin'))
    relative=root.relative_to(OUTPUT);record=OUTPUT/'clip_stats'/relative/(prefix+'.npz');record.parent.mkdir(parents=True,exist_ok=True)
    rows={k:np.concatenate([r[k] for r in row_stats]) for k in row_stats[0]};np.savez_compressed(record,**rows)
    return dequant,dict(clip_stats_path=str(record),clip_stats_sha256=pack.sha256_file(record),clip_choice_histogram=np.bincount(rows['choice'],minlength=80).tolist(),clipped_rows=int((rows['choice']>0).sum()),rows=n,**f['stats'],rtn_probe_nrmse=(rtn_sse/energy)**.5,gptq_probe_nrmse=(gptq_sse/energy)**.5,
        packed_roundtrip=True,endpoint_fraction=endpoints/(n*k),elapsed_s=time.monotonic()-started,
        grid='80_fixed_ranges_row_L2.4_then_GPTQ_signed_minus7_to7',groupsize=-1,act_order=True)

def prepare(v):
    root=OUTPUT/v;result=RESULT/v;assert not root.exists() and not result.exists()
    assert (RESULT/'clipping_oracle.json').exists()
    assert pack.sha256_file(PREVIOUS/'calibration.json')=='e65edb14cb774956df92b27b5dc728b976f7ba00e9435145004180c6538a1669'
    base=PREVIOUS_MODELS/v
    closure=json.loads((PREVIOUS/'closure.json').read_text());assert pack.sha256_file(PREVIOUS/'closure.json')=='0b0b183eeca9b96c38a2e78454e2c68a7e1442579c8ff8d6f420918dc2ca489a'
    assert pack.sha256_file(base/'manifest.json')==closure['packages'][v]['manifest_sha256']
    cal=json.loads((PREVIOUS/'calibration.json').read_text());raw=(PREVIOUS_MODELS/'calibration_ids_u32.bin').read_bytes()
    assert sha(raw)==cal['ids_sha256'];ids=torch.from_numpy(np.frombuffer(raw,dtype='<u4').astype(np.int64).reshape(64,128))
    origin=verify_origin();result.mkdir(parents=True);started=time.monotonic()
    model=load_model(ev.MODEL,torch.float32);data=ev.dataset()
    probes=[next(s for s in data['samples'] if s['id']==i) for i in [0,20]]
    refs=[rot.logits(model,s) for s in probes]
    if v!='A':rot.transform(model,v=='C')
    elif model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone());model.config.tie_word_embeddings=False
    invariance=[dict(sample=s['id'],**rot.difference(rot.logits(model,s),ref)) for s,ref in zip(probes,refs)]
    rot.write_json(result/'invariance.json',dict(checks=invariance,original_shards=origin));assert all(d['passed'] for d in invariance)
    shutil.copytree(base,root,copy_function=pack.copy_link)
    model.model.embed_tokens.half();hidden=model.model.embed_tokens(ids)
    position=torch.arange(128).unsqueeze(0);rope=model.model.rotary_emb(hidden,position)
    mask=torch.full((128,128),torch.finfo(torch.float16).min,dtype=torch.float16).triu(1)[None,None]
    stats={};calibration_checks=[];changed=[]
    def record_weight(name):
        changed.extend([name+'_weight_w4_hmx.bin',name+'_weight_w4_scale_f32.bin'])
    for i,layer in enumerate(model.model.layers):
        original={name:layer.get_submodule(long).weight.detach().clone() for name,long in rot.PROJECTIONS.items()}
        layer.half();incoming=hidden;normed=layer.input_layernorm(incoming)
        f=factor(normed)
        for name in ['q','k','v']:
            value,st=export(original[name],normed,f,root/f'layer{i}',name)
            layer.get_submodule(rot.PROJECTIONS[name]).weight.copy_(value);stats[f'layer{i}/{name}']=st;record_weight(f'layer{i}/{name}')
        del f,normed
        o=layer.self_attn.o_proj;layer.self_attn.o_proj=torch.nn.Identity()
        try:attention_input=layer.self_attn(layer.input_layernorm(incoming),position_embeddings=rope,attention_mask=mask)[0]
        finally:layer.self_attn.o_proj=o
        f=factor(attention_input);value,st=export(original['o'],attention_input,f,root/f'layer{i}','o')
        o.weight.copy_(value);stats[f'layer{i}/o']=st;record_weight(f'layer{i}/o');del f
        residual=incoming+o(attention_input);del attention_input
        normed=layer.post_attention_layernorm(residual);f=factor(normed)
        for name in ['gate','up']:
            value,st=export(original[name],normed,f,root/f'layer{i}',name)
            layer.get_submodule(rot.PROJECTIONS[name]).weight.copy_(value);stats[f'layer{i}/{name}']=st;record_weight(f'layer{i}/{name}')
        del f
        down_input=layer.mlp.act_fn(layer.mlp.gate_proj(normed))*layer.mlp.up_proj(normed);del normed
        f=factor(down_input);value,st=export(original['down'],down_input,f,root/f'layer{i}','down')
        layer.mlp.down_proj.weight.copy_(value);stats[f'layer{i}/down']=st;record_weight(f'layer{i}/down')
        hidden=residual+layer.mlp.down_proj(down_input)
        del f,down_input,residual,original,value
        # Independent unchanged HF layer call checks the staged calibration path.
        for sample in [0,32]:
            reference=layer(incoming[sample:sample+1],attention_mask=mask,position_embeddings=rope)[0]
            check=dict(layer=i,sample=sample,**metrics(hidden[sample:sample+1],reference))
            assert check['finite'] and check['nrmse']<=.003 and check['cosine']>=.99999,check
            calibration_checks.append(check)
        del incoming
        for short,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            name=f'layer{i}/{short}_norm_weight_f16.bin';assert (root/name).read_bytes()==layer.get_submodule(long).weight.numpy().tobytes()
        checkpoint=OUTPUT/'checkpoints'/v;checkpoint.mkdir(parents=True,exist_ok=True)
        np.save(checkpoint/f'layer{i}_hidden.npy',hidden.numpy())
        rot.write_json(result/f'layer{i}_stats.json',{k:val for k,val in stats.items() if k.startswith(f'layer{i}/')})
        print(json.dumps(dict(variant=v,completed_layer=i,elapsed_s=time.monotonic()-started)),flush=True)
    model.model.norm.half();load_packed(model.lm_head.weight,base,'generation_lm_head')
    del hidden;gc.collect()
    for name,value in [('generation_embedding_weight_f16.bin',model.model.embed_tokens.weight),('generation_final_norm_weight_f16.bin',model.model.norm.weight)]:
        assert (root/name).read_bytes()==value.half().numpy().tobytes()
    model.half();tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    generation=exact_generation(model,tok);rot.write_json(result/'software_generation.json',generation)
    name='generation_expected_token_ids_u32.bin';pack.atomic_write_bytes(root/name,np.asarray(generation['token_ids'],dtype='<u4').tobytes());changed.append(name)
    manifest=json.loads((base/'manifest.json').read_text());old=manifest['files']
    manifest['files']={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),val) for n,val in old.items()}
    for n in changed:manifest['files'][n]=pack.file_record(root/n)
    manifest.update(experiment='EXP-0223',variant=v,transform={'A':'original','B':'fresh_gamma_fold','C':'fresh_gamma_fold_H2048_R1_H128_R2'}[v],
        quantizer='GPTQ_row_L2.4_80range_clipping_signed_minus7_to7_no_groups',calibration_sha256=ev.digest(PREVIOUS/'calibration.json'),frozen_head_from='EXP0221_'+v,
        original_shards=origin,changed_files=changed,inherited_replay_references='historical placeholders; not valid for new layer replay')
    manifest['generation'].update(independent_expected_token_ids=generation['token_ids'],independent_expected_text=generation['text'],semantic_reference_sha256=ev.digest(result/'software_generation.json'))
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    frozen={n:d['sha256'] for n,d in manifest['files'].items() if n not in changed}
    assert all(pack.sha256_file(root/n)==h for n,h in frozen.items())
    rot.write_json(result/'frozen_files.json',frozen)
    rot.write_json(result/'weight_stats.json',stats);rot.write_json(result/'calibration_forward_checks.json',calibration_checks)
    rot.write_json(result/'package.json',dict(manifest_sha256=ev.digest(root/'manifest.json'),changed_files=changed,original_shards=origin,elapsed_s=time.monotonic()-started))
    rot.write_json(result/'software_quality.json',dict(samples=rot.quality(model,tok,data),role='FP16 software using exported GPTQ codes/scales; not exact DSP'))
    print(json.dumps(dict(variant=v,completed=True,elapsed_s=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    p=argparse.ArgumentParser();p.add_argument('variant',choices=VARIANTS);a=p.parse_args();prepare(a.variant)
