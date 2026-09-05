#!/usr/bin/env python3
"""Fixed offline R1/R2 for W4F16; no inference-time rotation operations.

Canonical OI algebra follows Dan1un1u/mllm at
2782f7ad9d1ee5e61f65d418cb89ae7217b5fb06/scripts/qwen3_block_rotation.py.
Extend its Layer-5 transform to embedding, all layers and the final head.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoTokenizer
import prepare_exp0164_generation_package as pack
from run_exp0164_semantic_gate import load_model, generate
from eval_exp0218 import dataset, grade, decode, MODEL

RESULT = Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0219')
OUTPUT = Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0219')
BASE = OUTPUT.parent/'exp0164/w4f16_greedy16'
REFERENCE = '2782f7ad9d1ee5e61f65d418cb89ae7217b5fb06'
PROJECTIONS = {'q':'self_attn.q_proj', 'k':'self_attn.k_proj',
               'v':'self_attn.v_proj', 'o':'self_attn.o_proj',
               'gate':'mlp.gate_proj', 'up':'mlp.up_proj', 'down':'mlp.down_proj'}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def fwht(value, dim=-1):
    """Normalized Sylvester butterfly, preserving input dtype and axis order."""
    n = value.shape[dim]
    if n <= 0 or n & (n-1):
        raise ValueError(f'Non-power-of-two Hadamard width {n}')
    work = value.movedim(dim, -1).contiguous()
    shape = work.shape
    for power in range(n.bit_length()-1):
        half = 1 << power
        pairs = work.reshape(-1, n//(2*half), 2, half)
        a, b = pairs[:,:,0,:], pairs[:,:,1,:]
        work = torch.stack((a+b, a-b), dim=2)
    return (work.reshape(shape)/math.sqrt(n)).movedim(-1, dim).contiguous()

def transform_layer(layer, rotate):
    gamma = layer.input_layernorm.weight.detach().clone()
    post = layer.post_attention_layernorm.weight.detach().clone()
    for name in ['q', 'k', 'v', 'gate', 'up']:
        param = layer.get_submodule(PROJECTIONS[name]).weight
        w = param * (post if name in ['gate','up'] else gamma)[None,:]
        if rotate:
            w = fwht(w, 1)
            if name == 'v':
                w = fwht(w.reshape(8,128,2048),1).reshape_as(w)
        param.copy_(w)
    if rotate:
        w = layer.self_attn.o_proj.weight
        w.copy_(fwht(fwht(w,0).reshape(2048,16,128),2).reshape_as(w))
        w = layer.mlp.down_proj.weight
        w.copy_(fwht(w,0))
    layer.input_layernorm.weight.fill_(1)
    layer.post_attention_layernorm.weight.fill_(1)

def transform(model, rotate):
    assert model.config.hidden_size == 2048 and model.config.head_dim == 128
    assert model.config.num_attention_heads == 16 and model.config.num_key_value_heads == 8
    assert not model.config.tie_word_embeddings, 'Tied head needs separate gamma handling'
    for layer in model.model.layers:
        transform_layer(layer, rotate)
    # Chunk the two vocabulary-sized matrices to bound temporary memory.
    for first in range(0, pack.VOCAB, 1024):
        last = first+1024
        if rotate:
            w = model.model.embed_tokens.weight[first:last]
            w.copy_(fwht(w,1))
        w = model.lm_head.weight[first:last]
        value = w * model.model.norm.weight[None,:]
        w.copy_(fwht(value,1) if rotate else value)
    model.model.norm.weight.fill_(1)

def algebra_test():
    torch.manual_seed(219)
    # Independently materialized dense Sylvester oracle (no exporter helper).
    h = torch.ones(1,1,dtype=torch.float64)
    for _ in range(3):
        h = torch.cat((torch.cat((h,h),1), torch.cat((h,-h),1)),0)
    h /= math.sqrt(8)
    x = torch.randn(3,8,dtype=torch.float64)
    w = torch.randn(12,8,dtype=torch.float64)
    gamma = torch.rand(8,dtype=torch.float64)
    errors = {'dense_fwht':float((fwht(x)-x@h).abs().max()),
              'orthogonal':float((h.T@h-torch.eye(8)).abs().max()),
              'gamma_fold':float((fwht(x)@fwht(w*gamma,1).T-(x*gamma)@w.T).abs().max())}
    assert max(errors.values()) <= 1e-10, errors
    return errors

def logits(model, sample):
    ids = torch.tensor([sample['prompt_ids']+sample['target_ids'][:-1]])
    hidden = model.model(ids, use_cache=False).last_hidden_state[0,63:79]
    return model.lm_head(hidden).float().clone()

def difference(actual, expected):
    a,b = actual.double().flatten(), expected.double().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    rel = float((a-b).norm()/b.norm())
    cosine = float(torch.nn.functional.cosine_similarity(a,b,dim=0))
    return dict(finite=finite, nrmse=rel, cosine=cosine,
                max_abs=float((a-b).abs().max()),
                top1_equal=int((actual.argmax(-1)==expected.argmax(-1)).sum()),
                passed=finite and rel<=0.003 and cosine>=0.99999)

def unpack(packed, n, k):
    # Inverse carrier for an independent byte-layout round trip.
    flat=np.empty((*packed.shape[:-1],1024),dtype=np.int8)
    flat[...,0::2]=(packed&15).astype(np.int8)
    flat[...,1::2]=(packed>>4).astype(np.int8)
    flat[flat>=8]-=16
    return torch.from_numpy(flat.reshape(n//32,k//32,8,32,4)
        .transpose(0,1,2,4,3).reshape(n//32,k//32,32,32)
        .transpose(0,3,1,2).copy().reshape(n,k))

def export_quantized(param, root, prefix, stats):
    carrier = root/(prefix+'_weight_w4_hmx.bin')
    scales = root/(prefix+'_weight_w4_scale_f32.bin')
    # Files inherited from the frozen baseline may be hard links: replace atomically.
    cp,sp = carrier.with_suffix('.new'), scales.with_suffix('.new')
    error2=energy=0.; peak_ratio=[]
    with cp.open('xb') as c, sp.open('xb') as s:
        for first in range(0,param.shape[0],1024):
            w=param[first:first+1024].float()
            packed,scale=pack.pack_w4_chunk(w)
            q=unpack(packed,*w.shape).float()
            expected=torch.round(w/torch.from_numpy(scale)[:,None]).clamp(-7,7)
            assert torch.equal(q,expected)
            dequant=q*torch.from_numpy(scale)[:,None]
            error2+=float((w-dequant).double().square().sum())
            energy+=float(w.double().square().sum())
            peak_ratio.extend((w.abs().amax(1)/w.square().mean(1).sqrt().clamp_min(1e-30)).tolist())
            c.write(packed.tobytes());s.write(scale.tobytes())
            # Software diagnostic consumes precisely the same codes/scales as DSP.
            param[first:first+1024].copy_(dequant.half().float())
    cp.replace(carrier);sp.replace(scales)
    stats[prefix]=dict(relative_weight_rmse=math.sqrt(error2/max(energy,1e-30)),
        mean_row_peak_over_rms=float(np.mean(peak_ratio)),packed_roundtrip=True)

def quality(model,tok,data):
    results=[]
    for sample in data['samples']:
        if sample['split']!='full':continue
        r=dict(id=sample['id'],kind=sample['kind'],language=sample['language'])
        if sample['kind']=='nll':
            value=logits(model,sample)
            target=torch.tensor(sample['target_ids']);v=value[torch.arange(16),target]
            r.update(nll=(torch.logsumexp(value,-1)-v).tolist(),top1=value.argmax(-1).tolist())
        else:
            ids=torch.tensor([sample['prompt_ids']])
            out=generate(model,tok,ids,torch.ones_like(ids),16)
            text=decode(tok,out['token_ids']);r.update(token_ids=out['token_ids'],text=text)
            if sample['kind']=='task':r['correct']=grade(sample,text)
        results.append(r)
        print(json.dumps(dict(software_sample=sample['id'])),flush=True)
    return results

def prepare(variant):
    data=dataset(); root=OUTPUT/variant; result=RESULT/variant
    assert not root.exists() and not result.exists(), 'Refusing to overwrite EXP-0219 evidence'
    result.mkdir(parents=True); start=time.monotonic()
    algebra=algebra_test()
    model=load_model(MODEL,torch.float32)
    probes=[next(s for s in data['samples'] if s['id']==i) for i in [0,20]]
    original=[logits(model,s) for s in probes]
    transform(model,variant=='C')
    checks=[dict(sample=s['id'],**difference(logits(model,s),ref)) for s,ref in zip(probes,original)]
    write_json(result/'invariance.json',dict(algebra=algebra,full_model_fp32=checks,reference_commit=REFERENCE))
    assert all(c['passed'] for c in checks),checks
    print(json.dumps(dict(variant=variant,invariance=checks)),flush=True)
    root.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(BASE,root,copy_function=pack.copy_link)
    stats={}
    for i,layer in enumerate(model.model.layers):
        for short,long in PROJECTIONS.items():
            sub={};export_quantized(layer.get_submodule(long).weight,root/f'layer{i}',short,sub)
            stats[f'layer{i}/{short}']=sub[short]
        for short,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            pack.atomic_write_bytes(root/f'layer{i}/{short}_norm_weight_f16.bin',
                layer.get_submodule(long).weight.half().numpy().tobytes())
        print(json.dumps(dict(variant=variant,exported_layer=i)),flush=True)
    export_quantized(model.lm_head.weight,root,'generation_lm_head',stats)
    pack.atomic_write_bytes(root/'generation_embedding_weight_f16.bin',model.model.embed_tokens.weight.half().numpy().tobytes())
    pack.atomic_write_bytes(root/'generation_final_norm_weight_f16.bin',model.model.norm.weight.half().numpy().tobytes())
    model.half();gc.collect()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=torch.tensor([np.fromfile(BASE/'generation_prompt_token_ids_u32.bin',dtype='<u4').tolist()])
    generation=generate(model,tok,ids,torch.ones_like(ids),16)
    assert len(generation['token_ids'])==16
    write_json(result/'software_generation.json',generation)
    pack.atomic_write_bytes(root/'generation_expected_token_ids_u32.bin',np.asarray(generation['token_ids'],dtype='<u4').tobytes())
    write_json(result/'weight_stats.json',stats)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest.update(experiment='EXP-0219',variant=variant,rotation='identity_fold' if variant=='B' else 'H2048_R1_H128_R2',
        rotation_reference_commit=REFERENCE,inherited_non_generation_replay_references='historical only; invalid for rotated replay; generation and eval modes only')
    manifest['generation'].update(independent_expected_token_ids=generation['token_ids'],independent_expected_text=generation['text'],semantic_reference_sha256=pack.sha256_file(result/'software_generation.json'))
    manifest['files']={str(p.relative_to(root)):pack.file_record(p) for p in sorted(root.rglob('*.bin'))}
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    write_json(result/'package.json',dict(package=str(root),manifest_sha256=pack.sha256_file(root/'manifest.json'),
        source_index=pack.file_record(MODEL/'model.safetensors.index.json'),elapsed_s=time.monotonic()-start))
    write_json(result/'software_quality.json',dict(role='independent FP16 diagnostic from actual packed W4; not DSP score',
        samples=quality(model,tok,data),dataset_sha256=pack.sha256_file(RESULT.parent/'exp0218/dataset_v1.json')))
    print(json.dumps(dict(variant=variant,completed=True,elapsed_s=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['B','C']);args=p.parse_args()
    torch.set_num_threads(16);torch.set_grad_enabled(False)
    prepare(args.variant)
