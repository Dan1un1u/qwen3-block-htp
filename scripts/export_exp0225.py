#!/usr/bin/env python3
"""True-sequential GPTQ A/B/C export with unchanged W4F16 DSP runtime."""
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
from experiment_exp0220 import BASE,verify_origin,exact_generation,metrics
from run_exp0164_semantic_gate import load_model
from gptq_exp0221 import factor,quantize,pack_codes,sha
from learned_rotation_exp0225 import RESULT,OUTPUT,PLAN,fold,orthogonality
from output_scale_exp0224 import select_output
from experiment_exp0221 import export as export_head
from experiment_exp0220 import load_packed
from pathlib import Path

def export(w,x,f,root,prefix):
    started=time.monotonic();n,k=w.shape;dequant=torch.empty((n,k),dtype=torch.float16)
    probe=x.reshape(-1,k)[::16].float();rtn_sse=gptq_sse=energy=0.;endpoints=0;row_stats=[]
    cp=root/(prefix+'_weight_w4_hmx.new');sp=root/(prefix+'_weight_w4_scale_f32.new')
    with cp.open('xb') as out,sp.open('xb') as scales:
        for first in range(0,n,1024):
            original=w[first:first+1024].float();codes,scale,rs=select_output(original,x,f);row_stats.append(rs)
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
    return dequant,dict(clip_stats_path=str(record),clip_stats_sha256=pack.sha256_file(record),clip_choice_histogram=np.bincount(rows['choice'],minlength=3).tolist(),clipped_rows=int((rows['scale']<rows['absmax_scale']).sum()),output_sse_by_candidate=rows['candidate_output_sse'].sum(0).tolist(),selected_output_sse=float(rows['selected_output_sse'].sum()),selection_positions=x.numel()//k,rows=n,**f['stats'],rtn_probe_nrmse=(rtn_sse/energy)**.5,gptq_probe_nrmse=(gptq_sse/energy)**.5,
        packed_roundtrip=True,endpoint_fraction=endpoints/(n*k),elapsed_s=time.monotonic()-started,
        grid='3_fixed_final_GPTQ_output_ranges_signed_minus7_to7',groupsize=-1,act_order=True)

def prepare(v):
    rotation_checkpoint=OUTPUT/'training'/f'{v}.pt'
    state=torch.load(rotation_checkpoint,map_location='cpu',weights_only=False)
    assert state['plan']==PLAN and state['data_sha256']==ev.digest(RESULT/'learning_data.json')
    rotations=state['rotations'];assert max(orthogonality(rotations).values())<.003
    root=OUTPUT/v;result=RESULT/v;assert not root.exists() and not result.exists()
    assert (RESULT/'algebra_oracle.json').exists() and (RESULT/'output_scale_oracle.json').exists()
    calpath=RESULT.parent/'exp0221/calibration.json';assert ev.digest(calpath)=='e65edb14cb774956df92b27b5dc728b976f7ba00e9435145004180c6538a1669'
    cal=json.loads(calpath.read_text());raw=(OUTPUT.parent/'exp0221/calibration_ids_u32.bin').read_bytes()
    assert sha(raw)==cal['ids_sha256'];ids=torch.from_numpy(np.frombuffer(raw,dtype='<u4').astype(np.int64).reshape(64,128))
    origin=verify_origin();result.mkdir(parents=True);started=time.monotonic()
    model=load_model(ev.MODEL,torch.float32)
    learning=json.loads((RESULT/'learning_data.json').read_text())
    probes=[dict(id=i,prompt_ids=learning['train'][i]['token_ids'][:64],target_ids=learning['train'][i]['token_ids'][64:80]) for i in [0,400]]
    refs=[rot.logits(model,s) for s in probes]
    fold(model,rotations)
    invariance=[dict(sample=s['id'],**rot.difference(rot.logits(model,s),ref)) for s,ref in zip(probes,refs)]
    rot.write_json(result/'invariance.json',dict(checks=invariance,original_shards=origin));assert all(d['passed'] for d in invariance)
    shutil.copytree(BASE,root,copy_function=pack.copy_link)
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
            name=f'layer{i}/{short}_norm_weight_f16.bin';pack.atomic_write_bytes(root/name,layer.get_submodule(long).weight.numpy().tobytes());changed.append(name)
        checkpoint=OUTPUT/'checkpoints'/v;checkpoint.mkdir(parents=True,exist_ok=True)
        np.save(checkpoint/f'layer{i}_hidden.npy',hidden.numpy())
        rot.write_json(result/f'layer{i}_stats.json',{k:val for k,val in stats.items() if k.startswith(f'layer{i}/')})
        print(json.dumps(dict(variant=v,completed_layer=i,elapsed_s=time.monotonic()-started)),flush=True)
    model.model.norm.half();head_input=model.model.norm(hidden);f=factor(head_input)
    value,st=export_head(model.lm_head.weight,head_input,f,root,'generation_lm_head')
    model.lm_head.weight=torch.nn.Parameter(value);stats['head']=st;record_weight('generation_lm_head')
    del hidden,head_input,f;gc.collect()
    for name,value in [('generation_embedding_weight_f16.bin',model.model.embed_tokens.weight),('generation_final_norm_weight_f16.bin',model.model.norm.weight)]:
        pack.atomic_write_bytes(root/name,value.half().numpy().tobytes());changed.append(name)
    model.half();tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    generation=exact_generation(model,tok);rot.write_json(result/'software_generation.json',generation)
    name='generation_expected_token_ids_u32.bin';pack.atomic_write_bytes(root/name,np.asarray(generation['token_ids'],dtype='<u4').tobytes());changed.append(name)
    manifest=json.loads((BASE/'manifest.json').read_text());old=manifest['files']
    manifest['files']={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),val) for n,val in old.items()}
    for n in changed:manifest['files'][n]=pack.file_record(root/n)
    manifest.update(experiment='EXP-0225',variant=v,transform='learned_offline_R1R2_fresh_original',rotation_checkpoint_sha256=ev.digest(rotation_checkpoint),
        quantizer='EXP0224_transformer_output_scale_GPTQ_absmax_head',calibration_sha256=ev.digest(calpath),
        original_shards=origin,changed_files=changed,inherited_replay_references='historical placeholders; not valid for new layer replay')
    manifest['generation'].update(independent_expected_token_ids=generation['token_ids'],independent_expected_text=generation['text'],semantic_reference_sha256=ev.digest(result/'software_generation.json'))
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    rot.write_json(result/'weight_stats.json',stats);rot.write_json(result/'calibration_forward_checks.json',calibration_checks)
    rot.write_json(result/'package.json',dict(manifest_sha256=ev.digest(root/'manifest.json'),changed_files=changed,original_shards=origin,elapsed_s=time.monotonic()-started))
    validate_model(model,v)
    print(json.dumps(dict(variant=v,completed=True,elapsed_s=time.monotonic()-started)),flush=True)


@torch.no_grad()
def validate_model(model,v):
    data=json.loads((RESULT/'learning_data.json').read_text());scores=[]
    for s in data['validation']:
        ids=torch.tensor([s['token_ids']]);h=model.model(ids[:,:-1],use_cache=False).last_hidden_state[0]
        # Chunk positions to bound vocabulary logits, unchanged CE definition.
        losses=[]
        for start in range(0,len(h),16):
            logits=model.lm_head(h[start:start+16]).float();target=ids[0,1+start:1+start+len(logits)]
            losses.extend(F.cross_entropy(logits,target,reduction='none').tolist())
        scores.append(dict(language=s['language'],row_index=s['row_index'],nll=float(np.mean(losses)),tokens=len(losses)))
    bylang={lang:float(np.mean([r['nll'] for r in scores if r['language']==lang])) for lang in ['en','zh']}
    rot.write_json(RESULT/v/'validation.json',dict(samples=scores,nll=float(np.mean([r['nll'] for r in scores])),language_nll=bylang,dataset_sha256=ev.digest(RESULT/'learning_data.json'),role='actual packed GPTQ software checkpoint selection; no qbh evaluation'))

def load_package(root):
    model=load_model(ev.MODEL,torch.float16)
    if model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone());model.config.tie_word_embeddings=False
    manifest=json.loads((root/'manifest.json').read_text())
    for i,layer in enumerate(model.model.layers):
        for short,long in rot.PROJECTIONS.items():load_packed(layer.get_submodule(long).weight,root/f'layer{i}',short)
        for short,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            value=np.fromfile(root/f'layer{i}/{short}_norm_weight_f16.bin',dtype='<f2');layer.get_submodule(long).weight.copy_(torch.from_numpy(value))
    load_packed(model.lm_head.weight,root,'generation_lm_head')
    for name,param in [('generation_embedding_weight_f16.bin',model.model.embed_tokens.weight),('generation_final_norm_weight_f16.bin',model.model.norm.weight)]:
        value=np.fromfile(root/name,dtype='<f2').reshape(param.shape);param.copy_(torch.from_numpy(value))
    return model

def select():
    rows={}
    for step in PLAN['checkpoints']:
        v=f'step{step:03d}';rows[v]=json.loads((RESULT/v/'validation.json').read_text())
    chosen=min(rows,key=lambda k:(rows[k]['nll'],int(k[4:])))
    rot.write_json(RESULT/'selection.json',dict(selected=chosen,rule=PLAN['checkpoint_selection'],scores={k:dict(nll=v['nll'],language_nll=v['language_nll']) for k,v in rows.items()},evaluation_used=False))
    print('SELECTED',chosen)

def final_quality():
    selected=json.loads((RESULT/'selection.json').read_text())['selected'];model=load_package(OUTPUT/selected)
    tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    rot.write_json(RESULT/selected/'software_quality.json',dict(samples=rot.quality(model,tok,ev.dataset()),role='selected actual packed W4 software; DSP scores separate'))

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['step000','step050','step100','control','select','quality']);a=p.parse_args()
    if a.phase=='select':select()
    elif a.phase=='quality':final_quality()
    elif a.phase=='control':
        (RESULT/'control_A').mkdir(exist_ok=True);validate_model(load_package(OUTPUT.parent/'exp0224/A'),'control_A')
    else:prepare(a.phase)
