#!/usr/bin/env python3
"""Unchanged CPU EXP224 quantizer; calibration coverage/budget is the intervention."""
import argparse,gc,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import experiment_exp0224 as old
import rotation_exp0219 as rot
import prepare_exp0164_generation_package as pack
from experiment_exp0220 import verify_origin,exact_generation,metrics,load_packed
from run_exp0164_semantic_gate import load_model
from gptq_exp0221 import factor
from data_exp0230 import RESULT,OUTPUT,MODEL,SOURCE,sha,write,preflight,verified

BASE_PACKAGE=OUTPUT.parent/'exp0224/A'
BASE_HASH='a5de4e6c4e02ac913e69fbddb0d4b0b9e12b5cfe88ff606cf1ea18842dc0c179'

def frozen():
    assert sha(RESULT/'dataset_freeze.json')=='fb99996e90f3b041ff3ab792b6ab41a187b20ef3ac9060f392b297fc00575831'
    f=json.loads((RESULT/'dataset_freeze.json').read_text())
    for n,h in f['files'].items():assert sha(RESULT/n)==h,n
    assert json.loads((RESULT/'independent_data_audit.json').read_text())['pass_all']
    assert sha(BASE_PACKAGE/'manifest.json')==BASE_HASH
    return json.loads((RESULT/'dataset.json').read_text())

def prepare(v):
    preflight();data=frozen();root=OUTPUT/v;result=RESULT/v
    assert not root.exists() and not result.exists(),('preserve partial attempt',v)
    original_ledger=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
    for n,h in original_ledger.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h)
    old.OUTPUT=OUTPUT;old.RESULT=RESULT
    rows=[r for r in data['samples'] if r['split']=='calibration' and (v=='C64' or r['in_C8'])]
    ids=torch.tensor([r['token_ids'] for r in rows],dtype=torch.long)
    assert ids.numel()==(8192 if v=='C8' else 65536) and ids.shape[1]==128
    raw=(RESULT/'inputs'/(v+'_calibration_u32.bin')).read_bytes()
    assert np.array_equal(np.frombuffer(raw,dtype='<u4').reshape(-1,128),ids.numpy())
    origin=verify_origin();result.mkdir(parents=True);started=time.monotonic()
    model=load_model(MODEL,torch.float32)
    if model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings=False
    shutil.copytree(BASE_PACKAGE,root,copy_function=pack.copy_link)
    model.model.embed_tokens.half();hidden=model.model.embed_tokens(ids)
    position=torch.arange(128).unsqueeze(0);rope=model.model.rotary_emb(hidden,position)
    mask=torch.full((128,128),torch.finfo(torch.float16).min,dtype=torch.float16).triu(1)[None,None]
    stats={};checks=[];changed=[]
    def record_weight(name):changed.extend([name+'_weight_w4_hmx.bin',name+'_weight_w4_scale_f32.bin'])
    for i,layer in enumerate(model.model.layers):
        original={name:layer.get_submodule(long).weight.detach().clone() for name,long in rot.PROJECTIONS.items()}
        layer.half();incoming=hidden;normed=layer.input_layernorm(incoming);f=factor(normed)
        for name in ['q','k','v']:
            value,st=old.export(original[name],normed,f,root/f'layer{i}',name)
            layer.get_submodule(rot.PROJECTIONS[name]).weight.copy_(value)
            stats[f'layer{i}/{name}']=st;record_weight(f'layer{i}/{name}')
        del f,normed
        o=layer.self_attn.o_proj;layer.self_attn.o_proj=torch.nn.Identity()
        try:attention_input=layer.self_attn(layer.input_layernorm(incoming),position_embeddings=rope,attention_mask=mask)[0]
        finally:layer.self_attn.o_proj=o
        f=factor(attention_input);value,st=old.export(original['o'],attention_input,f,root/f'layer{i}','o')
        o.weight.copy_(value);stats[f'layer{i}/o']=st;record_weight(f'layer{i}/o');del f
        residual=incoming+o(attention_input);del attention_input
        normed=layer.post_attention_layernorm(residual);f=factor(normed)
        for name in ['gate','up']:
            value,st=old.export(original[name],normed,f,root/f'layer{i}',name)
            layer.get_submodule(rot.PROJECTIONS[name]).weight.copy_(value)
            stats[f'layer{i}/{name}']=st;record_weight(f'layer{i}/{name}')
        del f
        down_input=layer.mlp.act_fn(layer.mlp.gate_proj(normed))*layer.mlp.up_proj(normed);del normed
        f=factor(down_input);value,st=old.export(original['down'],down_input,f,root/f'layer{i}','down')
        layer.mlp.down_proj.weight.copy_(value);stats[f'layer{i}/down']=st;record_weight(f'layer{i}/down')
        hidden=residual+layer.mlp.down_proj(down_input);del f,down_input,residual,original,value
        for sample in [0,1,2,3]:
            reference=layer(incoming[sample:sample+1],attention_mask=mask,position_embeddings=rope)[0]
            check=dict(layer=i,sample=sample,**metrics(hidden[sample:sample+1],reference))
            assert check['finite'] and check['nrmse']<=.003 and check['cosine']>=.99999,check
            checks.append(check)
        del incoming
        for short,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            assert (root/f'layer{i}/{short}_norm_weight_f16.bin').read_bytes()==layer.get_submodule(long).weight.numpy().tobytes()
        checkpoint=OUTPUT/'checkpoints'/v;checkpoint.mkdir(parents=True,exist_ok=True)
        np.save(checkpoint/f'layer{i}_hidden.npy',hidden.numpy())
        write(f'{v}/layer{i}_stats.json',{k:val for k,val in stats.items() if k.startswith(f'layer{i}/')})
        print('QUANTIZED_LAYER',v,i,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    model.model.norm.half();load_packed(model.lm_head.weight,BASE_PACKAGE,'generation_lm_head')
    del hidden;gc.collect();model.half()
    for n,value in [('generation_embedding_weight_f16.bin',model.model.embed_tokens.weight),
                    ('generation_final_norm_weight_f16.bin',model.model.norm.weight)]:
        assert (root/n).read_bytes()==value.numpy().tobytes()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);generation=exact_generation(model,tok)
    write(f'{v}/software_generation.json',generation)
    name='generation_expected_token_ids_u32.bin'
    pack.atomic_write_bytes(root/name,np.asarray(generation['token_ids'],dtype='<u4').tobytes());changed.append(name)
    manifest=json.loads((BASE_PACKAGE/'manifest.json').read_text());prior=manifest['files']
    manifest['files']={n.replace(chr(92),'/'):prior.get(n.replace(chr(92),'/'),x) for n,x in prior.items()}
    for n in changed:manifest['files'][n]=pack.file_record(root/n)
    manifest.update(experiment='EXP-0230',variant=v,calibration_sha256=sha(RESULT/'inputs'/(v+'_calibration_u32.bin')),
        calibration_tokens=ids.numel(),changed_files=changed,original_shards=origin,
        quantizer='unchanged_EXP0224_CPU_GPTQ_final_output_3range_per_row_signed_minus7_to7_no_groups',
        frozen_head_from='EXP0221_A',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip())
    manifest['generation'].update(independent_expected_token_ids=generation['token_ids'],
        independent_expected_text=generation['text'],semantic_reference_sha256=sha(RESULT/v/'software_generation.json'))
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    frozen_files={n:d['sha256'] for n,d in manifest['files'].items() if n not in changed}
    assert all(sha(root/n)==h for n,h in frozen_files.items())
    write(f'{v}/frozen_files.json',frozen_files);write(f'{v}/weight_stats.json',stats)
    write(f'{v}/calibration_forward_checks.json',checks)
    write(f'{v}/package.json',dict(manifest_sha256=sha(root/'manifest.json'),changed_files=changed,
        original_shards=origin,elapsed_s=time.monotonic()-started,calibration_tokens=ids.numel(),
        quantizer_source={n:sha(SOURCE/'scripts'/n) for n in ['gptq_exp0221.py','output_scale_exp0224.py','experiment_exp0224.py']},
        control_manifest_sha256=BASE_HASH,source_head=manifest['source_head']))
    print('PACKAGE_COMPLETE',v,round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['C8','C64']);a=p.parse_args();prepare(a.variant)
