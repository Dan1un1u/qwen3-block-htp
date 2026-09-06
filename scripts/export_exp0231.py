#!/usr/bin/env python3
"""Single software-only group128 candidate, fresh originals and frozen C8 calibration."""
import argparse,gc,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import group_exp0231 as group
import rotation_exp0219 as rot
import prepare_exp0164_generation_package as pack
from experiment_exp0220 import verify_origin,exact_generation,metrics,load_packed
from run_exp0164_semantic_gate import load_model
from gptq_exp0221 import factor
from data_exp0231 import RESULT,OUTPUT,MODEL,SOURCE,sha,write,preflight,verified

BASE_PACKAGE=OUTPUT.parent/'exp0224/A'
BASE_HASH='a5de4e6c4e02ac913e69fbddb0d4b0b9e12b5cfe88ff606cf1ea18842dc0c179'

def frozen():
    assert sha(RESULT/'dataset_freeze.json')=='6b987f5e8f7ce0e5704f3a59529f7814b859ccbc29b8906f64e7b2288c7666e0'
    f=json.loads((RESULT/'dataset_freeze.json').read_text())
    for n,h in f['files'].items():assert sha(RESULT/n)==h,n
    assert sha(Path('/home/daniuniu/work/qwen3-block-htp-project-memory/docs/experiments/EXP-0231.md'))==f['protocol_sha256']
    assert json.loads((RESULT/'independent_data_audit.json').read_text())['pass_all']
    assert sha(BASE_PACKAGE/'manifest.json')==BASE_HASH
    return json.loads((RESULT/'dataset.json').read_text())

def prepare(v):
    preflight();frozen();data=json.loads(verified('exp0230','dataset.json').read_text());v='G128';root=OUTPUT/v;result=RESULT/v
    assert json.loads((RESULT/'group_oracle.json').read_text())['pass_all']
    assert not root.exists() and not result.exists(),('preserve partial attempt',v)
    original_ledger=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
    for n,h in original_ledger.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h)
    rows=[r for r in data['samples'] if r['split']=='calibration' and r['in_C8']]
    ids=torch.tensor([r['token_ids'] for r in rows],dtype=torch.long)
    assert ids.numel()==8192 and ids.shape[1]==128
    raw=verified('exp0230','inputs/C8_calibration_u32.bin').read_bytes()
    assert np.array_equal(np.frombuffer(raw,dtype='<u4').reshape(-1,128),ids.numpy())
    origin=verify_origin();result.mkdir(parents=True);started=time.monotonic()
    model=load_model(MODEL,torch.float32)
    if model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings=False
    root.mkdir(parents=True)
    (root/'SOFTWARE_ONLY_GROUP128.txt').write_text(group.FORMAT+'\nNot an ordinary device package.\n')
    model.model.embed_tokens.half();hidden=model.model.embed_tokens(ids)
    position=torch.arange(128).unsqueeze(0);rope=model.model.rotary_emb(hidden,position)
    mask=torch.full((128,128),torch.finfo(torch.float16).min,dtype=torch.float16).triu(1)[None,None]
    stats={};checks=[];changed=[]
    def record_weight(name):changed.extend([name+'_group128_codes_hmx.bin',name+'_group128_scales_f32.bin'])
    for i,layer in enumerate(model.model.layers):
        original={name:layer.get_submodule(long).weight.detach().clone() for name,long in rot.PROJECTIONS.items()}
        layer.half();incoming=hidden;normed=layer.input_layernorm(incoming);f=factor(normed)
        for name in ['q','k','v']:
            value,st=group.export(original[name],normed,f,root/f'layer{i}',name)
            layer.get_submodule(rot.PROJECTIONS[name]).weight.copy_(value)
            stats[f'layer{i}/{name}']=st;record_weight(f'layer{i}/{name}')
        del f,normed
        o=layer.self_attn.o_proj;layer.self_attn.o_proj=torch.nn.Identity()
        try:attention_input=layer.self_attn(layer.input_layernorm(incoming),position_embeddings=rope,attention_mask=mask)[0]
        finally:layer.self_attn.o_proj=o
        f=factor(attention_input);value,st=group.export(original['o'],attention_input,f,root/f'layer{i}','o')
        o.weight.copy_(value);stats[f'layer{i}/o']=st;record_weight(f'layer{i}/o');del f
        residual=incoming+o(attention_input);del attention_input
        normed=layer.post_attention_layernorm(residual);f=factor(normed)
        for name in ['gate','up']:
            value,st=group.export(original[name],normed,f,root/f'layer{i}',name)
            layer.get_submodule(rot.PROJECTIONS[name]).weight.copy_(value)
            stats[f'layer{i}/{name}']=st;record_weight(f'layer{i}/{name}')
        del f
        down_input=layer.mlp.act_fn(layer.mlp.gate_proj(normed))*layer.mlp.up_proj(normed);del normed
        f=factor(down_input);value,st=group.export(original['down'],down_input,f,root/f'layer{i}','down')
        layer.mlp.down_proj.weight.copy_(value);stats[f'layer{i}/down']=st;record_weight(f'layer{i}/down')
        hidden=residual+layer.mlp.down_proj(down_input);del f,down_input,residual,original,value
        for sample in [0,1,2,3]:
            reference=layer(incoming[sample:sample+1],attention_mask=mask,position_embeddings=rope)[0]
            check=dict(layer=i,sample=sample,**metrics(hidden[sample:sample+1],reference))
            assert check['finite'] and check['nrmse']<=.003 and check['cosine']>=.99999,check
            checks.append(check)
        del incoming
        for short,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            assert (BASE_PACKAGE/f'layer{i}/{short}_norm_weight_f16.bin').read_bytes()==layer.get_submodule(long).weight.numpy().tobytes()
        checkpoint=OUTPUT/'checkpoints'/v;checkpoint.mkdir(parents=True,exist_ok=True)
        np.save(checkpoint/f'layer{i}_hidden.npy',hidden.numpy())
        write(f'{v}/layer{i}_stats.json',{k:val for k,val in stats.items() if k.startswith(f'layer{i}/')})
        print('QUANTIZED_LAYER',v,i,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    model.model.norm.half();load_packed(model.lm_head.weight,BASE_PACKAGE,'generation_lm_head')
    del hidden;gc.collect();model.half()
    for n,value in [('generation_embedding_weight_f16.bin',model.model.embed_tokens.weight),
                    ('generation_final_norm_weight_f16.bin',model.model.norm.weight)]:
        assert (BASE_PACKAGE/n).read_bytes()==value.numpy().tobytes()
    prior=json.loads((BASE_PACKAGE/'manifest.json').read_text())['files']
    prior={n.replace(chr(92),'/'):prior.get(n.replace(chr(92),'/'),x) for n,x in prior.items()}
    inherited={n:d['sha256'] for n,d in prior.items() if not any(n.startswith(f'layer{i}/{name}_weight_w4_') for i in range(28) for name in rot.PROJECTIONS)}
    assert all(sha(BASE_PACKAGE/n)==h for n,h in inherited.items())
    manifest=dict(experiment='EXP-0231',variant=v,format=group.FORMAT,groupsize=128,
        group_axis='contiguous_original_input_columns_per_output_row',grid=[-7,7],scale_dtype='float32',
        files={n:pack.file_record(root/n) for n in changed},original_shards=origin,
        frozen_base_package=str(BASE_PACKAGE),frozen_base_manifest_sha256=BASE_HASH,inherited_files=inherited,
        calibration_sha256=sha(verified('exp0230','inputs/C8_calibration_u32.bin')),calibration_tokens=8192,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip())
    pack.atomic_write_bytes(root/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    write(f'{v}/weight_stats.json',stats);write(f'{v}/calibration_forward_checks.json',checks)
    write(f'{v}/package.json',dict(manifest_sha256=sha(root/'manifest.json'),original_shards=origin,
        elapsed_s=time.monotonic()-started,calibration_tokens=8192,format=group.FORMAT,
        quantizer_source={n:sha(SOURCE/'scripts'/n) for n in ['gptq_exp0221.py','clipping_exp0223.py','group_exp0231.py','export_exp0231.py']},
        control_manifest_sha256=BASE_HASH,source_head=manifest['source_head']))
    print('PACKAGE_COMPLETE',v,round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    p=argparse.ArgumentParser();p.add_argument('variant',nargs='?',default='G128',choices=['G128']);a=p.parse_args();prepare(a.variant)
