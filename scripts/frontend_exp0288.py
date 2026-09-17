#!/usr/bin/env python3
"""Fresh 3B embedding/head package and complete independent greedy trajectory."""
import os,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from reference_exp0288 import *
from llama_u8_reference import _cached_w4_projection

def main(attempt):
    preflight();torch.set_num_threads(8);dst=OUT/attempt;dst.mkdir(exist_ok=False)
    qs=[]
    for i in range(28):
        p=dst/f'layer{i}';p.mkdir();qs.append(export_layer(i,p))
    tensor('model.embed_tokens.weight').half().numpy().astype('<f2').tofile(dst/'generation_embedding_weight_f16.bin')
    tensor('model.norm.weight').half().numpy().astype('<f2').tofile(dst/'generation_final_norm_weight_f16.bin')
    head=OUT/'head';hm=json.loads((head/'complete.json').read_text())
    for n,h in hm['files'].items():
        if n.startswith('generation_'):
            assert sha256(head/n)==h;os.link(head/n,dst/n)
    cal=json.loads((OUT/'calibration.json').read_text());gq=cal['generation_qparams'];write_qparams(dst/'generation_qparams_u8.bin',gq)
    w,sc=_cached_w4_projection(str(dst.resolve()),'generation_lm_head',151936,H)
    lo,hi=projection_bias_words(w,sc,gq['generation_final_norm_output'],gq['generation_lm_head_output'])
    np.concatenate((lo.reshape(-1,32),hi.view('<u4').reshape(-1,32)),1).astype('<u4').tofile(dst/'generation_lm_head_bias_u32.bin')
    # Tokenizer-identical fixed M64 benchmark prompt; never reuse 1B numerical evidence.
    old=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0271/sp2-fp32/generation_prompt_token_ids_u32.bin')
    ids=np.fromfile(old,'<u4');assert ids.shape==(64,);ids.tofile(dst/old.name)
    embed=np.memmap(dst/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(151936,H))
    gamma=np.fromfile(dst/'generation_final_norm_weight_f16.bin','<f2');cv=HmxU8Converter(ROOT/'build/reference/qbh_hmx_u8_reference.so')
    caches=[None]*28;tokens=[];codes=[]
    for step in range(64):
        x=np.array(embed[ids if not step else [tokens[-1]]],dtype='f4')
        if not step:padwrite(dst/'reference_w4u8_block_input_f32.bin',x)
        c,s=ropes(dst,0 if not step else 63+step,'rope' if not step else f'generation_decode_rope') if not step else (None,None)
        if step:
            c,s=rope(CFG,torch.arange(63+step,127+step)[None],torch.float16);c=c.numpy()[0];s=s.numpy()[0]
            for n,v in [('cos',c),('sin',s)]:v.astype('<f2').tofile(dst/f'generation_decode_rope_{n}_{step-1:02d}_f16.bin')
        for i in range(28):
            x,caches[i],_=layer(x,dst/f'layer{i}',qs[i],c,s,caches[i])
            if not step:
                native_cache(dst/f'layer{i}',caches[i],qs[i])
                for j,n in enumerate(['k','v']):
                    ref=np.full((KH,CAP,D),qs[i]['k_rope' if n=='k' else 'v']['zero_point'],'u1');ref[:,:64]=caches[i][j];ref.tofile(dst/f'layer{i}/reference_kv_cache_{n}_u8.bin')
            pass
        if not step:padwrite(dst/'reference_w4u8_block_output_f32.bin',x)
        np.asarray(x,'<f4').tofile(dst/f'audit_hidden_{step:02d}_f32.bin')
        act=norm(x[-1:],gamma,gq['generation_final_norm_output'])
        logits=project_w4u8(act,dst,'generation_lm_head',151936,H,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0]
        tok=int(logits.argmax());tokens.append(tok);codes.append(int(logits[tok]));print('FRONTEND_STEP',step,tok,codes[-1],flush=True)
    np.asarray(tokens[:16],'<u4').tofile(dst/'generation_expected_token_ids_u32.bin')
    for n in ['reference_w4u8_block_input_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin']:np.zeros((64,H),'u1').tofile(dst/n)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    save(RES/(attempt+'-teacher.json'),dict(prompt_ids=ids.tolist(),u8_generated_ids=tokens,u8_selected_codes=codes,text=tokenizer.decode(tokens,skip_special_tokens=True),quality_accepted=False,recipe='fresh Qwen0.6B GPTQ per-channel W4, minmax static A8, SP2 mode8, FP32 residual, no rotations',arithmetic='independent complete HMX conversion/integer attention/FP32 residual oracle',calibration_sha256=sha256(OUT/'calibration.json')))
    save(dst/'manifest.json',dict(experiment='EXP-0288',model='Qwen3-0.6B',layers=28,generation_tokens=64,prompt_tokens=64,cache_capacity=128,recipe='W4A8-SP2',rotation='OFF',fp32_residual=True,teacher_sha256=sha256(RES/(attempt+'-teacher.json')),files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in dst.rglob('*') if p.is_file()}))
    print('FRONTEND_COMPLETE',flush=True)
if __name__=='__main__':
    import sys
    main(sys.argv[1])