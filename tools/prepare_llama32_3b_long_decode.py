#!/usr/bin/env python3
"""L32-0044 independent64-step oracle, frozen0041 weights/scales; cache128."""
import os,json
from pathlib import Path
import numpy as np
import torch
from llama32_3b_reference import *
from llama_u8_reference import load_qparams_bin
R=RES.parent/'l32-0044';M=OUT.parent/'l32-0044'
def main():
    preflight();torch.set_num_threads(8)
    src=OUT/'frontend-a01';dst=M/'frontend64-a01';dst.mkdir(parents=True,exist_ok=False)
    mf=json.loads((src/'manifest.json').read_text())
    for n,h in mf['files'].items():
        assert sha256(src/n)==h['sha256'],n
        if 'kv_cache_' in n or n.startswith('generation_decode_rope_') or n=='generation_expected_token_ids_u32.bin':continue
        d=dst/n;d.parent.mkdir(exist_ok=True,parents=True);os.link(src/n,d)
    qs=[load_qparams_bin(dst/f'layer{i}/qparams_u8.bin') for i in range(28)]
    for i,q in enumerate(qs):
        for n,key in [('k','k_rope'),('v','v')]:
            np.full((KH,128,D),q[key]['zero_point'],'u1').tofile(dst/f'layer{i}/kv_cache_{n}_u8.bin')
    ids=np.fromfile(dst/'generation_prompt_token_ids_u32.bin','<u4')
    embed=np.memmap(dst/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(128256,H))
    gamma=np.fromfile(dst/'generation_final_norm_weight_f16.bin','<f2');gq=load_qparams_bin(dst/'generation_qparams_u8.bin')
    cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');caches=[None]*28;tokens=[];codes=[]
    old=json.loads((RES/'frontend-a01-teacher.json').read_text())
    for step in range(64):
        x=np.array(embed[ids if not step else [tokens[-1]]],dtype='f4')
        c,s=rope(CFG,torch.arange(0 if not step else 63+step,64 if not step else 127+step)[None],torch.float16);c=c.numpy()[0];s=s.numpy()[0]
        if step:
            for n,v in [('cos',c),('sin',s)]:v.astype('<f2').tofile(dst/f'generation_decode_rope_{n}_{step-1:02d}_f16.bin')
        for i in range(28):
            x,caches[i],_=layer(x,dst/f'layer{i}',qs[i],c,s,caches[i])
            if not step:
                for j,n in enumerate(['k','v']):
                    ref=np.full((KH,128,D),qs[i]['k_rope' if n=='k' else 'v']['zero_point'],'u1');ref[:,:64]=caches[i][j];ref.tofile(dst/f'layer{i}/reference_kv_cache_{n}_u8.bin')
        if not step:assert np.array_equal(x,np.fromfile(dst/'reference_w4u8_block_output_f32.bin','<f4').reshape(64,H))
        np.asarray(x,'<f4').tofile(dst/f'audit_hidden_{step:02d}_f32.bin')
        act=norm(x[-1:],gamma,gq['generation_final_norm_output'])
        logits=project_w4u8(act,dst,'generation_lm_head',128256,H,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0]
        tok=int(logits.argmax());tokens.append(tok);codes.append(int(logits[tok]))
        if step<16:assert tok==old['u8_generated_ids'][step] and codes[-1]==old['u8_selected_codes'][step],step
        save(R/f'oracle-step-{step:02d}.json',dict(step=step,token=tok,code=codes[-1],hidden_sha256=sha256(dst/f'audit_hidden_{step:02d}_f32.bin')))
        print('ORACLE_STEP',step,tok,codes[-1],flush=True)
    np.asarray(tokens,'<u4').tofile(dst/'generation_expected_token_ids_u32.bin')
    save(R/'frontend64-a01-teacher.json',dict(prompt_ids=ids.tolist(),u8_generated_ids=tokens,u8_selected_codes=codes,quality_accepted=False,first16_match_frozen0041=True))
    save(dst/'manifest.json',dict(experiment='L32-0044',layers=28,generation_tokens=64,prompt_tokens=64,cache_capacity=128,teacher_sha256=sha256(R/'frontend64-a01-teacher.json'),files={str(f.relative_to(dst)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in dst.rglob('*') if f.is_file()}))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
