#!/usr/bin/env python3
"""Package fresh C-RTN/SP2 binaries and independent single/3/16-layer fixtures."""
import argparse,json,os,struct
from pathlib import Path
import numpy as np
import torch
from llama_reference import sha256,rope,PROJECTIONS
from export_llama32_u8 import quantize,write_qparams,configs,layer
from llama_u8_reference import load_qparams_bin,HmxU8Converter,exact_rms_norm_u8,projection_bias_words,_cached_w4_projection
from llama32_sp2_contract import encode,quantize_integer,contract
from llama32_c_integer_oracle import NativeOracle
ROOT=Path(__file__).resolve().parents[1]
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013');RESULTS=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013')
QUANT=MODELS/'quant-a01';FRONT=MODELS/'frontend-a01';CAL=MODELS/'calibration-a01/calibration.json'

def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(root):
    m=json.loads((root/'manifest.json').read_text())
    for n,h in m['files'].items():assert sha256(root/n)==h['sha256']
    return m
def seal(root,**extra):
    save(root/'manifest.json',dict(experiment='L32-0013',recipe='W4A8',rotation='R1/R2 offline',sp2=True,sp2_encoding=contract(),cache_capacity=80,**extra,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in root.rglob('*') if p.is_file()}))
def prepare():
    m=verify(QUANT);cal=json.loads(CAL.read_text());assert sha256(QUANT/'manifest.json')==cal['weight_manifest_sha256']
    FRONT.mkdir(exist_ok=False)
    bounds=[]
    for index in range(16):
        out=FRONT/f'layer{index}';out.mkdir();q=cal['qparams'][index]
        for f in (QUANT/f'layer{index}').iterdir():
            if f.suffix=='.bin':os.link(f,out/f.name)
        for n in ['q_norm_weight_f16.bin','k_norm_weight_f16.bin']:np.ones(64,dtype='<f2').tofile(out/n)
        write_qparams(out/'qparams_u8.bin',q)
        (out/'attention_config_all_groups.bin').write_bytes(b''.join(struct.pack('<IIIIiiiiiIIIIII',*v) for v in configs(q)))
        w,ws=_cached_w4_projection(str(out.resolve()),'down',2048,8192)
        positive=np.maximum(w,0).astype('i4').sum(1);negative=np.minimum(w,0).astype('i4').sum(1)
        assert (positive*255<=8388607).all() and (negative*255>=-8388608).all(),('signed24 partial bound',index)
        mult=np.floor(float(q['middle']['scale'])*ws.astype('f8')/q['down']['scale']*2**31+.5).astype('i8')
        assert ((mult>0)&(mult<2**31)).all(),('Q31 multiplier bound',index)
        bounds.append(dict(layer=index,partial_min=int(negative.min()*255),partial_max=int(positive.max()*255),multiplier_min=int(mult.min()),multiplier_max=int(mult.max())))
        g=(np.arange(256,dtype='f4')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype='f4')-q['up']['zero_point'])*q['up']['scale']
        x=(g/(1+np.exp(-np.clip(g,-80,80))))[:,None]*u[None,:]
        v=quantize_integer(x,cal['sp2'][index]['alpha']);encode(v).tofile(out/'silu_up_lut_u16.bin');save(out/'sp2_encoding.json',contract())
        for n in ['k','v']:np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1').tofile(out/f'kv_cache_{n}_u8.bin')
    for n in ['generation_lm_head_weight_w4_hmx.bin','generation_lm_head_weight_w4_scale_f32.bin']:os.link(QUANT/'head'/n,FRONT/n)
    os.link(QUANT/'generation_final_norm_weight_f16.bin',FRONT/'generation_final_norm_weight_f16.bin')
    embedding=np.memmap(QUANT/'embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(128256,2048))
    with (FRONT/'generation_embedding_weight_u8.bin').open('xb') as f:
        for start in range(0,128256,1024):quantize(embedding[start:start+1024],cal['qparams'][0]['block_input']).tofile(f)
    old=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/frontend-a01/generation_prompt_token_ids_u32.bin')
    ids=np.fromfile(old,dtype='<u4');assert len(ids)==64;ids.tofile(FRONT/old.name)
    save(FRONT/'prompt_provenance.json',dict(scope='same frozen Llama chat prompt; no weight/calibration reuse',source=str(old),sha256=sha256(old)))
    cfg=json.loads(Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin/config.json').read_text())
    for step in range(16):
        pos=torch.arange(64) if step==0 else torch.full((64,),63+step)
        c,s=rope(cfg,pos[None],torch.float16)
        for name,array in [('cos',c),('sin',s)]:array.numpy().astype('<f2').tofile(FRONT/(f'rope_{name}_f16.bin' if step==0 else f'generation_decode_rope_{name}_{step-1:02d}_f16.bin'))
    q=cal['generation_qparams'];write_qparams(FRONT/'generation_qparams_u8.bin',q)
    w,scales=_cached_w4_projection(str(FRONT.resolve()),'generation_lm_head',128256,2048)
    lo,hi=projection_bias_words(w,scales,q['generation_final_norm_output'],q['generation_lm_head_output'])
    np.concatenate([lo.reshape(-1,32),hi.view('<u4').reshape(-1,32)],1).astype('<u4').tofile(FRONT/'generation_lm_head_bias_u32.bin')
    save(FRONT/'preparation.json',dict(weight_manifest_sha256=sha256(QUANT/'manifest.json'),calibration_sha256=sha256(CAL),original=m['original'],down_bounds=bounds))
    print('C_PACKAGE_PREPARED',flush=True)

def oracle():
    cal=json.loads(CAL.read_text());helper=NativeOracle();helper.install();conv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
    qparams=[load_qparams_bin(FRONT/f'layer{i}/qparams_u8.bin') for i in range(16)]
    embed=np.memmap(FRONT/'generation_embedding_weight_u8.bin',dtype='u1',mode='r',shape=(128256,2048));ids=np.fromfile(FRONT/'generation_prompt_token_ids_u32.bin',dtype='<u4').tolist()
    cache=[None]*16;tokens=[];selected=[];out=RESULTS/'oracle-a01';out.mkdir(exist_ok=False)
    gamma=np.fromfile(FRONT/'generation_final_norm_weight_f16.bin',dtype='<f2');gq=cal['generation_qparams']
    for step in range(16):
        x=np.array(embed[ids if step==0 else [tokens[-1]]]);name='rope_cos_f16.bin' if step==0 else f'generation_decode_rope_cos_{step-1:02d}_f16.bin'
        c=np.fromfile(FRONT/name,dtype='<f2').reshape(64,64);s=np.fromfile(FRONT/name.replace('cos','sin'),dtype='<f2').reshape(64,64)
        if step==0:x.tofile(FRONT/'reference_w4u8_block_input_u8.bin')
        for index in range(16):
            if step<2:np.save(out/f'step{step}-layer{index}-input.npy',x)
            x,cache[index],diag=layer(x,FRONT/f'layer{index}',qparams[index],c,s,cache[index],conv,sp2=True)
            if step<2:
                np.save(out/f'step{step}-layer{index}-output.npy',x)
                for j,n in enumerate(['k','v']):
                    v=np.full((8,80,64),qparams[index]['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');v[:,:64+step]=cache[index][j]
                    np.save(out/f'step{step}-layer{index}-cache-{n}.npy',v)
                    if step==0:v.tofile(FRONT/f'layer{index}/reference_kv_cache_{n}_u8.bin')
            print('C_INTEGER_ORACLE_LAYER',step,index,flush=True)
        if step==0:x.tofile(FRONT/'reference_w4u8_integer_attention_block_output_u8.bin')
        n=exact_rms_norm_u8(x[-1:],qparams[-1]['block_output'],gamma,gq['generation_final_norm_output'])
        logits=helper.project(n,FRONT,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],conv)[0]
        token=int(logits.argmax());tokens.append(token);selected.append(int(logits[token]));save(out/f'step-{step:02d}.json',dict(token=token,code=selected[-1]))
    np.array(tokens,dtype='<u4').tofile(FRONT/'generation_expected_token_ids_u32.bin')
    save(out/'teacher.json',dict(u8_generated_ids=tokens,u8_selected_codes=selected,prompt_ids=ids,scope='exact native arithmetic oracle, not BF16 quality teacher',integer_dot_audited_projections=len(helper.audited)))
    prep=json.loads((FRONT/'preparation.json').read_text());seal(FRONT,layers=16,generation_tokens=16,prompt_tokens=64,original=prep['original'],frontend_teacher_sha256=sha256(out/'teacher.json'),calibration_sha256=sha256(CAL))
    print('C_INTEGER_ORACLE_COMPLETE',flush=True)

def fixtures():
    m=verify(FRONT);q=[load_qparams_bin(FRONT/f'layer{i}/qparams_u8.bin') for i in range(16)]
    oracle=RESULTS/'oracle-a01';out=MODELS/'gates-a01';out.mkdir(exist_ok=False)
    for start,count in [(0,1),(7,1),(15,1),(0,3)]:
        dst=out/f'layer{start}-count{count}';dst.mkdir()
        for local,index in enumerate(range(start,start+count)):
            layer=dst/f'layer{local}';layer.mkdir()
            for f in (FRONT/f'layer{index}').iterdir():
                if f.name.startswith('reference_'):continue
                os.link(f,layer/f.name)
            for n in ['k','v']:np.load(oracle/f'step1-layer{index}-cache-{n}.npy').tofile(layer/f'reference_kv_cache_{n}_u8.bin')
        for step in [0,1]:
            x=np.load(oracle/f'step{step}-layer{start}-input.npy');y=np.load(oracle/f'step{step}-layer{start+count-1}-output.npy')
            def pad(a,z):
                v=np.full((64,2048),z,dtype='u1');v[:len(a)]=a;return v
            pad(x,q[start]['block_input']['zero_point']).tofile(dst/('reference_w4u8_block_input_u8.bin' if step==0 else 'replay_decode_input_00_u8.bin'))
            pad(y,q[start+count-1]['block_output']['zero_point']).tofile(dst/('reference_w4u8_integer_attention_block_output_u8.bin' if step==0 else 'replay_decode_reference_00_u8.bin'))
        for name in ['cos','sin']:
            os.link(FRONT/f'rope_{name}_f16.bin',dst/f'rope_{name}_f16.bin');os.link(FRONT/f'generation_decode_rope_{name}_00_f16.bin',dst/f'replay_decode_rope_{name}_00_f16.bin')
        seal(dst,layers=count,source_layer=start,decode_steps=1,parent_package_manifest_sha256=sha256(FRONT/'manifest.json'))
    print('C_GATES_PREPARED',flush=True)
if __name__=='__main__':
    import subprocess
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    torch.set_num_threads(8);p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','oracle','fixtures']);globals()[p.parse_args().stage]()
