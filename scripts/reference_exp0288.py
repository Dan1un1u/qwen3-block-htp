#!/usr/bin/env python3
"""Qwen0.6B shape-aware independent FP32-residual/SP2 oracle and immutable packages."""
import argparse,json,os,struct
from pathlib import Path
import numpy as np
import torch
from prepare_exp0288 import ROOT,OUT,RES,MODEL,tensor,save,qp,preflight,rope
from llama_reference import sha256,PROJECTIONS
from reference_math_exp0288 import norm,layer as qwen_layer
from llama_u8_reference import HmxU8Converter,unpack_w4_codes,load_qparams_bin,project_w4u8,exact_qk_norm_rope_u8,exact_attention_dynamic,projection_bias_words
from export_llama32_u8 import carrier,write_qparams,divide
from prototype_llama32_sp2 import oracle
from llama_sp2_reference import table
CFG=json.loads((MODEL/'config.json').read_text())
H=1024;D=128;NH=16;KH=8;F=3072;CAP=128
def configs(q):
    import math
    score=carrier(q['q_rope']['scale']*q['k_rope']['scale']*(D**-.5)/(math.log(2)/8))
    den=max(q['v']['zero_point'],255-q['v']['zero_point'])
    av=carrier(q['attention_probability']['scale']*q['v']['scale']*den/127/q['attention_concat']['scale'])
    return [(1,g,3,1,q['q_rope']['zero_point'],q['k_rope']['zero_point'],q['v']['zero_point'],0,q['attention_concat']['zero_point'],127,den,*score,*av) for g in range(KH)]
def layer(x,p,q,c,s,past=None):
    return qwen_layer(x,p,c,s,past)
def export_layer(i,dst):
    src=OUT/f'layer{i}';done=json.loads((src/'complete.json').read_text())
    for n,h in done['files'].items():
        if 'weight_' in n:assert sha256(src/n)==h;os.link(src/n,dst/n)

    ranges=json.loads((src/'ranges.json').read_text());q={n:qp(*v) for n,v in ranges.items()};q['attention_probability']=qp(0,1)
    if i:q['block_input']=qp(*json.loads((OUT/f'layer{i-1}/ranges.json').read_text())['block_output'])
    levels=np.array(sorted({0}|{s*2**p for s in [-1,1] for p in range(15)}|{s*(2**p+2**j) for s in [-1,1] for p in range(15) for j in range(p)}),dtype='i2');assert len(levels)==241
    alpha=float(np.float32(max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))/24576))
    g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
    z=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:];grid=levels.astype('f8')*alpha
    ix=np.searchsorted(grid,z).clip(1,len(grid)-1);ix-=abs(z-grid[ix-1])<=abs(z-grid[ix]);v=levels[ix].astype('i4')
    for first in range(0,256,8):assert np.array_equal(v[first:first+8],levels[np.abs(z[first:first+8,:,None]-grid).argmin(-1)])
    enc=(v+32768).astype('<u2');enc.tofile(dst/'silu_up_lut_u16.bin')
    assert np.array_equal((enc&255).astype('i4')+256*(enc>>8).astype('i4')-32768,v)
    q['middle']['scale']=alpha;q['middle']['zero_point']=0;write_qparams(dst/'qparams_u8.bin',q)
    (dst/'attention_config_all_groups.bin').write_bytes(b''.join(struct.pack('<IIIIiiiiiIIIIII',*v) for v in configs(q)))
    for j,n in enumerate(['k','v']):
        np.full((KH,CAP,D),q['k_rope' if n=='k' else 'v']['zero_point'],'u1').tofile(dst/f'kv_cache_{n}_u8.bin')
    for n,k in [('o',NH*D),('down',F)]:
        w=unpack_w4_codes(dst,n,H,k).astype('i4');pos=np.maximum(w,0).sum(1);neg=np.minimum(w,0).sum(1)
        assert int(pos.max())*255<=8388607 and int(neg.min())*255>=-8388608
        assert n!='down' or int((pos-neg).max())*24576<2**31
    return q

def native_cache(dst,cache,q):
    from prepare_exp0161_segmented_cache import pack_k_segment,pack_v_segment
    cfg=configs(q);ks=[];vs=[]
    for g in range(KH):
        ks.append(b''.join(pack_k_segment(cache[0][g,i:i+32],cfg[g]) for i in (0,32))+bytes(4352+4096))
        seg=[pack_v_segment(cache[1][g,i:i+32],cfg[g]) for i in (0,32)]
        vs.append(b''.join(seg[0][0][n*1024:(n+1)*1024]+seg[1][0][n*1024:(n+1)*1024]+bytes(1024) for n in range(4))+seg[0][1]+bytes(4096))
    for n,data in [('k',b''.join(ks)),('v',b''.join(vs))]:
        (dst/f'kv_cache_{n}_hmx_u8_segmented.bin').write_bytes(bytes(len(data)))
        (dst/f'reference_kv_cache_{n}_hmx_u8_segmented_step00.bin').write_bytes(data)

def padwrite(p,x,dtype='<f4'):
    v=np.zeros((64,H),dtype);v[:len(x)]=x;v.tofile(p)
def ropes(dst,start,prefix='rope'):
    c,s=rope(CFG,torch.arange(start,start+64)[None],torch.float16)
    c=c.numpy()[0];s=s.numpy()[0]
    for n,v in [('cos',c),('sin',s)]:v.astype('<f2').tofile(dst/f'{prefix}_{n}_f16.bin')
    return c,s
def package(first,count,attempt):
    preflight();torch.set_num_threads(8);dst=OUT/attempt;dst.mkdir(exist_ok=False)
    qs=[]
    for i in range(count):
        p=dst/f'layer{i}';p.mkdir();qs.append(export_layer(first+i,p))
    c,s=ropes(dst,0)
    dc,ds=rope(CFG,torch.arange(64,128)[None],torch.float16);dc=dc.numpy()[0];ds=ds.numpy()[0]
    for n,v in [('cos',dc),('sin',ds)]:v.astype('<f2').tofile(dst/f'replay_decode_rope_{n}_00_f16.bin')
    ids=np.fromfile('/mnt/d/llm_exp/results/qwen3-block-htp/exp0230/inputs/C64_calibration_u32.bin','<u4').reshape(512,128)[0]
    if first==0:inputs=tensor('model.embed_tokens.weight')[torch.from_numpy(ids.astype('i8'))].half().float().numpy()
    else:inputs=np.load(OUT/f'layer{first-1}/hidden.npy',mmap_mode='r')[0].astype('f4')
    x=inputs[:64];dx=inputs[64:65]
    padwrite(dst/'reference_w4u8_block_input_f32.bin',x);padwrite(dst/'replay_decode_input_00_f32.bin',dx)
    # U8 slots are unused reserved references under the explicit FP32 contract.
    for n in ['reference_w4u8_block_input_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin','replay_decode_input_00_u8.bin','replay_decode_reference_00_u8.bin']:np.zeros((64,H),'u1').tofile(dst/n)
    for i,q in enumerate(qs):
        p=dst/f'layer{i}';x,cache,diag=layer(x,p,q,c,s);native_cache(p,cache,q);dx,dcache,ddiag=layer(dx,p,q,dc,ds,cache)
        for phase,di in [('prefill',diag),('decode',ddiag)]:
            for n,v in di.items():np.save(dst/f'l{i}-{phase}-{n}.npy',v)
        for j,n in enumerate(['k','v']):
            ref=np.full((KH,CAP,D),q['k_rope' if n=='k' else 'v']['zero_point'],'u1');ref[:,:65]=dcache[j];ref.tofile(p/f'reference_kv_cache_{n}_u8.bin')
        np.save(dst/f'l{i}-prefill-hidden.npy',x);np.save(dst/f'l{i}-decode-hidden.npy',dx);print('REFERENCE_LAYER',first+i,flush=True)
    padwrite(dst/'reference_w4u8_block_output_f32.bin',x);padwrite(dst/'replay_decode_reference_00_f32.bin',dx)
    save(dst/'manifest.json',dict(experiment='EXP-0288',model='Qwen3-0.6B',source_layer=first,layers=count,recipe='W4A8-SP2',fp32_residual=True,rotation='OFF',files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in dst.rglob('*') if p.is_file()}))
    print('PACKAGE',dst,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--first',type=int,default=0);a.add_argument('--layers',type=int,default=1);a.add_argument('--attempt',required=True);v=a.parse_args();package(v.first,v.layers,v.attempt)
