#!/usr/bin/env python3
"""Transport-only radix257 gate on sealed historical stimuli, not C model export."""
import json,os
from pathlib import Path
import numpy as np
from llama_reference import sha256
from llama32_sp2_contract import contract,encode,quantize_integer
from llama_u8_reference import load_qparams_bin,unpack_w4_codes,exact_residual_add_u8
from export_llama32_u8 import write_qparams
from prototype_llama32_sp2 import oracle,preflight
BASE=Path("/mnt/d/llm_exp/models/llama32-htp/l32-0003/layers-a01")
OUT=Path("/mnt/d/llm_exp/models/llama32-htp/l32-0013/transport-gates-a01")
def main():
    preflight();OUT.mkdir(parents=True,exist_ok=False)
    for index in [0,7,15]:
        roots=[BASE/f"layer{index}-{phase}" for phase in ["prefill","decode"]]
        for root in roots:
            for n,h in json.loads((root/"manifest.json").read_text())["files"].items():assert sha256(root/n)==h["sha256"]
        q=load_qparams_bin(roots[0]/"qparams_u8.bin")
        alpha=np.float32(max(abs(q['middle']['minimum']),abs(q['middle']['maximum'])))
        g=(np.arange(256,dtype=np.float32)-q['gate']['zero_point'])*q['gate']['scale']
        u=(np.arange(256,dtype=np.float32)-q['up']['zero_point'])*q['up']['scale']
        x=(g/(1+np.exp(-np.clip(g,-80,80))))[:,None]*u[None,:]
        v=quantize_integer(x,alpha);table=encode(v)
        q['middle']['scale']=float(np.float32(alpha/32768));q['middle']['zero_point']=0
        w=unpack_w4_codes(roots[0],'down',2048,8192)
        ws=np.fromfile(roots[0]/'down_weight_w4_scale_f32.bin',dtype='<f4')
        mult=np.floor(q['middle']['scale']*ws.astype('f8')/q['down']['scale']*2**31+.5).astype('i8')
        assert ((mult>0)&(mult<2**31)).all()
        outputs=[]
        for root in roots:
            a=v[np.load(root/'reference_gate.npy'),np.load(root/'reference_up.npy')]
            acc=oracle(a,w)
            down=np.clip(((acc*mult+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
            y=exact_residual_add_u8(np.load(root/'reference_residual.npy'),q['post_attention_residual'],down,q['down'],q['block_output'])
            padded=np.full((64,2048),q['block_output']['zero_point'],dtype='u1');padded[:len(y)]=y;outputs.append(padded)
        out=OUT/f'layer{index}';layer=out/'layer0';layer.mkdir(parents=True)
        p,d=roots
        skip=['silu_up_lut_u16.bin','qparams_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin','manifest.json']
        for f in p.iterdir():
            if f.name not in skip:os.link(d/f.name if f.name.startswith('reference_kv_cache') else f,layer/f.name)
        table.tofile(layer/'silu_up_lut_u16.bin');write_qparams(layer/'qparams_u8.bin',q)
        (layer/'sp2_encoding.json').write_text(json.dumps(contract(),indent=2)+'\n')
        for n in ['block_input_u8.bin','rope_cos_f16.bin','rope_sin_f16.bin']:os.link(p/n,out/('reference_w4u8_block_input_u8.bin' if n=='block_input_u8.bin' else n))
        for dst,src in [('replay_decode_input_00_u8.bin','block_input_u8.bin'),('replay_decode_rope_cos_00_f16.bin','rope_cos_f16.bin'),('replay_decode_rope_sin_00_f16.bin','rope_sin_f16.bin')]:os.link(d/src,out/dst)
        outputs[0].tofile(layer/'reference_w4u8_integer_attention_block_output_u8.bin');outputs[0].tofile(out/'reference_w4u8_integer_attention_block_output_u8.bin');outputs[1].tofile(out/'replay_decode_reference_00_u8.bin')
        m=dict(experiment='L32-0013',recipe='W4A8',layers=1,source_layer=index,cache_capacity=80,decode_steps=1,sp2=True,sp2_encoding=contract(),scope='historical sealed stimuli for native transport correctness only; not new C weights or quality',source_manifests={str(r):sha256(r/'manifest.json') for r in roots},files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.rglob('*') if f.is_file()})
        (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('TRANSPORT_FIXTURE',index,flush=True)
if __name__=='__main__':main()
