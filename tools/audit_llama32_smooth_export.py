#!/usr/bin/env python3
"""Verify every fresh W4 grid/code against float control and native packing."""
import json
import numpy as np
import torch
from llama32_smooth_accuracy import MOD,OUT,ROOT,save
from llama_reference import PROJECTIONS,sha256
from llama_u8_reference import unpack_w4_codes
if __name__=='__main__':
    torch.set_num_threads(4);data=torch.load(MOD/'quantized.pt',weights_only=True);state=torch.load(MOD/'gptq.pt',weights_only=True);rows=[]
    for i in range(16):
        for short,name in PROJECTIONS.items():
            key=f'{i}.{short}';c=data['codes'][key];s=data['scales'][key]
            assert c.dtype==torch.int8 and int(c.min())>=-7 and int(c.max())<=7
            assert s.shape==(c.shape[0],1) and (s>0).all() and torch.isfinite(s).all()
            expected=(c.float()*s).bfloat16();assert torch.equal(expected,state[f'model.layers.{i}.{name}.weight'])
            packed=unpack_w4_codes(MOD/'native-a01'/f'layer{i}',short,*c.shape)
            assert np.array_equal(packed,c.numpy())
            exported=np.fromfile(MOD/'native-a01'/f'layer{i}'/(short+'_weight_w4_scale_f32.bin'),dtype='<f4')
            assert np.array_equal(exported,s.numpy().reshape(-1))
            rows.append(dict(layer=i,projection=short,shape=list(c.shape),codes_min=int(c.min()),codes_max=int(c.max()),exact=True))
    save(OUT/'export-audit.json',dict(pass_gate=True,matrices=len(rows),code_values=sum(int(np.prod(r['shape'])) for r in rows),rows=rows,signed_range=[-7,7],granularity='per-output-channel only',sp2=False,source_quantized_sha256=sha256(MOD/'quantized.pt'),sdk_reference_sha256=sha256(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')))
    print('EXPORT_AUDIT_PASS',len(rows),flush=True)
