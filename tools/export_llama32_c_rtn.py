#!/usr/bin/env python3
"""Fresh FP64 R1/R2 fold and learned-scale native per-channel RTN export."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from llama_reference import PROJECTIONS,sha256,provenance
from quantize_llama32 import pack
ROOT=Path(__file__).resolve().parents[1]
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013')
ORIGINAL=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin')

def verify_stage(stage):
    root=MODELS/stage;m=json.loads((root/'complete.json').read_text())
    assert m['updates']==100
    for n,h in m['artifacts'].items():assert sha256(root/n)==h
    return root/'rotation'

def write_projection(out,name,w,scale):
    scale=scale.reshape(-1).float()
    assert len(scale)==len(w) and torch.isfinite(scale).all() and (scale>0).all()
    q=(w.float()/scale[:,None]).round().clamp(-7,7).to(torch.int8)
    pack(q).tofile(out/(name+'_weight_w4_hmx.bin'))
    scale.cpu().numpy().astype('<f4').tofile(out/(name+'_weight_w4_scale_f32.bin'))
    # Native converter's unexpanded dot uses these original FP32 scales.
    dequant=(q.float()*scale[:,None]).half()
    return dequant

@torch.inference_mode()
def main():
    import subprocess
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    rot=verify_stage('c');prov=provenance(ORIGINAL)
    state=torch.load(rot/'R.bin',map_location='cpu',weights_only=True)
    scales=torch.load(rot/'quant_scales.pt',map_location='cpu',weights_only=True)
    assert len(scales['weight'])==112 and len(scales['activation'])==96
    out=MODELS/'quant-a01';out.mkdir(exist_ok=False)
    r=state['R1'].cuda().double();assert r.shape==(2048,2048)
    ortho=float((r.T@r-torch.eye(2048,device='cuda')).abs().max());assert ortho<1e-3
    # Reproduce pinned gamma-fold BF16 rounding and reference embedding centering.
    # Centering is recorded separately: RMSNorm does not make it algebraically exact.
    with safe_open(ORIGINAL/'model.safetensors',framework='pt',device='cpu') as f:
        norm=f.get_tensor('model.norm.weight').cuda()
        embed=f.get_tensor('model.embed_tokens.weight')
        ep=out/'embedding_weight_f16.bin';hp=out/'head';hp.mkdir()
        with ep.open('xb') as ef,(hp/'generation_lm_head_weight_w4_hmx.bin').open('xb') as cf,(hp/'generation_lm_head_weight_w4_scale_f32.bin').open('xb') as sf,(hp/'dequant_f16.bin').open('xb') as hf:
            for start in range(0,128256,1024):
                w=embed[start:start+1024].cuda()
                centered=(w.double()-w.double().mean(-1,keepdim=True)).bfloat16()
                (centered.double()@r).bfloat16().half().cpu().numpy().astype('<f2').tofile(ef)
                folded=(w.double()*norm.double()).bfloat16()
                head=(folded.double()@r).bfloat16().float()
                scale=(head.abs().amax(1)/7).clamp_min(1e-8)
                q=(head/scale[:,None]).round().clamp(-7,7).to(torch.int8)
                cf.write(pack(q).tobytes());sf.write(scale.cpu().numpy().astype('<f4').tobytes());hf.write((q.float()*scale[:,None]).half().cpu().numpy().astype('<f2').tobytes())
        np.ones(2048,dtype='<f2').tofile(out/'generation_final_norm_weight_f16.bin')
        del embed
        for index in range(16):
            layer=out/f'layer{index}';layer.mkdir();values={};r2=state[f'model.layers.{index}.self_attn.R2'].cuda().double();assert r2.shape==(64,64)
            assert float((r2.T@r2-torch.eye(64,device='cuda')).abs().max())<1e-3
            pre=f.get_tensor(f'model.layers.{index}.input_layernorm.weight').cuda().double()
            post=f.get_tensor(f'model.layers.{index}.post_attention_layernorm.weight').cuda().double()
            for short,name in PROJECTIONS.items():
                w=f.get_tensor(f'model.layers.{index}.{name}.weight').cuda()
                if short in ['q','k','v']:w=(w.double()*pre).bfloat16()
                if short in ['gate','up']:w=(w.double()*post).bfloat16()
                w=(r.T@w.double() if short in ['o','down'] else w.double()@r).bfloat16()
                if short=='v':
                    shape=w.T.shape;w=(w.T.reshape(-1,shape[-1]//64,64).double()@r2).reshape(shape).T.bfloat16()
                if short=='o':
                    shape=w.shape;w=(w.reshape(-1,shape[-1]//64,64).double()@r2).reshape(shape).bfloat16()
                key=f'model.layers.{index}.{name}.module.quantizer'
                scale=scales['weight'][key].cuda()
                values[name+'.weight']=write_projection(layer,short,w,scale).cpu().contiguous()
            for name in ['input_layernorm.weight','post_attention_layernorm.weight']:values[name]=torch.ones(2048,dtype=torch.float16)
            save_file(values,layer/'dequant.safetensors')
            np.ones(2048,dtype='<f2').tofile(layer/'input_norm_weight_f16.bin');np.ones(2048,dtype='<f2').tofile(layer/'post_norm_weight_f16.bin')
            (layer/'complete.json').write_text(json.dumps(dict(layer=index,files={p.name:sha256(p) for p in layer.iterdir() if p.is_file()}),indent=2)+'\n')
            print('C_RTN_EXPORTED_LAYER',index,flush=True)
    sa={n:float(v.item()) for n,v in scales['activation'].items()}
    for index in range(16):
        for group in [('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'),('mlp.gate_proj','mlp.up_proj')]:
            assert len({sa[f'model.layers.{index}.{n}.quantizer'] for n in group})==1
    (out/'activation_scales.json').write_text(json.dumps(sa,indent=2)+'\n')
    manifest=dict(experiment='L32-0013',original=prov,rotation='learned R1/R2 offline only',rotation_sha256=sha256(rot/'R.bin'),scales_sha256=sha256(rot/'quant_scales.pt'),orthogonal_r1_max_abs=ortho,weight_method='fresh learned SW C-RTN backbone112; native signed[-7,7] per-output; head original hardware W4 contract, fresh absmax RTN',folding='FP64 multiply, BF16 cast after gamma/R1/R2 stages following pinned reference; reference embedding mean centering retained, not asserted algebraically equivalent to RMSNorm teacher',files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in out.rglob('*') if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('C_RTN_EXPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
