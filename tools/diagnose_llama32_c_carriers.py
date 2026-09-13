#!/usr/bin/env python3
"""Standalone boundary quantization diagnostics on the C BF16 control trajectory.

This is not a cumulative hardware ablation and does not change the candidate.
"""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from evaluate_llama32_c import input_only_model,FRONT,MODELS,RESULTS,ROOT
from export_llama32_u8 import quantize
from llama_reference import sha256

@torch.inference_mode()
def main():
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.32)
    out=RESULTS/'float-carrier-diagnostic-a01';out.mkdir(exist_ok=False)
    model=input_only_model();values={};handles=[]
    def capture(name):
        def hook(module,args,result):
            value=result[0] if isinstance(result,tuple) else result
            values[name]=value.detach().float().cpu().numpy()[0]
        return hook
    def capture_input(name):
        def hook(module,args):values[name]=args[0].detach().float().cpu().numpy()[0]
        return hook
    for i,block in enumerate(model.model.layers):
        handles.append(block.register_forward_pre_hook(capture_input((i,'block_input'))))
        handles.append(block.self_attn.o_proj.register_forward_hook(capture((i,'attention_projection'))))
        handles.append(block.post_attention_layernorm.register_forward_pre_hook(capture_input((i,'post_attention_residual'))))
        handles.append(block.mlp.down_proj.register_forward_hook(capture((i,'down'))))
        handles.append(block.register_forward_hook(capture((i,'block_output'))))
    ids=np.fromfile(FRONT/'generation_prompt_token_ids_u32.bin',dtype='<u4').astype('i8')
    model(torch.tensor(ids[None],device='cuda'),use_cache=False)
    for h in handles:h.remove()
    cal=json.loads((MODELS/'calibration-a01/calibration.json').read_text());rows=[]
    for (i,name),x in sorted(values.items()):
        np.save(out/f'layer{i}-{name}.npy',x)
        q=cal['qparams'][i][name];codes=quantize(x,q);restored=(codes.astype('f4')-q['zero_point'])*q['scale']
        slices={}
        for label,part in [('position0',slice(0,1)),('positions1to63',slice(1,None)),('all',slice(None))]:
            a=x[part].astype('f8');b=restored[part].astype('f8');v=codes[part]
            slices[label]=dict(rms=float(np.sqrt(np.mean(a*a))),nrmse=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),quantized_zero_fraction=float((v==q['zero_point']).mean()),saturation_fraction=float(((v==0)|(v==255)).mean()))
        rows.append(dict(layer=i,boundary=name,scale=q['scale'],slices=slices))
    result=dict(scope='standalone native affine boundary quantization on BF16 C-control chat-prefill trajectory; no cumulative hardware ablation',calibration_sha256=sha256(MODELS/'calibration-a01/calibration.json'),package_manifest_sha256=sha256(FRONT/'manifest.json'),rows=rows,arrays={p.name:sha256(p) for p in out.glob('*.npy')})
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    for row in rows:
        if row['layer']<3 and row['boundary'] in ['down','block_output']:print(row,flush=True)
if __name__=='__main__':main()
