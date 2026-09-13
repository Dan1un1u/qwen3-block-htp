#!/usr/bin/env python3
"""Bound FP32 training-fold perturbation relative to the pinned FP64 method."""
import json,sys,time
from pathlib import Path
import torch
from safetensors import safe_open
from llama32_rotation_math import rotated_weight_fp32
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"build/l32-0013/reference/repos/SpinQuant"))
from train_utils.quant_linear import QuantizeLinear
@torch.inference_mode()
def main():
    torch.set_num_threads(8);torch.manual_seed(13);torch.backends.cuda.matmul.allow_tf32=False
    # Dense learned-like orthogonal matrices exercise sums absent in signed Hadamard.
    r=torch.linalg.qr(torch.randn(2048,2048,device="cuda",dtype=torch.float32))[0]
    r2=torch.linalg.qr(torch.randn(64,64,device="cuda",dtype=torch.float32))[0]
    rows=[]
    model="/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin/model.safetensors"
    with safe_open(model,framework="pt",device="cpu") as f:
        for layer in [0,7,15]:
            for name,transpose,rr in [("self_attn.q_proj",False,None),("self_attn.v_proj",False,r2),("self_attn.o_proj",True,r2),("mlp.down_proj",True,None)]:
                w=f.get_tensor(f"model.layers.{layer}.{name}.weight").cuda()
                obj=type("WeightHolder",(),{"weight":w})()
                torch.cuda.synchronize();t=time.monotonic();reference=QuantizeLinear.rotated_weight(obj,r,rr,transpose);torch.cuda.synchronize();slow=time.monotonic()-t
                t=time.monotonic();candidate=rotated_weight_fp32(obj,r,rr,transpose);torch.cuda.synchronize();fast=time.monotonic()-t
                delta=(candidate.float()-reference.float());nrmse=float(delta.square().mean().sqrt()/reference.float().square().mean().sqrt())
                row=dict(layer=layer,projection=name,nrmse=nrmse,max_abs=float(delta.abs().max()),different_fraction=float((delta!=0).float().mean()),fp64_seconds=slow,fp32_seconds=fast,pass_gate=nrmse<=1e-4)
                rows.append(row);print(json.dumps(row),flush=True)
    report=dict(rows=rows,pass_gate=all(r["pass_gate"] for r in rows),gate="NRMSE <=1e-4 after BF16 cast; training implementation only, not checkpoint equivalence",tf32=False)
    p=Path("/mnt/d/llm_exp/results/llama32-htp/l32-0013/rotation-fp32-selective-fp64-audit.json");p.open("x").write(json.dumps(report,indent=2)+"\n")
    assert report["pass_gate"]
if __name__=="__main__":main()
