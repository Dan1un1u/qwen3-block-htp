#!/usr/bin/env python3
"""Export a continuous unrotated FP16 Llama slice from original tensors."""
import argparse,json,struct
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from llama_quantized import load_quantized,link_projection
from llama_reference import PROJECTIONS,provenance,pack_weight,rope,layer,sha256
from export_llama32 import write,QP_NAMES

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers",type=int,choices=[3,16],required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--fixture",type=Path,required=True)
    ap.add_argument("--model",type=Path,default=Path("/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin"))
    ap.add_argument("--quantized",type=Path)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    prov=provenance(args.model);args.output.mkdir(parents=True)
    cfg=json.loads((args.model/"config.json").read_text())
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,attn_implementation="eager",local_files_only=True).cuda().eval()
    if args.quantized:load_quantized(model,args.quantized)
    w=model.state_dict();ids=torch.tensor([json.loads(args.fixture.read_text())["ids"]],device="cuda")
    x=torch.nn.functional.embedding(ids[:,:64],w["model.embed_tokens.weight"])
    dx=torch.nn.functional.embedding(ids[:,64:],w["model.embed_tokens.weight"])
    def padded(t):
        a=torch.zeros((64,2048),dtype=torch.float16,device="cuda");a[:t.shape[1]]=t[0];return a
    write(args.output/"block_input_f16.bin",padded(x))
    write(args.output/"replay_decode_input_00_f16.bin",padded(dx))
    for start,suffix in [(0,None),(64,"00")]:
        c,s=rope(cfg,torch.arange(start,start+64,device="cuda")[None],torch.float16)
        for name,v in [("cos",c),("sin",s)]:
            file=f"rope_{name}_f16.bin" if suffix is None else f"replay_decode_rope_{name}_{suffix}_f16.bin"
            write(args.output/file,v)
    for i in range(args.layers):
        out=args.output/f"layer{i}";out.mkdir()
        x,cache=layer(x,w,i,cfg,torch.arange(64,device="cuda")[None])
        dx,full=layer(dx,w,i,cfg,torch.tensor([[64]],device="cuda"),cache)
        for name,key in PROJECTIONS.items():
            if args.quantized:link_projection(args.quantized,i,name,out)
            else:write(out/f"{name}_weight_f16_hmx.bin",pack_weight(w[f"model.layers.{i}.{key}.weight"]))
        for name,key in [("input","input_layernorm"),("post","post_attention_layernorm")]:write(out/f"{name}_norm_weight_f16.bin",w[f"model.layers.{i}.{key}.weight"])
        for name in ["q","k"]:write(out/f"{name}_norm_weight_f16.bin",np.zeros(64,dtype="<f2"))
        (out/"qparams_u8.bin").write_bytes(b"".join(struct.pack("<32sfi2f",n.encode(),1.,0,0.,255.) for n in QP_NAMES))
        for j,name in enumerate(["k","v"]):
            a=torch.zeros((8,80,64),dtype=torch.float16,device="cuda")
            write(out/f"kv_cache_{name}_f16.bin",a)
            a[:,:65]=full[j][0];write(out/f"reference_kv_cache_{name}_f16.bin",a)
        print(f"EXPORTED_LAYER={i}",flush=True)
    write(args.output/("reference_w4f16_block_output_f16.bin" if args.quantized else "reference_f16f16_block_output_f16.bin"),padded(x))
    write(args.output/"replay_decode_reference_00_f16.bin",padded(dx))
    manifest={"experiment":("L32-0002" if args.quantized else "L32-0001"),"recipe":("W4A16" if args.quantized else "W16A16"),"layers":args.layers,"cache_capacity":80,"decode_steps":1,"original":prov,"fixture_sha256":sha256(args.fixture),"reference":"independent sequential FP16 Llama math, before final model RMSNorm","files":{str(f.relative_to(args.output)):{"bytes":f.stat().st_size,"sha256":sha256(f)} for f in sorted(args.output.rglob("*")) if f.is_file()}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":main()
