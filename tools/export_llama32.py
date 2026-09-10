#!/usr/bin/env python3
"""Create immutable original-Llama FP16 single-layer prefill/decode fixtures."""
import argparse
import json
import os
import struct
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from llama_reference import PROJECTIONS, provenance, pack_weight, rope, layer, sha256

QP_NAMES = ["block_input", "input_norm", "q_projection", "k_projection", "v", "q_rope", "k_rope", "attention_probability", "attention_concat", "attention_projection", "post_attention_residual", "post_attention_norm", "gate", "up", "middle", "down", "block_output"]

def write(path, tensor):
    if path.exists():
        raise FileExistsError(path)
    data = tensor.detach().cpu().half().numpy() if isinstance(tensor, torch.Tensor) else tensor
    np.asarray(data).tofile(path)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prov = provenance(args.model)
    args.output.mkdir(parents=True)
    config = json.loads((args.model / "config.json").read_text())
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="eager", local_files_only=True).cuda().eval()
    weights = teacher.state_dict()
    ids = torch.tensor([json.loads(args.fixture.read_text())["ids"]], device="cuda")
    reference = teacher(ids, output_hidden_states=True, use_cache=False)
    packages = []
    for index in [0, 7, 15]:
        x = reference.hidden_states[index]
        _, past = layer(x[:, :64], weights, index, config, torch.arange(64, device="cuda")[None])
        for phase, count, start in [("prefill",64,0),("decode",1,64)]:
            out = args.output / f"layer{index}-{phase}"
            out.mkdir()
            positions = torch.arange(start, start+count, device="cuda")[None]
            actual_input = x[:, start:start+count]
            expected, cache = layer(actual_input, weights, index, config, positions, past if phase=="decode" else None)
            for name, value in [("block_input_f16",actual_input),("reference_f16f16_block_output_f16",expected)]:
                padded = torch.zeros((64,2048), device="cuda",dtype=torch.float16)
                padded[:count] = value[0]
                write(out/(name+".bin"),padded)
            cos, sin = rope(config, torch.arange(start,start+64,device="cuda")[None],torch.float16)
            write(out/"rope_cos_f16.bin",cos); write(out/"rope_sin_f16.bin",sin)
            for name, key in [("input","input_layernorm"),("post","post_attention_layernorm")]:
                write(out/f"{name}_norm_weight_f16.bin",weights[f"model.layers.{index}.{key}.weight"])
            # ABI-reserved slots only: Llama native code never consumes Q/K gamma or A8 qparams.
            for name in ["q","k"]:
                write(out/f"{name}_norm_weight_f16.bin",np.zeros(64,dtype="<f2"))
            (out/"qparams_u8.bin").write_bytes(b"".join(struct.pack("<32sfi2f",name.encode(),1.0,0,0.0,255.0) for name in QP_NAMES))
            for name,key in PROJECTIONS.items():
                file=out/f"{name}_weight_f16_hmx.bin"
                if phase=="decode":
                    os.link(args.output/f"layer{index}-prefill"/file.name,file)
                else:
                    write(file,pack_weight(weights[f"model.layers.{index}.{key}.weight"]))
            for part,name in enumerate(["k","v"]):
                initial=torch.zeros((8,80,64),dtype=torch.float16,device="cuda")
                if phase=="decode":initial[:,:64]=past[part][0]
                final=initial.clone(); final[:,:start+count]=cache[part][0]
                write(out/f"kv_cache_{name}_f16.bin",initial)
                write(out/f"reference_kv_cache_{name}_f16.bin",final)
            manifest={"experiment":"L32-0001","recipe":"W16A16","layer":index,"phase":phase,"logical_rows":count,"past_tokens":start,"cache_capacity":80,"reference":"independent Llama math on FP16 teacher layer inputs; not an HMX bit oracle","reserved_metadata":"q/k gamma are zero padding ignored by Llama; A8 qparams are unused ABI slots, not calibrated parameters","files":{f.name:{"bytes":f.stat().st_size,"sha256":sha256(f)} for f in sorted(out.iterdir())}}
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            packages.append({"path":str(out),"manifest_sha256":sha256(out/"manifest.json")})
            print("EXPORTED="+str(out),flush=True)
    (args.output/"provenance.json").write_text(json.dumps({"original":prov,"fixture_sha256":sha256(args.fixture),"packages":packages},indent=2)+"\n")

if __name__=="__main__":main()
