#!/usr/bin/env python3
"""Independent Llama math, original checkpoint provenance and FP16 HMX export.

No Qwen exporters or model helpers are imported. The Transformers implementation
is only used by the validation command as a separate oracle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXPECTED = {
    "config.json": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb",
    "generation_config.json": "88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36",
    "model.safetensors": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
}
PROJECTIONS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj",
               "v": "self_attn.v_proj", "o": "self_attn.o_proj",
               "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def provenance(root):
    files = {}
    for name, expected in EXPECTED.items():
        path = root / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Original checkpoint hash mismatch: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return {"source": "https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct",
            "original_root": str(root), "files": files}


def rope(config, positions, dtype):
    dim = config["head_dim"]
    inv = 1.0 / config["rope_theta"] ** (torch.arange(0, dim, 2, device=positions.device).float() / dim)
    r = config["rope_scaling"]
    wave = 2.0 * math.pi / inv
    blend = (r["original_max_position_embeddings"] / wave - r["low_freq_factor"]) / (r["high_freq_factor"] - r["low_freq_factor"])
    scaled = (1.0 - blend) * inv / r["factor"] + blend * inv
    inv = torch.where(wave > r["original_max_position_embeddings"] / r["low_freq_factor"], inv / r["factor"],
                      torch.where(wave < r["original_max_position_embeddings"] / r["high_freq_factor"], inv, scaled))
    angle = positions.float().unsqueeze(-1) * inv
    angle = torch.cat((angle, angle), -1)
    return angle.cos().to(dtype), angle.sin().to(dtype)


def rms(x, weight, eps):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * weight


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat((-x[..., half:], x[..., :half]), -1) * sin[:, None]


def layer(x, weights, index, config, positions, cache=None):
    w = {k.removeprefix(f"model.layers.{index}."): v for k, v in weights.items() if k.startswith(f"model.layers.{index}.")}
    norm = rms(x, w["input_layernorm.weight"], config["rms_norm_eps"])
    batch, count, _ = x.shape
    dim, heads, kvheads = config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
    cos, sin = rope(config, positions, x.dtype)
    q = rotate(F.linear(norm, w["self_attn.q_proj.weight"]).view(batch, count, heads, dim).transpose(1, 2), cos, sin)
    k = rotate(F.linear(norm, w["self_attn.k_proj.weight"]).view(batch, count, kvheads, dim).transpose(1, 2), cos, sin)
    v = F.linear(norm, w["self_attn.v_proj.weight"]).view(batch, count, kvheads, dim).transpose(1, 2)
    past = 0 if cache is None else cache[0].shape[2]
    if cache is not None:
        k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
    kk, vv = k.repeat_interleave(heads // kvheads, 1), v.repeat_interleave(heads // kvheads, 1)
    scores = (q @ kk.transpose(-1, -2)) * (dim ** -0.5)
    mask = torch.arange(k.shape[2], device=x.device)[None, :] > (past + torch.arange(count, device=x.device))[:, None]
    scores = scores.masked_fill(mask, torch.finfo(x.dtype).min)
    prob = scores.float().softmax(-1).to(x.dtype)
    av = (prob @ vv).transpose(1, 2).contiguous().view(batch, count, heads * dim)
    residual = x + F.linear(av, w["self_attn.o_proj.weight"])
    post = rms(residual, w["post_attention_layernorm.weight"], config["rms_norm_eps"])
    middle = F.silu(F.linear(post, w["mlp.gate_proj.weight"])) * F.linear(post, w["mlp.up_proj.weight"])
    output = residual + F.linear(middle, w["mlp.down_proj.weight"])
    return output, (k, v)


def pack_weight(weight):
    a = weight.detach().cpu().half().numpy()
    n, k = a.shape
    if n % 32 or k % 32:
        raise ValueError(f"Non-tile-aligned weight: {a.shape}")
    return np.ascontiguousarray(a.reshape(n // 32, 32, k // 32, 16, 2).transpose(0, 2, 3, 1, 4))


def metrics(a, b):
    a, b = a.detach().float(), b.detach().float()
    return {"max_abs": (a-b).abs().max().item(),
            "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-20)).item(),
            "finite": bool(a.isfinite().all() and b.isfinite().all())}


@torch.inference_mode()
def validate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {"experiment": "L32-0001", "provenance": provenance(args.model),
              "scope": "host independent math, not HMX arithmetic or device acceptance",
              "thresholds": {"float32_relative_l2": 0.00002, "float16_relative_l2": 0.005}, "cases": []}
    config = json.loads((args.model / "config.json").read_text())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    passage = "The river flows through the city. Scientists measure light and study the natural world. A good explanation connects evidence with a clear conclusion. " * 8
    token_ids = tokenizer.encode(passage, add_special_tokens=True)[:65]
    (args.output / "fixture.json").write_text(json.dumps({"text": passage, "ids": token_ids, "position_offsets": [0, 8192, 32768], "decode_past": 64, "seed": 0}, indent=2))
    report["fixture_sha256"] = sha256(args.output / "fixture.json")
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True).to(args.device).eval()
    ids = torch.tensor([token_ids], device=args.device)
    for dtype in [torch.float32, torch.float16]:
        teacher.to(dtype=dtype)
        weights = teacher.state_dict()
        for offset in [0, 8192, 32768]:
            pos = torch.arange(offset, offset + 64, device=args.device)[None]
            ref = teacher(ids[:, :64], position_ids=pos, use_cache=True, output_hidden_states=True)
            x = F.embedding(ids[:, :64], weights["model.embed_tokens.weight"])
            caches, layer_metrics = [], []
            for index in range(config["num_hidden_layers"]):
                x, cache = layer(x, weights, index, config, pos)
                caches.append(cache)
                if index + 1 < config["num_hidden_layers"]:
                    layer_metrics.append(metrics(x, ref.hidden_states[index+1]))
            logits = F.linear(rms(x, weights["model.norm.weight"], config["rms_norm_eps"]), weights["model.embed_tokens.weight"])
            record = {"dtype": str(dtype), "position_offset": offset, "layers": layer_metrics, "prefill_logits": metrics(logits, ref.logits)}
            dx = F.embedding(ids[:, 64:], weights["model.embed_tokens.weight"])
            dpos = torch.tensor([[offset + 64]], device=args.device)
            dec = teacher(ids[:, 64:], position_ids=dpos, past_key_values=ref.past_key_values, use_cache=True)
            for index in range(config["num_hidden_layers"]):
                dx, _ = layer(dx, weights, index, config, dpos, caches[index])
            dlogits = F.linear(rms(dx, weights["model.norm.weight"], config["rms_norm_eps"]), weights["model.embed_tokens.weight"])
            record["decode_logits"] = metrics(dlogits, dec.logits)
            limit = report["thresholds"]["float32_relative_l2" if dtype == torch.float32 else "float16_relative_l2"]
            record["pass"] = all(m["finite"] and m["relative_l2"] <= limit for m in layer_metrics + [record["prefill_logits"], record["decode_logits"]])
            report["cases"].append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='layers'}), flush=True)
            del ref, dec, logits, dlogits, caches
    prompt_ids = tokenizer.apply_chat_template([{"role":"user", "content":"What is the capital of France? Answer briefly."}], tokenize=True, add_generation_prompt=True, date_string="10 Sep 2026", return_tensors="pt").to(args.device)
    generated = teacher.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), do_sample=False, max_new_tokens=32, pad_token_id=128001)
    report["fp16_teacher_text"] = tokenizer.decode(generated[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    report["fp16_teacher_generated_ids"] = generated[0, prompt_ids.shape[1]:].tolist()
    report["pass"] = all(c["pass"] for c in report["cases"])
    (args.output / "reference_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass":report["pass"],"teacher_text":report["fp16_teacher_text"]}), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["validate"])
    p.add_argument("--model", type=Path, default=Path("/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    validate(args)

if __name__ == "__main__":
    main()
