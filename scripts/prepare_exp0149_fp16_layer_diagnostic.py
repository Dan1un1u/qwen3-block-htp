#!/usr/bin/env python3
"""Prepare one real EXP-0149 FP16 layer for chained device diagnosis.

This helper is deliberately outside the formal vertical-slice execution
contract.  It accepts the actual FP16 output captured from the preceding DSP
layer, builds an independent software reference for the next layer, and emits
a normal single-layer package.  Running layers 13, 14 and 15 in sequence lets
us locate the first local numerical divergence without changing the resident
three-layer implementation or its zero-intermediate-DDR contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import export_exp0022_block as base


M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
KV_HEADS = 8
HEAD_DIM = 128
LAYERS = (13, 14, 15)
PROJECTION_SHAPES = {
    "q": (HIDDEN, HIDDEN),
    "k": (KV_HEADS * HEAD_DIM, HIDDEN),
    "v": (KV_HEADS * HEAD_DIM, HIDDEN),
    "o": (HIDDEN, HIDDEN),
    "gate": (INTERMEDIATE, HIDDEN),
    "up": (INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    parser.add_argument("--layer", type=int, choices=LAYERS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--exp0149-root",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0149"),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0149-repair"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_keys(layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}"
    return {
        "q": f"{prefix}.self_attn.q_proj.weight",
        "k": f"{prefix}.self_attn.k_proj.weight",
        "v": f"{prefix}.self_attn.v_proj.weight",
        "o": f"{prefix}.self_attn.o_proj.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "down": f"{prefix}.mlp.down_proj.weight",
    }


def load_logical_weights(
    model: Path, layer: int, recipe: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    keys = projection_keys(layer)
    prefix = f"model.layers.{layer}"
    required = [
        *keys.values(),
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
    ]
    shards = {index["weight_map"][name] for name in required}
    if len(shards) != 1:
        raise ValueError(f"layer {layer} spans shards: {sorted(shards)}")
    with safe_open(model / next(iter(shards)), framework="pt", device="cpu") as source:
        original = {
            name: source.get_tensor(key).to(torch.float16).cpu()
            for name, key in keys.items()
        }
        norms = {
            "input": source.get_tensor(
                f"{prefix}.input_layernorm.weight"
            ).to(torch.float16).cpu(),
            "post": source.get_tensor(
                f"{prefix}.post_attention_layernorm.weight"
            ).to(torch.float16).cpu(),
            "q": source.get_tensor(
                f"{prefix}.self_attn.q_norm.weight"
            ).to(torch.float16).cpu(),
            "k": source.get_tensor(
                f"{prefix}.self_attn.k_norm.weight"
            ).to(torch.float16).cpu(),
        }
    if recipe == "f16f16":
        return original, norms
    dequantized: dict[str, torch.Tensor] = {}
    for name, weight in original.items():
        _, _, dequantized[name] = base.quantize_w4_per_output(weight)
    return dequantized, norms


def write(path: Path, value: np.ndarray) -> None:
    np.ascontiguousarray(value).tofile(path)


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    source = (args.exp0149_root.resolve() / args.recipe / f"layer{args.layer}")
    input_path = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace diagnostic package: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)

    block_input = np.fromfile(input_path, dtype="<f2")
    if block_input.size != M * HIDDEN:
        raise ValueError(
            f"{input_path}: expected {M * HIDDEN} FP16 values, got {block_input.size}"
        )
    block_input = block_input.reshape(1, M, HIDDEN)
    weights, norms = load_logical_weights(model, args.layer, args.recipe)
    base.M = M
    cos, sin = base.rope_tables(torch.float16)
    output_tensor, boundaries = base.layer_forward_f16(
        torch.from_numpy(block_input.copy()), weights, norms, cos, sin
    )

    cache_k = np.ascontiguousarray(
        boundaries["k_rope"]
        .detach().cpu().numpy().astype("<f2", copy=False)
        .reshape(M, KV_HEADS, HEAD_DIM)
        .transpose(1, 0, 2)
    )
    cache_v = np.ascontiguousarray(
        boundaries["v"]
        .detach().cpu().numpy().astype("<f2", copy=False)
        .reshape(M, KV_HEADS, HEAD_DIM)
        .transpose(1, 0, 2)
    )
    empty_cache = np.zeros_like(cache_k)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="layer-", dir=args.staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        write(staging / "block_input_f16.bin", block_input.astype("<f2"))
        write(
            staging / f"reference_{args.recipe}_block_output_f16.bin",
            output_tensor.detach().cpu().numpy().astype("<f2", copy=False),
        )
        write(
            staging / "rope_cos_f16.bin",
            cos.detach().cpu().numpy().astype("<f2", copy=False),
        )
        write(
            staging / "rope_sin_f16.bin",
            sin.detach().cpu().numpy().astype("<f2", copy=False),
        )
        write(staging / "kv_cache_k_f16.bin", empty_cache)
        write(staging / "kv_cache_v_f16.bin", empty_cache)
        write(staging / "reference_kv_cache_k_f16.bin", cache_k)
        write(staging / "reference_kv_cache_v_f16.bin", cache_v)

        manifest = {
            "experiment": "EXP-0149",
            "role": "diagnostic_only_single_layer_chain",
            "recipe": args.recipe,
            "layer": args.layer,
            "input_source": str(input_path),
            "shape_scan": {
                "cell": f"diagnostic_layer{args.layer}_prefill_m64",
                "recipe": args.recipe,
                "mode": "prefill",
                "logical_m": M,
                "physical_chunks": 1,
                "initial_kv_length": 0,
                "kv_cache_capacity": M,
            },
            "contract": {
                "formal_vertical_slice": False,
                "purpose": "localize_FP16_composition_error",
                "actual_preceding_layer_output_used_as_input": args.layer != 13,
            },
        }
        manifest["files"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(json.dumps({"output": str(output), **manifest}, indent=2))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    main()
