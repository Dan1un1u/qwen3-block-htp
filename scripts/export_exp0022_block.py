#!/usr/bin/env python3
"""Export the real Qwen3-1.7B layer-14 package used by EXP-0022.

The exporter deliberately does not depend on QNN or mllm.  It executes the
original BF16 model through layer 14 with PyTorch, derives the three declared
variant references, and emits hardware-oriented FP16 and signed-W4 weight
carriers plus an auditable manifest.  Staging happens on the WSL ext4 volume;
only the completed package is copied to the Windows archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer


LAYER = 14
M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
RMS_EPS = 1.0e-6

PROMPT = (
    "Explain how low-bit quantization changes memory traffic, matrix "
    "multiplication, calibration error, and pipeline scheduling on a mobile "
    "neural processing unit. Compare weight-only and weight-activation "
    "quantization, then discuss why lower precision does not automatically "
    "produce lower latency. Give a careful hardware-aware explanation with "
    "attention, normalization, residual paths, and feed-forward layers. "
)

PROJECTIONS = {
    "q": f"model.layers.{LAYER}.self_attn.q_proj.weight",
    "k": f"model.layers.{LAYER}.self_attn.k_proj.weight",
    "v": f"model.layers.{LAYER}.self_attn.v_proj.weight",
    "o": f"model.layers.{LAYER}.self_attn.o_proj.weight",
    "gate": f"model.layers.{LAYER}.mlp.gate_proj.weight",
    "up": f"model.layers.{LAYER}.mlp.up_proj.weight",
    "down": f"model.layers.{LAYER}.mlp.down_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/"
            "block_package_layer14_m64"
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0022"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    source_root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def as_numpy_f16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype("<f2", copy=False)


def write_array(
    root: Path,
    name: str,
    array: np.ndarray,
    logical_shape: tuple[int, ...],
    tensors: dict[str, dict[str, object]],
    *,
    layout: str,
) -> None:
    path = root / f"{name}.bin"
    contiguous = np.ascontiguousarray(array)
    with path.open("wb") as stream:
        stream.write(contiguous.tobytes(order="C"))
    tensors[name] = {
        "file": path.name,
        "dtype": str(contiguous.dtype),
        "logical_shape": list(logical_shape),
        "stored_shape": list(contiguous.shape),
        "layout": layout,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def pack_fp16_hmx_weight(weight: torch.Tensor) -> np.ndarray:
    """Pack logical [N,K] into N32/K32 HMX half tiles.

    Each 32x32 tile stores a pair of K rows in each 32-bit word, matching the
    `activation.hf`/`weight.hf` carrier consumed by V79 HMX.
    """

    logical = as_numpy_f16(weight)
    n, k = logical.shape
    if n % 32 or k % 32:
        raise ValueError(f"HMX FP16 shape must be /32, got {(n, k)}")
    tiles = logical.reshape(n // 32, 32, k // 32, 32)
    tiles = tiles.transpose(0, 2, 3, 1)
    tiles = tiles.reshape(n // 32, k // 32, 16, 2, 32)
    return np.ascontiguousarray(tiles.transpose(0, 1, 2, 4, 3))


def quantize_w4_per_output(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    logical = weight.detach().float().cpu()
    max_abs = logical.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    q = torch.round(logical / scale[:, None]).clamp(-7, 7).to(torch.int8)
    n, k = q.shape
    if n % 32 or k % 32:
        raise ValueError(f"HMX W4 shape must be /32, got {(n, k)}")

    physical = q.numpy().reshape(n // 32, 32, k // 32, 32)
    physical = physical.transpose(0, 2, 3, 1)
    physical = physical.reshape(n // 32, k // 32, 8, 4, 32)
    physical = np.ascontiguousarray(physical.transpose(0, 1, 2, 4, 3))
    flat = physical.reshape(n // 32, k // 32, 1024).astype(np.int16)
    nibble = (flat & 0xF).astype(np.uint8)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    dequant = (q.float() * scale[:, None]).to(torch.float16)
    return np.ascontiguousarray(packed), scale.numpy().astype("<f4"), dequant


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return weight * normalized.to(dtype)


def rope_tables(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_THETA ** (indices / HEAD_DIM))
    positions = torch.arange(M, dtype=torch.float32)
    frequency = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequency, frequency), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return torch.cat((-second, first), dim=-1)

    broadcast_cos = cos[None, None, :, :]
    broadcast_sin = sin[None, None, :, :]
    return (
        q * broadcast_cos + rotate_half(q) * broadcast_sin,
        k * broadcast_cos + rotate_half(k) * broadcast_sin,
    )


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repeated_k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
    repeated_v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
    scores = torch.matmul(q, repeated_k.transpose(2, 3)) * (HEAD_DIM ** -0.5)
    causal = torch.triu(torch.ones(M, M, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
    probability = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(probability, repeated_v)
    output = output.transpose(1, 2).contiguous().reshape(1, M, HIDDEN)
    return output, scores, probability


def layer_forward_bf16(
    hidden: torch.Tensor,
    get_weight: Callable[[str], torch.Tensor],
    layer_index: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    capture: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prefix = f"model.layers.{layer_index}"
    boundaries: dict[str, torch.Tensor] = {}
    residual = hidden
    normalized = rms_norm(hidden, get_weight(f"{prefix}.input_layernorm.weight"))
    q_projected = F.linear(
        normalized, get_weight(f"{prefix}.self_attn.q_proj.weight")
    )
    k_projected = F.linear(
        normalized, get_weight(f"{prefix}.self_attn.k_proj.weight")
    )
    v = F.linear(normalized, get_weight(f"{prefix}.self_attn.v_proj.weight"))
    q = rms_norm(
        q_projected.view(1, M, HEADS, HEAD_DIM),
        get_weight(f"{prefix}.self_attn.q_norm.weight"),
    ).transpose(1, 2)
    k = rms_norm(
        k_projected.view(1, M, KV_HEADS, HEAD_DIM),
        get_weight(f"{prefix}.self_attn.k_norm.weight"),
    ).transpose(1, 2)
    v = v.view(1, M, KV_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    attention_output, scores, probability = attention(q, k, v)
    projected = F.linear(
        attention_output, get_weight(f"{prefix}.self_attn.o_proj.weight")
    )
    post_attention_residual = residual + projected
    post_normalized = rms_norm(
        post_attention_residual,
        get_weight(f"{prefix}.post_attention_layernorm.weight"),
    )
    gate = F.linear(post_normalized, get_weight(f"{prefix}.mlp.gate_proj.weight"))
    up = F.linear(post_normalized, get_weight(f"{prefix}.mlp.up_proj.weight"))
    middle = F.silu(gate) * up
    down = F.linear(middle, get_weight(f"{prefix}.mlp.down_proj.weight"))
    output = post_attention_residual + down

    if capture:
        boundaries = {
            "block_input": hidden,
            "input_norm": normalized,
            "q_projection": q_projected,
            "k_projection": k_projected,
            "q_rope": q.transpose(1, 2).contiguous().reshape(1, M, HIDDEN),
            "k_rope": k.transpose(1, 2).contiguous().reshape(1, M, KV_HEADS * HEAD_DIM),
            "v": v.transpose(1, 2).contiguous().reshape(1, M, KV_HEADS * HEAD_DIM),
            "attention_scores": scores,
            "attention_probability": probability,
            "attention_concat": attention_output,
            "attention_projection": projected,
            "post_attention_residual": post_attention_residual,
            "post_attention_norm": post_normalized,
            "gate": gate,
            "up": up,
            "middle": middle,
            "down": down,
            "block_output": output,
        }
    return output, boundaries


def cast_half(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().to(torch.float16)


def linear_half(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.float(), weight.float().transpose(-1, -2)).to(torch.float16)


def layer_forward_f16(
    block_input: torch.Tensor,
    weights: dict[str, torch.Tensor],
    norm_weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = cast_half(block_input)
    normalized = rms_norm(hidden, norm_weights["input"])
    q_projected = linear_half(normalized, weights["q"])
    k_projected = linear_half(normalized, weights["k"])
    v = linear_half(normalized, weights["v"])
    q = rms_norm(q_projected.view(1, M, HEADS, HEAD_DIM), norm_weights["q"]).transpose(1, 2)
    k = rms_norm(k_projected.view(1, M, KV_HEADS, HEAD_DIM), norm_weights["k"]).transpose(1, 2)
    v = v.view(1, M, KV_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    attention_output, scores, probability = attention(q, k, v)
    projected = linear_half(attention_output, weights["o"])
    post_attention_residual = cast_half(hidden + projected)
    post_normalized = rms_norm(post_attention_residual, norm_weights["post"])
    gate = linear_half(post_normalized, weights["gate"])
    up = linear_half(post_normalized, weights["up"])
    middle = cast_half(F.silu(gate.float()) * up.float())
    down = linear_half(middle, weights["down"])
    output = cast_half(post_attention_residual + down)
    boundaries = {
        "block_input": hidden,
        "input_norm": normalized,
        "q_projection": q_projected,
        "k_projection": k_projected,
        "q_rope": q.transpose(1, 2).contiguous().reshape(1, M, HIDDEN),
        "k_rope": k.transpose(1, 2).contiguous().reshape(1, M, KV_HEADS * HEAD_DIM),
        "v": v.transpose(1, 2).contiguous().reshape(1, M, KV_HEADS * HEAD_DIM),
        "attention_scores": scores,
        "attention_probability": probability,
        "attention_concat": attention_output,
        "attention_projection": projected,
        "post_attention_residual": post_attention_residual,
        "post_attention_norm": post_normalized,
        "gate": gate,
        "up": up,
        "middle": middle,
        "down": down,
        "block_output": output,
    }
    return output, boundaries


def asymmetric_qparam(tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().float()
    minimum = float(values.amin())
    maximum = float(values.amax())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("non-finite calibration tensor")
    if maximum <= minimum:
        scale = 1.0
        zero_point = 0
    else:
        scale = (maximum - minimum) / 255.0
        zero_point = int(round(-minimum / scale))
        zero_point = min(255, max(0, zero_point))
    return {
        "scale": scale,
        "zero_point": zero_point,
        "minimum": minimum,
        "maximum": maximum,
        "observer": "per_tensor_asymmetric_minmax_u8",
    }


def quantize_u8(tensor: torch.Tensor, qparam: dict[str, object]) -> torch.Tensor:
    scale = float(qparam["scale"])
    zero_point = int(qparam["zero_point"])
    return torch.round(tensor.float() / scale + zero_point).clamp(0, 255).to(torch.uint8)


def dequantize_u8(tensor: torch.Tensor, qparam: dict[str, object]) -> torch.Tensor:
    return (tensor.float() - int(qparam["zero_point"])) * float(qparam["scale"])


def qdq_half(tensor: torch.Tensor, qparam: dict[str, object]) -> torch.Tensor:
    return dequantize_u8(quantize_u8(tensor, qparam), qparam).to(torch.float16)


def layer_forward_w4u8(
    block_input: torch.Tensor,
    weights: dict[str, torch.Tensor],
    norm_weights: dict[str, torch.Tensor],
    qparams: dict[str, dict[str, object]],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    encoded: dict[str, torch.Tensor] = {}

    def boundary(name: str, value: torch.Tensor) -> torch.Tensor:
        encoded[name] = quantize_u8(value, qparams[name])
        return dequantize_u8(encoded[name], qparams[name]).to(torch.float16)

    hidden = boundary("block_input", block_input)
    normalized = boundary("input_norm", rms_norm(hidden, norm_weights["input"]))
    q_projected = boundary("q_projection", linear_half(normalized, weights["q"]))
    k_projected = boundary("k_projection", linear_half(normalized, weights["k"]))
    v_flat = boundary("v", linear_half(normalized, weights["v"]))
    q = rms_norm(q_projected.view(1, M, HEADS, HEAD_DIM), norm_weights["q"]).transpose(1, 2)
    k = rms_norm(k_projected.view(1, M, KV_HEADS, HEAD_DIM), norm_weights["k"]).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    q = boundary("q_rope", q.transpose(1, 2).contiguous().reshape(1, M, HIDDEN))
    k = boundary(
        "k_rope",
        k.transpose(1, 2).contiguous().reshape(1, M, KV_HEADS * HEAD_DIM),
    )
    q = q.view(1, M, HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(1, M, KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v_flat.view(1, M, KV_HEADS, HEAD_DIM).transpose(1, 2)
    attention_output, scores, probability = attention(q, k, v)
    attention_output = boundary("attention_concat", attention_output)
    projected = boundary("attention_projection", linear_half(attention_output, weights["o"]))
    post_attention_residual = boundary(
        "post_attention_residual", hidden + projected
    )
    post_normalized = boundary(
        "post_attention_norm", rms_norm(post_attention_residual, norm_weights["post"])
    )
    gate = boundary("gate", linear_half(post_normalized, weights["gate"]))
    up = boundary("up", linear_half(post_normalized, weights["up"]))
    middle = boundary("middle", F.silu(gate.float()) * up.float())
    down = boundary("down", linear_half(middle, weights["down"]))
    output = boundary("block_output", post_attention_residual + down)
    boundaries = {
        "block_input": hidden,
        "input_norm": normalized,
        "q_projection": q_projected,
        "k_projection": k_projected,
        "q_rope": q.transpose(1, 2).contiguous().reshape(1, M, HIDDEN),
        "k_rope": k.transpose(1, 2).contiguous().reshape(1, M, KV_HEADS * HEAD_DIM),
        "v": v_flat,
        "attention_scores": scores,
        "attention_probability": probability,
        "attention_concat": attention_output,
        "attention_projection": projected,
        "post_attention_residual": post_attention_residual,
        "post_attention_norm": post_normalized,
        "gate": gate,
        "up": up,
        "middle": middle,
        "down": down,
        "block_output": output,
    }
    return output, boundaries, encoded


def error_metrics(actual: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    a = actual.detach().float().reshape(-1)
    b = teacher.detach().float().reshape(-1)
    difference = (a - b).abs()
    cosine = float(F.cosine_similarity(a[None, :], b[None, :]).item())
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2))),
        "cosine": cosine,
    }


def export(args: argparse.Namespace) -> None:
    model = args.model.resolve()
    output = args.output.resolve()
    staging_root = args.staging_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="package-", dir=staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    tensors: dict[str, dict[str, object]] = {}

    try:
        tokenizer = Tokenizer.from_file(str(model / "qwen3-tokenizer.json"))
        token_ids = tokenizer.encode(PROMPT * 2, add_special_tokens=True).ids[:M]
        if len(token_ids) != M:
            raise ValueError(f"prompt produced {len(token_ids)} tokens, expected {M}")
        input_ids = torch.tensor(token_ids, dtype=torch.long)[None, :]

        index_path = model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index["weight_map"]["model.embed_tokens.weight"]
        shard = model / shard_name
        required = ["model.embed_tokens.weight"]
        for layer_index in range(LAYER + 1):
            prefix = f"model.layers.{layer_index}"
            required.extend(
                [
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight",
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ]
            )
        wrong_shard = [name for name in required if index["weight_map"][name] != shard_name]
        if wrong_shard:
            raise ValueError(f"required weights span multiple shards: {wrong_shard[:3]}")

        with safe_open(shard, framework="pt", device="cpu") as weights_file:
            get_weight = weights_file.get_tensor
            hidden = F.embedding(input_ids, get_weight("model.embed_tokens.weight"))
            cos_bf16, sin_bf16 = rope_tables(torch.bfloat16)
            teacher_boundaries: dict[str, torch.Tensor] = {}
            for layer_index in range(LAYER + 1):
                hidden, captured = layer_forward_bf16(
                    hidden,
                    get_weight,
                    layer_index,
                    cos_bf16,
                    sin_bf16,
                    layer_index == LAYER,
                )
                if captured:
                    teacher_boundaries = captured

            original_weights = {
                name: get_weight(key).to(torch.float16)
                for name, key in PROJECTIONS.items()
            }
            norm_weights = {
                "input": get_weight(
                    f"model.layers.{LAYER}.input_layernorm.weight"
                ).to(torch.float16),
                "post": get_weight(
                    f"model.layers.{LAYER}.post_attention_layernorm.weight"
                ).to(torch.float16),
                "q": get_weight(
                    f"model.layers.{LAYER}.self_attn.q_norm.weight"
                ).to(torch.float16),
                "k": get_weight(
                    f"model.layers.{LAYER}.self_attn.k_norm.weight"
                ).to(torch.float16),
            }

        cos_f16, sin_f16 = rope_tables(torch.float16)
        block_input = teacher_boundaries["block_input"]
        f16_output, f16_boundaries = layer_forward_f16(
            block_input,
            original_weights,
            norm_weights,
            cos_f16,
            sin_f16,
        )

        w4_weights: dict[str, torch.Tensor] = {}
        w4_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, weight in original_weights.items():
            packed, scale, dequant = quantize_w4_per_output(weight)
            w4_payloads[name] = (packed, scale)
            w4_weights[name] = dequant

        w4f16_output, w4f16_boundaries = layer_forward_f16(
            block_input,
            w4_weights,
            norm_weights,
            cos_f16,
            sin_f16,
        )

        qparam_names = [
            "block_input",
            "input_norm",
            "q_projection",
            "k_projection",
            "q_rope",
            "k_rope",
            "v",
            "attention_concat",
            "attention_projection",
            "post_attention_residual",
            "post_attention_norm",
            "gate",
            "up",
            "middle",
            "down",
            "block_output",
        ]
        qparams = {
            name: asymmetric_qparam(teacher_boundaries[name])
            for name in qparam_names
        }
        qparams["attention_probability"] = {
            "scale": 1.0 / 255.0,
            "zero_point": 0,
            "minimum": 0.0,
            "maximum": 1.0,
            "observer": "fixed_probability_u8",
        }
        w4u8_output, w4u8_boundaries, w4u8_encoded = layer_forward_w4u8(
            block_input,
            w4_weights,
            norm_weights,
            qparams,
            cos_f16,
            sin_f16,
        )

        write_array(
            staging,
            "block_input_f16",
            as_numpy_f16(block_input),
            tuple(block_input.shape),
            tensors,
            layout="row_major",
        )
        write_array(
            staging,
            "teacher_block_output_bf16_as_f16",
            as_numpy_f16(teacher_boundaries["block_output"]),
            tuple(teacher_boundaries["block_output"].shape),
            tensors,
            layout="row_major",
        )
        for name, weight in norm_weights.items():
            write_array(
                staging,
                f"{name}_norm_weight_f16",
                as_numpy_f16(weight),
                tuple(weight.shape),
                tensors,
                layout="row_major",
            )
        write_array(
            staging,
            "rope_cos_f16",
            as_numpy_f16(cos_f16),
            tuple(cos_f16.shape),
            tensors,
            layout="position_head_dim",
        )
        write_array(
            staging,
            "rope_sin_f16",
            as_numpy_f16(sin_f16),
            tuple(sin_f16.shape),
            tensors,
            layout="position_head_dim",
        )

        for name, weight in original_weights.items():
            write_array(
                staging,
                f"{name}_weight_f16_hmx",
                pack_fp16_hmx_weight(weight),
                tuple(weight.shape),
                tensors,
                layout="hmx_fp16_n32_k32_kpair_interleaved",
            )
            packed, scale = w4_payloads[name]
            write_array(
                staging,
                f"{name}_weight_w4_hmx",
                packed,
                tuple(weight.shape),
                tensors,
                layout="hmx_s8_n32_k32_k4_interleaved_packed_nibbles",
            )
            write_array(
                staging,
                f"{name}_weight_w4_scale_f32",
                scale,
                (weight.shape[0],),
                tensors,
                layout="per_output_channel",
            )

        references = {
            "teacher": teacher_boundaries,
            "f16f16": f16_boundaries,
            "w4f16": w4f16_boundaries,
            "w4u8": w4u8_boundaries,
        }
        for variant, boundaries in references.items():
            for name, tensor in boundaries.items():
                write_array(
                    staging,
                    f"reference_{variant}_{name}_f16",
                    as_numpy_f16(tensor),
                    tuple(tensor.shape),
                    tensors,
                    layout="row_major",
                )
        for name, encoded in w4u8_encoded.items():
            write_array(
                staging,
                f"reference_w4u8_{name}_u8",
                encoded.cpu().numpy().astype(np.uint8, copy=False),
                tuple(encoded.shape),
                tensors,
                layout="row_major",
            )

        qparam_record = struct.Struct("<32sfi2f")
        qparam_path = staging / "qparams_u8.bin"
        with qparam_path.open("wb") as stream:
            for name in sorted(qparams):
                encoded_name = name.encode("ascii")
                if len(encoded_name) >= 32:
                    raise ValueError(f"qparam name too long: {name}")
                record = qparams[name]
                stream.write(
                    qparam_record.pack(
                        encoded_name,
                        float(record["scale"]),
                        int(record["zero_point"]),
                        float(record["minimum"]),
                        float(record["maximum"]),
                    )
                )
        tensors["qparams_u8"] = {
            "file": qparam_path.name,
            "dtype": "record<name32,f32,i32,f32,f32>",
            "logical_shape": [len(qparams)],
            "stored_shape": [len(qparams)],
            "layout": "sorted_by_ascii_name",
            "bytes": qparam_path.stat().st_size,
            "sha256": sha256_file(qparam_path),
        }

        metrics = {
            "f16f16_vs_bf16_teacher": error_metrics(
                f16_output, teacher_boundaries["block_output"]
            ),
            "w4f16_vs_bf16_teacher": error_metrics(
                w4f16_output, teacher_boundaries["block_output"]
            ),
            "w4u8_vs_bf16_teacher": error_metrics(
                w4u8_output, teacher_boundaries["block_output"]
            ),
        }
        manifest = {
            "experiment": "EXP-0022",
            "package_abi": 1,
            "source_commit": git_head(),
            "source_model": str(model),
            "source_model_index_sha256": sha256_file(index_path),
            "source_shard": shard_name,
            "source_shard_sha256": sha256_file(shard),
            "layer": LAYER,
            "sequence_length": M,
            "positions": [0, M - 1],
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rope_theta": ROPE_THETA,
            "rms_norm_eps": RMS_EPS,
            "prompt": PROMPT * 2,
            "token_ids": token_ids,
            "variants": {
                "F16F16": "original FP16 HMX weight carrier and FP16 activations",
                "W4F16": "shared per-output-channel signed W4 carrier expanded to FP16 HMX",
                "W4U8": "same W4 carrier/scales with asymmetric U8 projection boundaries",
            },
            "w4_contract": {
                "qmin": -7,
                "qmax": 7,
                "scale_granularity": "per_output_channel",
                "shared_carrier_between": ["W4F16", "W4U8"],
            },
            "w4_identity": {
                name: {
                    "carrier_sha256": tensors[f"{name}_weight_w4_hmx"]["sha256"],
                    "scale_sha256": tensors[f"{name}_weight_w4_scale_f32"]["sha256"],
                }
                for name in PROJECTIONS
            },
            "u8_qparams": qparams,
            "reference_metrics": metrics,
            "tensors": tensors,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        package_digest = hashlib.sha256()
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            package_digest.update(path.name.encode("utf-8"))
            package_digest.update(bytes.fromhex(sha256_file(path)))
        summary = {
            "experiment": "EXP-0022",
            "package": str(output),
            "package_digest": package_digest.hexdigest(),
            "tensor_count": len(tensors),
            "total_tensor_bytes": sum(int(item["bytes"]) for item in tensors.values()),
            "reference_metrics": metrics,
        }
        (staging / "export_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        if publish.exists():
            raise FileExistsError(f"stale publishing directory exists: {publish}")
        shutil.copytree(staging, publish)
        os.rename(publish, output)
        print(json.dumps(summary, sort_keys=True))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    export(parse_args())


if __name__ == "__main__":
    main()
