#!/usr/bin/env python3
"""Export the real Qwen3 layers 13 -> 14 -> 15 replay slice for EXP-0149.

Only the hidden state entering layer 13 is taken from the BF16 teacher.  Every
recipe then executes all three layers consecutively, so the references exercise
the same layer-to-layer dependency as the standalone DSP slice.  The package
contains empty, independent cache objects for all three layers and final
capacity-72 cache references; runtime K/V import is never used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import export_exp0022_block as base  # noqa: E402
from export_exp0148_replay import (  # noqa: E402
    integer_attention,
    load_qparams,
    w4u8_output_from_attention,
)
from prepare_exp0042_attention import (  # noqa: E402
    ABI_VERSION,
    CONFIG,
    DIVISION_MODES,
    choose_carrier,
)


PREFILL_M = 64
DECODE_STEPS = 8
TOTAL_M = PREFILL_M + DECODE_STEPS
PHYSICAL_M = 64
LAYERS = (13, 14, 15)
DECLARED_LAYERS = 28
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS
HMX_N = 32
HMX_BIAS_BYTES = 256

QPARAM_NAMES = (
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
)

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
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--exp0148-root",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0148"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0149"),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0149"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(value).tofile(path)


def f16(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype("<f2", copy=False)


def layer_projection_keys(layer: int) -> dict[str, str]:
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


def prefill_view(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError(value.shape)
    if value.shape[1] == TOTAL_M:
        return value[:, :PREFILL_M]
    if value.ndim >= 3 and value.shape[-2] == TOTAL_M:
        slices = [slice(None)] * value.ndim
        slices[-2] = slice(0, PREFILL_M)
        slices[-1] = slice(0, PREFILL_M)
        return value[tuple(slices)]
    raise ValueError(f"cannot select prefill rows from {tuple(value.shape)}")


def load_teacher_and_weights(
    model: Path,
) -> tuple[
    list[int],
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
]:
    tokenizer = Tokenizer.from_file(str(model / "qwen3-tokenizer.json"))
    token_ids = tokenizer.encode(
        base.PROMPT * 2, add_special_tokens=True
    ).ids[:TOTAL_M]
    if len(token_ids) != TOTAL_M:
        raise ValueError(
            f"prompt produced {len(token_ids)} tokens, expected {TOTAL_M}"
        )
    input_ids = torch.tensor(token_ids, dtype=torch.long)[None, :]
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_name = index["weight_map"]["model.embed_tokens.weight"]
    required = ["model.embed_tokens.weight"]
    for layer in range(LAYERS[-1] + 1):
        prefix = f"model.layers.{layer}"
        required.extend(
            f"{prefix}.{suffix}"
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
    wrong = [
        name for name in required if index["weight_map"][name] != shard_name
    ]
    if wrong:
        raise ValueError(f"teacher weights span shards: {wrong[:3]}")

    boundaries: dict[int, dict[str, torch.Tensor]] = {}
    original_weights: dict[int, dict[str, torch.Tensor]] = {}
    norm_weights: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(
        model / shard_name, framework="pt", device="cpu"
    ) as source:
        get_weight = source.get_tensor
        hidden = F.embedding(
            input_ids, get_weight("model.embed_tokens.weight")
        )
        cos_bf16, sin_bf16 = base.rope_tables(torch.bfloat16)
        for layer in range(LAYERS[-1] + 1):
            hidden, captured = base.layer_forward_bf16(
                hidden, get_weight, layer, cos_bf16, sin_bf16,
                layer in LAYERS,
            )
            if captured:
                boundaries[layer] = {
                    name: value.detach().cpu()
                    for name, value in captured.items()
                }
        for layer in LAYERS:
            keys = layer_projection_keys(layer)
            original_weights[layer] = {
                name: get_weight(key).to(torch.float16).cpu()
                for name, key in keys.items()
            }
            prefix = f"model.layers.{layer}"
            norm_weights[layer] = {
                "input": get_weight(
                    f"{prefix}.input_layernorm.weight"
                ).to(torch.float16).cpu(),
                "post": get_weight(
                    f"{prefix}.post_attention_layernorm.weight"
                ).to(torch.float16).cpu(),
                "q": get_weight(
                    f"{prefix}.self_attn.q_norm.weight"
                ).to(torch.float16).cpu(),
                "k": get_weight(
                    f"{prefix}.self_attn.k_norm.weight"
                ).to(torch.float16).cpu(),
            }
    return token_ids, boundaries, original_weights, norm_weights


def derive_qparams(
    teacher: dict[int, dict[str, torch.Tensor]],
    retained_layer14: dict[str, dict[str, object]],
) -> dict[int, dict[str, dict[str, object]]]:
    result: dict[int, dict[str, dict[str, object]]] = {}
    for layer in LAYERS:
        if layer == 14:
            result[layer] = copy.deepcopy(retained_layer14)
            continue
        result[layer] = {
            name: base.asymmetric_qparam(
                prefill_view(teacher[layer][name])
            )
            for name in QPARAM_NAMES
        }
        result[layer]["attention_probability"] = {
            "scale": 1.0 / 255.0,
            "zero_point": 0,
            "minimum": 0.0,
            "maximum": 1.0,
            "observer": "fixed_probability_u8",
        }

    # Preserve EXP-0148 layer-14 encodings and make both resident handoffs
    # byte-preserving.  There is no hidden U8 -> float -> U8 conversion in the
    # DSP path, so producer output and consumer input must have one encoding.
    result[13]["block_output"] = copy.deepcopy(
        result[14]["block_input"]
    )
    result[15]["block_input"] = copy.deepcopy(
        result[14]["block_output"]
    )
    return result


def build_attention_config(
    encoded_v: np.ndarray,
    qparams: dict[str, dict[str, object]],
) -> bytes:
    q_qp = qparams["q_rope"]
    k_qp = qparams["k_rope"]
    v_qp = qparams["v"]
    p_qp = qparams["attention_probability"]
    y_qp = qparams["attention_concat"]
    score_step = math.log(2.0) / (1 << 3)
    score_ratio = (
        float(q_qp["scale"])
        * float(k_qp["scale"])
        / math.sqrt(HEAD_DIM)
        / score_step
    )
    score_multiplier, score_shift, _ = choose_carrier(score_ratio)
    records: list[bytes] = []
    for group in range(KV_HEADS):
        centered = (
            encoded_v[:PREFILL_M, group].astype(np.int32)
            - int(v_qp["zero_point"])
        )
        denominator = max(
            1, int(np.max(np.abs(centered.astype(np.int64))))
        )
        numerator = 127
        v_s8_scale = (
            float(v_qp["scale"]) * denominator / numerator
        )
        av_ratio = (
            float(p_qp["scale"])
            * v_s8_scale
            / float(y_qp["scale"])
        )
        av_multiplier, av_shift, _ = choose_carrier(av_ratio)
        records.append(
            CONFIG.pack(
                ABI_VERSION,
                group,
                3,
                DIVISION_MODES["sole"],
                int(q_qp["zero_point"]),
                int(k_qp["zero_point"]),
                int(v_qp["zero_point"]),
                int(p_qp["zero_point"]),
                int(y_qp["zero_point"]),
                numerator,
                denominator,
                score_shift,
                score_multiplier,
                av_shift,
                av_multiplier,
            )
        )
    return b"".join(records)


def build_silu_lut(
    qparams: dict[str, dict[str, object]],
) -> np.ndarray:
    gate_values = np.arange(256, dtype=np.float32)[:, None]
    up_values = np.arange(256, dtype=np.float32)[None, :]
    gate_real = (
        gate_values - np.float32(qparams["gate"]["zero_point"])
    ) * np.float32(qparams["gate"]["scale"])
    up_real = (
        up_values - np.float32(qparams["up"]["zero_point"])
    ) * np.float32(qparams["up"]["scale"])
    middle = gate_real / (
        np.float32(1.0) + np.exp(-gate_real, dtype=np.float32)
    ) * up_real
    encoded = np.rint(
        middle / np.float32(qparams["middle"]["scale"])
        + np.float32(qparams["middle"]["zero_point"])
    ).clip(0, 255).astype(np.uint8)
    return encoded.astype("<u2")


def write_qparams(
    path: Path, qparams: dict[str, dict[str, object]]
) -> None:
    record = struct.Struct("<32sfi2f")
    with path.open("wb") as stream:
        for name in sorted(qparams):
            encoded_name = name.encode("ascii")
            if len(encoded_name) >= 32:
                raise ValueError(name)
            value = qparams[name]
            stream.write(
                record.pack(
                    encoded_name,
                    float(value["scale"]),
                    int(value["zero_point"]),
                    float(value["minimum"]),
                    float(value["maximum"]),
                )
            )


def run_w4u8_layer(
    encoded_input: np.ndarray,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    qparams: dict[str, dict[str, object]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    retained_config: bytes | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes]:
    encoded: dict[str, torch.Tensor] = {
        "block_input": torch.from_numpy(
            np.ascontiguousarray(encoded_input)
        )
    }

    def boundary(name: str, value: torch.Tensor) -> torch.Tensor:
        encoded[name] = base.quantize_u8(value, qparams[name])
        return base.dequantize_u8(
            encoded[name], qparams[name]
        ).to(torch.float16)

    hidden = base.dequantize_u8(
        encoded["block_input"], qparams["block_input"]
    ).to(torch.float16)
    normalized = boundary(
        "input_norm", base.rms_norm(hidden, norms["input"])
    )
    q_projected = boundary(
        "q_projection", base.linear_half(normalized, weights["q"])
    )
    k_projected = boundary(
        "k_projection", base.linear_half(normalized, weights["k"])
    )
    v_flat = boundary(
        "v", base.linear_half(normalized, weights["v"])
    )
    q = base.rms_norm(
        q_projected.view(1, TOTAL_M, HEADS, HEAD_DIM),
        norms["q"],
    ).transpose(1, 2)
    k = base.rms_norm(
        k_projected.view(1, TOTAL_M, KV_HEADS, HEAD_DIM),
        norms["k"],
    ).transpose(1, 2)
    q, k = base.apply_rope(q, k, cos, sin)
    boundary(
        "q_rope",
        q.transpose(1, 2).contiguous().reshape(
            1, TOTAL_M, HIDDEN
        ),
    )
    boundary(
        "k_rope",
        k.transpose(1, 2).contiguous().reshape(
            1, TOTAL_M, KV_HEADS * HEAD_DIM
        ),
    )
    q_u8 = encoded["q_rope"].cpu().numpy().reshape(
        TOTAL_M, HEADS, HEAD_DIM
    )
    k_u8 = encoded["k_rope"].cpu().numpy().reshape(
        TOTAL_M, KV_HEADS, HEAD_DIM
    )
    v_u8 = encoded["v"].cpu().numpy().reshape(
        TOTAL_M, KV_HEADS, HEAD_DIM
    )
    config = retained_config or build_attention_config(v_u8, qparams)
    attention_output = integer_attention(q_u8, k_u8, v_u8, config)
    final = w4u8_output_from_attention(
        encoded["block_input"],
        attention_output,
        weights,
        norms["post"],
        qparams,
    )
    return final, k_u8, v_u8, config


def build_recipe_references(
    teacher: dict[int, dict[str, torch.Tensor]],
    original: dict[int, dict[str, torch.Tensor]],
    w4: dict[int, dict[str, torch.Tensor]],
    norms: dict[int, dict[str, torch.Tensor]],
    qparams: dict[int, dict[str, dict[str, object]]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    retained_layer14_config: bytes,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    initial = teacher[13]["block_input"]
    for recipe in ("f16f16", "w4f16"):
        hidden = initial
        layer_data: dict[int, dict[str, np.ndarray]] = {}
        for layer in LAYERS:
            weights = original[layer] if recipe == "f16f16" else w4[layer]
            hidden, boundaries = base.layer_forward_f16(
                hidden, weights, norms[layer], cos, sin
            )
            layer_data[layer] = {
                "k": np.ascontiguousarray(
                    f16(boundaries["k_rope"])
                    .reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
                    .transpose(1, 0, 2)
                ),
                "v": np.ascontiguousarray(
                    f16(boundaries["v"])
                    .reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
                    .transpose(1, 0, 2)
                ),
            }
        result[recipe] = {
            "input": f16(initial).reshape(TOTAL_M, HIDDEN),
            "output": f16(hidden).reshape(TOTAL_M, HIDDEN),
            "layers": layer_data,
        }

    encoded_hidden = base.quantize_u8(
        initial, qparams[13]["block_input"]
    ).cpu().numpy().reshape(TOTAL_M, HIDDEN)
    u8_layers: dict[int, dict[str, object]] = {}
    for layer in LAYERS:
        final, k, v, config = run_w4u8_layer(
            encoded_hidden.reshape(1, TOTAL_M, HIDDEN),
            w4[layer],
            norms[layer],
            qparams[layer],
            cos,
            sin,
            retained_layer14_config if layer == 14 else None,
        )
        u8_layers[layer] = {
            "k": np.ascontiguousarray(k.transpose(1, 0, 2)),
            "v": np.ascontiguousarray(v.transpose(1, 0, 2)),
            "config": config,
        }
        encoded_hidden = final
    result["w4u8"] = {
        "input": base.quantize_u8(
            initial, qparams[13]["block_input"]
        ).cpu().numpy().reshape(TOTAL_M, HIDDEN),
        "output": encoded_hidden.reshape(TOTAL_M, HIDDEN),
        "layers": u8_layers,
    }
    return result


def publish_recipe(
    recipe: str,
    archive: Path,
    staging_root: Path,
    references: dict[str, object],
    original: dict[int, dict[str, torch.Tensor]],
    w4_payloads: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    norms: dict[int, dict[str, torch.Tensor]],
    qparams: dict[int, dict[str, dict[str, object]]],
    cos: np.ndarray,
    sin: np.ndarray,
    token_ids: list[int],
) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f"{recipe}-", dir=staging_root)
    )
    destination = archive / recipe
    suffix = "u8" if recipe == "w4u8" else "f16"
    element_fill = (
        int(qparams[13]["block_input"]["zero_point"])
        if recipe == "w4u8" else 0
    )
    output_fill = (
        int(qparams[15]["block_output"]["zero_point"])
        if recipe == "w4u8" else 0
    )
    try:
        input_all = np.asarray(references["input"])
        output_all = np.asarray(references["output"])
        write(
            staging / (
                "reference_w4u8_block_input_u8.bin"
                if recipe == "w4u8" else "block_input_f16.bin"
            ),
            input_all[:PREFILL_M],
        )
        reference_name = {
            "f16f16": "reference_f16f16_block_output_f16.bin",
            "w4f16": "reference_w4f16_block_output_f16.bin",
            "w4u8":
                "reference_w4u8_integer_attention_block_output_u8.bin",
        }[recipe]
        write(staging / reference_name, output_all[:PREFILL_M])
        write(staging / "rope_cos_f16.bin", cos[:PREFILL_M])
        write(staging / "rope_sin_f16.bin", sin[:PREFILL_M])

        identity_cos = np.ones((PHYSICAL_M, HEAD_DIM), dtype="<f2")
        identity_sin = np.zeros((PHYSICAL_M, HEAD_DIM), dtype="<f2")
        for index in range(DECODE_STEPS):
            position = PREFILL_M + index
            input_row = np.full(
                (PHYSICAL_M, HIDDEN),
                element_fill,
                dtype=input_all.dtype,
            )
            output_row = np.full(
                (PHYSICAL_M, HIDDEN),
                output_fill,
                dtype=output_all.dtype,
            )
            input_row[0] = input_all[position]
            output_row[0] = output_all[position]
            step_cos = identity_cos.copy()
            step_sin = identity_sin.copy()
            step_cos[0] = cos[position]
            step_sin[0] = sin[position]
            write(
                staging /
                    f"replay_decode_input_{index:02d}_{suffix}.bin",
                input_row,
            )
            write(
                staging /
                    f"replay_decode_reference_{index:02d}_{suffix}.bin",
                output_row,
            )
            write(
                staging /
                    f"replay_decode_rope_cos_{index:02d}_f16.bin",
                step_cos,
            )
            write(
                staging /
                    f"replay_decode_rope_sin_{index:02d}_f16.bin",
                step_sin,
            )

        for layer in LAYERS:
            layer_root = staging / f"layer{layer}"
            layer_root.mkdir()
            write_qparams(layer_root / "qparams_u8.bin", qparams[layer])
            for name, file_name in (
                ("input", "input_norm_weight_f16.bin"),
                ("post", "post_norm_weight_f16.bin"),
                ("q", "q_norm_weight_f16.bin"),
                ("k", "k_norm_weight_f16.bin"),
            ):
                write(layer_root / file_name, f16(norms[layer][name]))
            for name in PROJECTION_SHAPES:
                if recipe == "f16f16":
                    write(
                        layer_root / f"{name}_weight_f16_hmx.bin",
                        base.pack_fp16_hmx_weight(original[layer][name]),
                    )
                else:
                    packed, scales = w4_payloads[layer][name]
                    write(
                        layer_root / f"{name}_weight_w4_hmx.bin",
                        packed,
                    )
                    write(
                        layer_root / f"{name}_weight_w4_scale_f32.bin",
                        scales,
                    )
            layer_reference = references["layers"][layer]
            cache_dtype = (
                np.uint8 if recipe == "w4u8" else np.dtype("<f2")
            )
            empty_cache = np.zeros(
                (KV_HEADS, TOTAL_M, HEAD_DIM), dtype=cache_dtype
            )
            write(layer_root / f"kv_cache_k_{suffix}.bin", empty_cache)
            write(layer_root / f"kv_cache_v_{suffix}.bin", empty_cache)
            write(
                layer_root /
                    f"reference_kv_cache_k_{suffix}.bin",
                np.asarray(layer_reference["k"]),
            )
            write(
                layer_root /
                    f"reference_kv_cache_v_{suffix}.bin",
                np.asarray(layer_reference["v"]),
            )
            if recipe == "w4u8":
                (layer_root / "attention_config_all_groups.bin").write_bytes(
                    layer_reference["config"]
                )
                write(
                    layer_root / "silu_up_lut_u16.bin",
                    build_silu_lut(qparams[layer]),
                )

        files = {
            str(path.relative_to(staging)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "experiment": "EXP-0149",
            "recipe": recipe,
            "execution_unit":
                "qwen3_real_layers13_14_15_one_dsp_invocation",
            "token_ids": token_ids,
            "contract": {
                "declared_transformer_layers": DECLARED_LAYERS,
                "active_layers": list(LAYERS),
                "prefill_positions": [0, PREFILL_M - 1],
                "decode_positions": [PREFILL_M, TOTAL_M - 1],
                "cache_capacity_per_layer": TOTAL_M,
                "runtime_kv_imported": False,
                "host_hidden_handoff": False,
                "intermediate_hidden_residency": "VTCM",
                "one_rpc_per_three_layer_step": True,
            },
            "handoff_qparams": {
                "layer13_output_equals_layer14_input":
                    qparams[13]["block_output"] ==
                    qparams[14]["block_input"],
                "layer14_output_equals_layer15_input":
                    qparams[14]["block_output"] ==
                    qparams[15]["block_input"],
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copytree(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    exp0148_root = args.exp0148_root.resolve()
    output = args.output.resolve()
    staging_root = args.staging_root.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing EXP-0149 archive: {output}"
        )
    retained_w4u8 = exp0148_root / "w4u8_formal"
    if not (retained_w4u8 / "manifest.json").is_file():
        raise FileNotFoundError(retained_w4u8 / "manifest.json")
    retained_qparams = load_qparams(retained_w4u8)
    retained_config = (
        retained_w4u8 / "attention_config_all_groups.bin"
    ).read_bytes()
    if len(retained_config) != KV_HEADS * CONFIG.size:
        raise ValueError("retained layer-14 Attention config is invalid")

    staging_root.mkdir(parents=True, exist_ok=True)
    archive = Path(
        tempfile.mkdtemp(prefix="archive-", dir=staging_root)
    )
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"
    try:
        base.M = TOTAL_M
        token_ids, teacher, original, norms = load_teacher_and_weights(
            model
        )
        qparams = derive_qparams(teacher, retained_qparams)
        w4_payloads: dict[
            int, dict[str, tuple[np.ndarray, np.ndarray]]
        ] = {}
        w4_weights: dict[int, dict[str, torch.Tensor]] = {}
        for layer in LAYERS:
            w4_payloads[layer] = {}
            w4_weights[layer] = {}
            for name, weight in original[layer].items():
                packed, scales, dequant = base.quantize_w4_per_output(
                    weight
                )
                w4_payloads[layer][name] = (packed, scales)
                w4_weights[layer][name] = dequant

        # Retained layer-14 carrier identity is a hard provenance check.
        for name in PROJECTION_SHAPES:
            retained_carrier = retained_w4u8 / f"{name}_weight_w4_hmx.bin"
            retained_scale = (
                retained_w4u8 / f"{name}_weight_w4_scale_f32.bin"
            )
            temporary_carrier = archive / f".{name}.carrier"
            temporary_scale = archive / f".{name}.scale"
            write(temporary_carrier, w4_payloads[14][name][0])
            write(temporary_scale, w4_payloads[14][name][1])
            if (
                sha256(temporary_carrier) != sha256(retained_carrier)
                or sha256(temporary_scale) != sha256(retained_scale)
            ):
                raise ValueError(
                    f"layer-14 retained W4 identity failed: {name}"
                )
            temporary_carrier.unlink()
            temporary_scale.unlink()

        cos_t, sin_t = base.rope_tables(torch.float16)
        cos = f16(cos_t)
        sin = f16(sin_t)
        references = build_recipe_references(
            teacher,
            original,
            w4_weights,
            norms,
            qparams,
            cos_t,
            sin_t,
            retained_config,
        )
        for recipe in ("f16f16", "w4f16", "w4u8"):
            publish_recipe(
                recipe,
                archive,
                staging_root,
                references[recipe],
                original,
                w4_payloads,
                norms,
                qparams,
                cos,
                sin,
                token_ids,
            )

        trace = archive / "teacher_trace_layers13_15_p64_d8"
        trace.mkdir()
        write(trace / "token_ids_i32.bin", np.asarray(token_ids, dtype="<i4"))
        for layer in LAYERS:
            write(
                trace / f"layer{layer}_input_bf16_as_f16.bin",
                f16(teacher[layer]["block_input"]),
            )
            write(
                trace / f"layer{layer}_output_bf16_as_f16.bin",
                f16(teacher[layer]["block_output"]),
            )
        (trace / "manifest.json").write_text(
            json.dumps(
                {
                    "experiment": "EXP-0149",
                    "authority":
                        "original_Qwen3_BF16_layers_0_through_15",
                    "active_layers": list(LAYERS),
                    "positions": TOTAL_M,
                    "token_ids": token_ids,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        if publish.exists():
            raise FileExistsError(publish)
        shutil.copytree(archive, publish)
        os.rename(publish, output)
        print(
            json.dumps(
                {
                    "experiment": "EXP-0149",
                    "output": str(output),
                    "recipes": ["f16f16", "w4f16", "w4u8"],
                    "layers": list(LAYERS),
                },
                indent=2,
            )
        )
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(archive, ignore_errors=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    main()
