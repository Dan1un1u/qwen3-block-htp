#!/usr/bin/env python3
"""Export genuine layer-14 prefill -> continuous decode replay packages.

The full-model BF16 teacher executes tokens 0..71 through layers 0..14.  Its
layer-14 input is then replayed through the three project recipes.  Runtime K/V
is never imported: the package contains an empty capacity-72 cache and an
independent final cache reference produced from the same recipe math.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
from prepare_exp0042_attention import CONFIG, HMX_CENTER, centered_hmx_requant  # noqa: E402
from prepare_exp0042_block import unpack_w4_weight  # noqa: E402
from prepare_exp0147_decode import (  # noqa: E402
    DIVISION_NAMES,
    log2_softmax_decode,
    recenter_v,
)


PREFILL_M = 64
DECODE_STEPS = 8
TOTAL_M = PREFILL_M + DECODE_STEPS
PHYSICAL_M = 64
LAYER = 14
DECLARED_LAYERS = 28
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS

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
        "--model", type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--base-root", type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0147"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0148"),
    )
    parser.add_argument(
        "--staging-root", type=Path,
        default=Path("/home/daniuniu/.cache/qwen3-block-htp-exp0148"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: np.ndarray) -> None:
    np.ascontiguousarray(value).tofile(path)


def f16(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype("<f2", copy=False)


def load_qparams(package: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if "u8_qparams" in manifest:
        return manifest["u8_qparams"]
    if "reference_source" not in manifest:
        if "source_package" not in manifest:
            raise KeyError(f"no qparam provenance in {package / 'manifest.json'}")
        return load_qparams(Path(manifest["source_package"]))
    reference = Path(manifest["reference_source"])
    reference_manifest = json.loads(
        (reference / "manifest.json").read_text(encoding="utf-8")
    )
    return reference_manifest["u8_qparams"]


def load_teacher(
    model: Path,
) -> tuple[list[int], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tokenizer = Tokenizer.from_file(str(model / "qwen3-tokenizer.json"))
    token_ids = tokenizer.encode(base.PROMPT * 2, add_special_tokens=True).ids[:TOTAL_M]
    if len(token_ids) != TOTAL_M:
        raise ValueError(f"prompt produced {len(token_ids)} tokens, expected {TOTAL_M}")
    input_ids = torch.tensor(token_ids, dtype=torch.long)[None, :]
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard_name = index["weight_map"]["model.embed_tokens.weight"]
    required = ["model.embed_tokens.weight"]
    for layer_index in range(LAYER + 1):
        prefix = f"model.layers.{layer_index}"
        required.extend(
            f"{prefix}.{suffix}" for suffix in (
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
    wrong = [name for name in required if index["weight_map"][name] != shard_name]
    if wrong:
        raise ValueError(f"teacher weights span shards: {wrong[:3]}")

    with safe_open(model / shard_name, framework="pt", device="cpu") as source:
        get_weight = source.get_tensor
        hidden = F.embedding(input_ids, get_weight("model.embed_tokens.weight"))
        cos_bf16, sin_bf16 = base.rope_tables(torch.bfloat16)
        captured: dict[str, torch.Tensor] = {}
        for layer_index in range(LAYER + 1):
            hidden, boundaries = base.layer_forward_bf16(
                hidden, get_weight, layer_index, cos_bf16, sin_bf16,
                layer_index == LAYER,
            )
            if boundaries:
                captured = {key: value.detach().cpu() for key, value in boundaries.items()}
        original = {
            name: get_weight(key).to(torch.float16).cpu()
            for name, key in base.PROJECTIONS.items()
        }
        norms = {
            "input": get_weight(f"model.layers.{LAYER}.input_layernorm.weight").to(torch.float16).cpu(),
            "post": get_weight(f"model.layers.{LAYER}.post_attention_layernorm.weight").to(torch.float16).cpu(),
            "q": get_weight(f"model.layers.{LAYER}.self_attn.q_norm.weight").to(torch.float16).cpu(),
            "k": get_weight(f"model.layers.{LAYER}.self_attn.k_norm.weight").to(torch.float16).cpu(),
        }
    return token_ids, captured, original, norms


def integer_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, config_bytes: bytes,
) -> np.ndarray:
    """Run the accepted causal integer Attention contract for all 72 rows."""
    rows = int(q.shape[0])
    if k.shape[0] != rows or v.shape[0] != rows:
        raise ValueError(
            f"Q/K/V row mismatch: q={q.shape} k={k.shape} v={v.shape}"
        )
    output = np.empty((rows, HEADS, HEAD_DIM), dtype=np.uint8)
    if len(config_bytes) != KV_HEADS * CONFIG.size:
        raise ValueError("unexpected integer Attention config size")
    for position in range(rows):
        valid = position + 1
        for group in range(KV_HEADS):
            fields = CONFIG.unpack_from(config_bytes, group * CONFIG.size)
            (
                abi, config_group, fraction_bits, division_mode,
                q_zero_point, k_zero_point, v_zero_point,
                _probability_zero_point, output_zero_point,
                v_numerator, v_denominator, score_shift, score_multiplier,
                av_shift, av_multiplier,
            ) = fields
            if abi != 1 or config_group != group or division_mode not in DIVISION_NAMES:
                raise ValueError(f"invalid Attention config for group {group}")
            first_head = group * Q_HEADS_PER_GROUP
            q_centered = (
                q[position, first_head:first_head + Q_HEADS_PER_GROUP].astype(np.int32)
                - q_zero_point
            )
            k_centered = np.clip(
                k[:valid, group].astype(np.int32) - k_zero_point, -128, 127
            ).astype(np.int8)
            score_accumulator = np.matmul(
                q_centered, k_centered.astype(np.int32).T
            )
            score, _ = centered_hmx_requant(
                score_accumulator, score_multiplier, score_shift, HMX_CENTER
            )
            probability = log2_softmax_decode(
                score, fraction_bits, DIVISION_NAMES[division_mode]
            )
            v_centered = v[:valid, group].astype(np.int32) - v_zero_point
            v_signed = recenter_v(v_centered, v_numerator, v_denominator)
            av_accumulator = np.matmul(
                probability.astype(np.int32), v_signed.astype(np.int32)
            )
            group_output, _ = centered_hmx_requant(
                av_accumulator, av_multiplier, av_shift, output_zero_point
            )
            output[position, first_head:first_head + Q_HEADS_PER_GROUP] = group_output
    return output


def w4u8_output_from_attention(
    encoded_input: torch.Tensor,
    attention_output: np.ndarray,
    weights: dict[str, torch.Tensor],
    post_weight: torch.Tensor,
    qparams: dict[str, dict[str, object]],
) -> np.ndarray:
    rows = int(encoded_input.shape[1])
    if attention_output.shape[0] != rows:
        raise ValueError(
            "Attention/output row mismatch: "
            f"input={rows} attention={attention_output.shape[0]}"
        )
    hidden = base.dequantize_u8(encoded_input, qparams["block_input"]).to(torch.float16)
    attention_encoded = torch.from_numpy(attention_output.reshape(1, rows, HIDDEN))
    attention_half = base.dequantize_u8(
        attention_encoded, qparams["attention_concat"]
    ).to(torch.float16)

    def boundary(name: str, value: torch.Tensor) -> torch.Tensor:
        return base.dequantize_u8(
            base.quantize_u8(value, qparams[name]), qparams[name]
        ).to(torch.float16)

    projected = boundary(
        "attention_projection", base.linear_half(attention_half, weights["o"])
    )
    residual = boundary("post_attention_residual", hidden + projected)
    normalized = boundary(
        "post_attention_norm", base.rms_norm(residual, post_weight)
    )
    gate = boundary("gate", base.linear_half(normalized, weights["gate"]))
    up = boundary("up", base.linear_half(normalized, weights["up"]))
    middle = boundary("middle", F.silu(gate.float()) * up.float())
    down = boundary("down", base.linear_half(middle, weights["down"]))
    return base.quantize_u8(
        residual + down, qparams["block_output"]
    ).cpu().numpy().reshape(rows, HIDDEN)


def padded_rows(value: np.ndarray, fill: int | float = 0) -> np.ndarray:
    shape = (PHYSICAL_M,) + value.shape[1:]
    result = np.full(shape, fill, dtype=value.dtype)
    result[0] = value[0]
    return result


def publish_recipe(
    recipe: str, base_package: Path, output_root: Path,
    staging_root: Path, teacher_input: torch.Tensor,
    recipe_output: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
    cos: np.ndarray, sin: np.ndarray, encoded_input: np.ndarray | None,
    diagnostics: dict[str, object],
) -> None:
    destination = output_root / recipe
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=f"{recipe}-", dir=staging_root))
    publish = output_root / f".{recipe}.publishing-{os.getpid()}"
    try:
        shutil.copytree(base_package, staging, dirs_exist_ok=True)
        is_u8 = recipe == "w4u8"
        element_fill = diagnostics["input_zero_point"] if is_u8 else 0
        output_fill = diagnostics["output_zero_point"] if is_u8 else 0
        input_all = encoded_input if is_u8 else f16(teacher_input).reshape(TOTAL_M, HIDDEN)
        write(
            staging / ("reference_w4u8_block_input_u8.bin" if is_u8 else "block_input_f16.bin"),
            input_all[:PREFILL_M],
        )
        reference_name = {
            "f16f16": "reference_f16f16_block_output_f16.bin",
            "w4f16": "reference_w4f16_block_output_f16.bin",
            "w4u8": "reference_w4u8_integer_attention_block_output_u8.bin",
        }[recipe]
        write(staging / reference_name, recipe_output[:PREFILL_M])
        write(staging / "rope_cos_f16.bin", cos[:PREFILL_M])
        write(staging / "rope_sin_f16.bin", sin[:PREFILL_M])

        suffix = "u8" if is_u8 else "f16"
        cache_dtype = np.uint8 if is_u8 else np.dtype("<f2")
        empty_cache = np.zeros((KV_HEADS, TOTAL_M, HEAD_DIM), dtype=cache_dtype)
        write(staging / f"kv_cache_k_{suffix}.bin", empty_cache)
        write(staging / f"kv_cache_v_{suffix}.bin", empty_cache)
        write(staging / f"reference_kv_cache_k_{suffix}.bin", k_cache)
        write(staging / f"reference_kv_cache_v_{suffix}.bin", v_cache)

        identity_cos = np.ones((PHYSICAL_M, HEAD_DIM), dtype="<f2")
        identity_sin = np.zeros((PHYSICAL_M, HEAD_DIM), dtype="<f2")
        for index in range(DECODE_STEPS):
            position = PREFILL_M + index
            input_row = np.full(
                (PHYSICAL_M, HIDDEN), element_fill, dtype=input_all.dtype
            )
            output_row = np.full(
                (PHYSICAL_M, HIDDEN), output_fill, dtype=recipe_output.dtype
            )
            input_row[0] = input_all[position]
            output_row[0] = recipe_output[position]
            step_cos = identity_cos.copy()
            step_sin = identity_sin.copy()
            step_cos[0] = cos[position]
            step_sin[0] = sin[position]
            write(staging / f"replay_decode_input_{index:02d}_{suffix}.bin", input_row)
            write(staging / f"replay_decode_reference_{index:02d}_{suffix}.bin", output_row)
            write(staging / f"replay_decode_rope_cos_{index:02d}_f16.bin", step_cos)
            write(staging / f"replay_decode_rope_sin_{index:02d}_f16.bin", step_sin)

        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "experiment": "EXP-0148",
            "recipe": recipe,
            "execution_unit": "qwen3_layer14_real_replay_prefill_continuous_decode",
            "source_package": str(base_package),
            "source_manifest_sha256": sha256(base_package / "manifest.json"),
            "teacher_trace": "../teacher_trace_layer14_p64_d8",
            "contract": {
                "declared_transformer_layers": DECLARED_LAYERS,
                "active_layer": LAYER,
                "prefill_positions": [0, PREFILL_M - 1],
                "decode_positions": [PREFILL_M, TOTAL_M - 1],
                "cache_capacity": TOTAL_M,
                "cache_format": "head_major_row_v1",
                "runtime_kv_imported": False,
                "one_prepared_session": True,
                "one_rpc_per_step": True,
            },
            "diagnostics": diagnostics,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if publish.exists():
            raise FileExistsError(publish)
        shutil.copytree(staging, publish)
        os.rename(publish, destination)
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    base_root = args.base_root.resolve()
    output = args.output.resolve()
    staging_root = args.staging_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing EXP-0148 archive: {output}")
    base_packages = {
        recipe: base_root / f"prefill_m64_{recipe}"
        for recipe in ("f16f16", "w4f16", "w4u8")
    }
    for package in base_packages.values():
        if not (package / "manifest.json").is_file():
            raise FileNotFoundError(package / "manifest.json")
    staging_root.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix="archive-", dir=staging_root))
    publish = output.parent / f".{output.name}.publishing-{os.getpid()}"

    try:
        base.M = TOTAL_M
        token_ids, teacher, original_weights, norm_weights = load_teacher(model)
        cos_t, sin_t = base.rope_tables(torch.float16)
        cos = f16(cos_t)
        sin = f16(sin_t)
        teacher_input = teacher["block_input"]

        f16_output_t, f16_boundaries = base.layer_forward_f16(
            teacher_input, original_weights, norm_weights, cos_t, sin_t
        )
        w4_weights = {
            name: unpack_w4_weight(base_packages["w4f16"], name, *shape)
            for name, shape in PROJECTION_SHAPES.items()
        }
        w4f16_output_t, w4f16_boundaries = base.layer_forward_f16(
            teacher_input, w4_weights, norm_weights, cos_t, sin_t
        )

        qparams = load_qparams(base_packages["w4u8"])
        _standard_output, _boundaries, encoded = base.layer_forward_w4u8(
            teacher_input, w4_weights, norm_weights, qparams, cos_t, sin_t
        )
        q = encoded["q_rope"].cpu().numpy().reshape(TOTAL_M, HEADS, HEAD_DIM)
        k = encoded["k_rope"].cpu().numpy().reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
        v = encoded["v"].cpu().numpy().reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
        config_bytes = (base_packages["w4u8"] / "attention_config_all_groups.bin").read_bytes()
        integer_av = integer_attention(q, k, v, config_bytes)
        w4u8_output = w4u8_output_from_attention(
            encoded["block_input"], integer_av, w4_weights,
            norm_weights["post"], qparams,
        )

        # The first 64 positions reproduce the already accepted M64 package.
        # This catches accidental changes in tokenization, qparams, W4 packing,
        # or integer Attention math before publishing the replay archive.
        accepted_u8 = np.fromfile(
            base_packages["w4u8"] / "reference_w4u8_integer_attention_block_output_u8.bin",
            dtype=np.uint8,
        ).reshape(PREFILL_M, HIDDEN)
        accepted_k = np.fromfile(
            base_packages["w4u8"] / "reference_kv_cache_k_u8.bin", dtype=np.uint8
        ).reshape(KV_HEADS, PREFILL_M, HEAD_DIM)
        accepted_v = np.fromfile(
            base_packages["w4u8"] / "reference_kv_cache_v_u8.bin", dtype=np.uint8
        ).reshape(KV_HEADS, PREFILL_M, HEAD_DIM)
        u8_k_cache = np.ascontiguousarray(k.transpose(1, 0, 2))
        u8_v_cache = np.ascontiguousarray(v.transpose(1, 0, 2))
        legacy_mismatches = {
            "output": int(np.count_nonzero(w4u8_output[:PREFILL_M] != accepted_u8)),
            "k_cache": int(np.count_nonzero(u8_k_cache[:, :PREFILL_M] != accepted_k)),
            "v_cache": int(np.count_nonzero(u8_v_cache[:, :PREFILL_M] != accepted_v)),
        }
        if legacy_mismatches["output"] != 0:
            raise ValueError(f"M64 W4U8 output identity failed: {legacy_mismatches}")

        trace_dir = archive / "teacher_trace_layer14_p64_d8"
        trace_dir.mkdir()
        write(trace_dir / "token_ids_i32.bin", np.asarray(token_ids, dtype="<i4"))
        write(trace_dir / "layer14_input_bf16_as_f16.bin", f16(teacher_input))
        write(trace_dir / "layer14_output_bf16_as_f16.bin", f16(teacher["block_output"]))
        write(trace_dir / "rope_cos_f16.bin", cos)
        write(trace_dir / "rope_sin_f16.bin", sin)
        trace_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(trace_dir.iterdir())
        }
        (trace_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "experiment": "EXP-0148",
                    "authority": "original_Qwen3_BF16_layers_0_through_14",
                    "source_model": str(model),
                    "active_layer": LAYER,
                    "positions": TOTAL_M,
                    "causal_equivalence": "full causal row p equals incremental replay at position p",
                    "token_ids": token_ids,
                    "files": trace_files,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        publish_recipe(
            "f16f16", base_packages["f16f16"], archive, staging_root,
            teacher_input, f16(f16_output_t).reshape(TOTAL_M, HIDDEN),
            np.ascontiguousarray(
                f16_boundaries["k_rope"]
                .reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
                .cpu().numpy().astype("<f2").transpose(1, 0, 2)
            ),
            np.ascontiguousarray(
                f16_boundaries["v"]
                .reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
                .cpu().numpy().astype("<f2").transpose(1, 0, 2)
            ),
            cos, sin, None,
            {
                "teacher_input": "real_layer14_hidden",
                "input_zero_point": 0,
                "output_zero_point": 0,
            },
        )
        publish_recipe(
            "w4f16", base_packages["w4f16"], archive, staging_root,
            teacher_input, f16(w4f16_output_t).reshape(TOTAL_M, HIDDEN),
            np.ascontiguousarray(
                w4f16_boundaries["k_rope"]
                .reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
                .cpu().numpy().astype("<f2").transpose(1, 0, 2)
            ),
            np.ascontiguousarray(
                w4f16_boundaries["v"]
                .reshape(TOTAL_M, KV_HEADS, HEAD_DIM)
                .cpu().numpy().astype("<f2").transpose(1, 0, 2)
            ),
            cos, sin, None,
            {
                "teacher_input": "real_layer14_hidden",
                "input_zero_point": 0,
                "output_zero_point": 0,
            },
        )
        publish_recipe(
            "w4u8", base_packages["w4u8"], archive, staging_root,
            teacher_input, w4u8_output, u8_k_cache, u8_v_cache,
            cos, sin,
            encoded["block_input"].cpu().numpy().reshape(TOTAL_M, HIDDEN),
            {
                "teacher_input":
                    "real_layer14_hidden_quantized_with_existing_qparams",
                "input_zero_point":
                    int(qparams["block_input"]["zero_point"]),
                "output_zero_point":
                    int(qparams["block_output"]["zero_point"]),
                "legacy_m64_identity_mismatches": legacy_mismatches,
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if publish.exists():
            raise FileExistsError(publish)
        shutil.copytree(archive, publish)
        os.rename(publish, output)
        print(json.dumps(
            {"output": str(output), "legacy_m64_identity": legacy_mismatches},
            indent=2,
        ))
    finally:
        if publish.exists():
            shutil.rmtree(publish)
        shutil.rmtree(archive, ignore_errors=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    main()
