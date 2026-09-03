#!/usr/bin/env python3
"""Publish the EXP-0164 W4F16 deterministic-generation package.

The accepted EXP-0158 transformer package is hard-linked into a new archive.
Only the model boundaries are added: real prompt token IDs, FP16 embedding,
final RMSNorm, per-output-channel W4 LM head, and an independent greedy token
sequence.  Persistent FP16 HMX-native caches are resized for the 16-token
closed-loop generation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


EXPERIMENT = "EXP-0164"
VOCAB = 151_936
HIDDEN = 2_048
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
PREFILL = 64
GENERATED_TOKENS = 16
# Only the first 15 generated tokens are fed back; token 16 is returned by the
# final pass.  A little spare capacity makes the ABI explicit and reusable.
CACHE_CAPACITY = 80
HMX_ROWS = 32
HMX_COLS = 32
W4_TILE_BYTES = 512
ROPE_THETA = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0158/w4f16"
        ),
    )
    parser.add_argument(
        "--semantic-reference",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/results/qwen3-block-htp/exp0164/"
            "semantic_gate/teacher_w4f16_greedy16.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/"
            "w4f16_greedy16"
        ),
    )
    parser.add_argument("--row-chunk", type=int, default=1024)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def model_tensor_location(model: Path, name: str) -> Path:
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    return model / index["weight_map"][name]


def copy_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def pack_w4_chunk(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logical = weight.detach().float().cpu()
    maximum = logical.abs().amax(dim=1)
    scale = torch.where(
        maximum > 0.0, maximum / 7.0, torch.ones_like(maximum)
    )
    quantized = torch.round(logical / scale[:, None]).clamp(-7, 7)
    quantized = quantized.to(torch.int8)
    n, k = quantized.shape
    if n % HMX_COLS or k % HMX_ROWS:
        raise ValueError(f"HMX W4 shape must be /32, got {(n, k)}")
    physical = quantized.numpy().reshape(n // 32, 32, k // 32, 32)
    physical = physical.transpose(0, 2, 3, 1)
    physical = physical.reshape(n // 32, k // 32, 8, 4, 32)
    physical = np.ascontiguousarray(physical.transpose(0, 1, 2, 4, 3))
    flat = physical.reshape(n // 32, k // 32, 1024).astype(np.int16)
    nibble = (flat & 0xF).astype(np.uint8)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    return np.ascontiguousarray(packed), scale.numpy().astype("<f4")


def export_embedding_and_head(
    model: Path, root: Path, row_chunk: int
) -> dict[str, dict[str, object]]:
    embedding_name = "model.embed_tokens.weight"
    head_name = "lm_head.weight"
    embedding_shard = model_tensor_location(model, embedding_name)
    head_shard = model_tensor_location(model, head_name)
    embedding_path = root / "generation_embedding_weight_f16.bin"
    head_path = root / "generation_lm_head_weight_w4_hmx.bin"
    scales_path = root / "generation_lm_head_weight_w4_scale_f32.bin"

    with safe_open(embedding_shard, framework="pt", device="cpu") as source:
        embedding = source.get_tensor(embedding_name)
        if tuple(embedding.shape) != (VOCAB, HIDDEN):
            raise ValueError(f"unexpected embedding shape {tuple(embedding.shape)}")
        fp16 = embedding.to(torch.float16).cpu().numpy().astype("<f2", copy=False)
        atomic_write_bytes(embedding_path, np.ascontiguousarray(fp16).tobytes())
        del embedding, fp16

    if row_chunk <= 0 or row_chunk % HMX_COLS:
        raise ValueError("--row-chunk must be a positive multiple of 32")
    temporary_head = head_path.with_suffix(head_path.suffix + ".tmp")
    temporary_scales = scales_path.with_suffix(scales_path.suffix + ".tmp")
    with safe_open(head_shard, framework="pt", device="cpu") as source:
        head = source.get_slice(head_name)
        if tuple(head.get_shape()) != (VOCAB, HIDDEN):
            raise ValueError(f"unexpected lm_head shape {tuple(head.get_shape())}")
        with temporary_head.open("wb") as carrier, temporary_scales.open("wb") as scale_file:
            for first in range(0, VOCAB, row_chunk):
                last = min(first + row_chunk, VOCAB)
                # Vocabulary size and the selected chunk both preserve N32.
                packed, scales = pack_w4_chunk(head[first:last, :])
                carrier.write(packed.tobytes(order="C"))
                scale_file.write(scales.tobytes(order="C"))
    os.replace(temporary_head, head_path)
    os.replace(temporary_scales, scales_path)
    expected_head_bytes = VOCAB * HIDDEN // 2
    if head_path.stat().st_size != expected_head_bytes:
        raise ValueError(
            f"LM-head carrier bytes {head_path.stat().st_size} != {expected_head_bytes}"
        )
    if scales_path.stat().st_size != VOCAB * 4:
        raise ValueError("LM-head scale file has the wrong size")
    return {
        str(embedding_path.relative_to(root)): file_record(embedding_path),
        str(head_path.relative_to(root)): file_record(head_path),
        str(scales_path.relative_to(root)): file_record(scales_path),
    }


def export_final_norm(model: Path, root: Path) -> dict[str, object]:
    name = "model.norm.weight"
    shard = model_tensor_location(model, name)
    path = root / "generation_final_norm_weight_f16.bin"
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor(name)
    if tuple(weight.shape) != (HIDDEN,):
        raise ValueError(f"unexpected final norm shape {tuple(weight.shape)}")
    payload = weight.to(torch.float16).cpu().numpy().astype("<f2", copy=False)
    atomic_write_bytes(path, np.ascontiguousarray(payload).tobytes())
    return file_record(path)


def rope_rows(first_position: int, row_count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
    inverse = 1.0 / (ROPE_THETA ** (indices / HEAD_DIM))
    positions = torch.arange(
        first_position, first_position + row_count, dtype=torch.float32
    )
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    return (
        embedding.cos().to(torch.float16).numpy().astype("<f2", copy=False),
        embedding.sin().to(torch.float16).numpy().astype("<f2", copy=False),
    )


def export_rope(root: Path) -> list[Path]:
    generated: list[Path] = []
    cosine, sine = rope_rows(0, PREFILL)
    for kind, values in (("cos", cosine), ("sin", sine)):
        path = root / f"rope_{kind}_f16.bin"
        atomic_write_bytes(path, np.ascontiguousarray(values).tobytes())
        generated.append(path)
    identity_cos = np.ones((PREFILL, HEAD_DIM), dtype="<f2")
    identity_sin = np.zeros((PREFILL, HEAD_DIM), dtype="<f2")
    for decode_index in range(GENERATED_TOKENS - 1):
        cosine, sine = rope_rows(PREFILL + decode_index, 1)
        for kind, identity, row in (
            ("cos", identity_cos, cosine),
            ("sin", identity_sin, sine),
        ):
            values = identity.copy()
            values[0] = row[0]
            path = root / f"generation_decode_rope_{kind}_{decode_index:02d}_f16.bin"
            atomic_write_bytes(path, values.tobytes(order="C"))
            generated.append(path)
    return generated


def export_zero_caches(root: Path) -> list[Path]:
    generated: list[Path] = []
    bytes_per_cache = (
        KV_HEADS * (PREFILL + (CACHE_CAPACITY - PREFILL)) * HEAD_DIM * 2
    )
    # F16 HMX carrier and FP16 delta journal have the same aggregate byte
    # count, although the first M64 rows use the tiled physical layout.
    zeros = bytes(bytes_per_cache)
    for layer in range(LAYERS):
        layer_root = root / f"layer{layer}"
        for kind in ("k", "v"):
            for name in (
                f"kv_cache_{kind}_hmx_f16.bin",
                f"reference_kv_cache_{kind}_hmx_f16_step00.bin",
            ):
                path = layer_root / name
                atomic_write_bytes(path, zeros)
                generated.append(path)
    return generated


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    source = args.source.resolve()
    semantic_path = args.semantic_reference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing archive: {output}")
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    prompt_ids = semantic["prompt_token_ids"]
    expected_ids = semantic["w4f16"]["token_ids"]
    if len(prompt_ids) != PREFILL or len(expected_ids) != GENERATED_TOKENS:
        raise ValueError("semantic reference does not match the EXP-0164 shape")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.publishing-", dir=output.parent)
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, copy_function=copy_link)
        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})

        token_path = staging / "generation_prompt_token_ids_u32.bin"
        expected_path = staging / "generation_expected_token_ids_u32.bin"
        atomic_write_bytes(token_path, np.asarray(prompt_ids, dtype="<u4").tobytes())
        atomic_write_bytes(expected_path, np.asarray(expected_ids, dtype="<u4").tobytes())
        files[token_path.name] = file_record(token_path)
        files[expected_path.name] = file_record(expected_path)
        files["generation_final_norm_weight_f16.bin"] = export_final_norm(
            model, staging
        )
        files.update(export_embedding_and_head(model, staging, args.row_chunk))
        for path in export_rope(staging) + export_zero_caches(staging):
            files[str(path.relative_to(staging))] = file_record(path)

        manifest["experiment"] = EXPERIMENT
        manifest["execution_unit"] = (
            "real_token_ids_embedding_layers0_27_final_norm_streaming_"
            "w4f16_lm_head_greedy_feedback"
        )
        manifest["generation"] = {
            "prompt_tokens": PREFILL,
            "generated_tokens": GENERATED_TOKENS,
            "feedback_decode_calls": GENERATED_TOKENS - 1,
            "cache_capacity": CACHE_CAPACITY,
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "decode": "deterministic_greedy_strict_greater_lower_id_tie",
            "timed_full_logits_ddr": False,
            "source_model_index_sha256": sha256_file(
                model / "model.safetensors.index.json"
            ),
            "source_tokenizer_sha256": semantic["tokenizer_sha256"],
            "semantic_reference_sha256": sha256_file(semantic_path),
            "independent_expected_token_ids": expected_ids,
            "independent_expected_text": semantic["w4f16"]["text"],
            "weight_quantization": (
                "signed_symmetric_per_output_channel_w4_qmin_-7_qmax_7"
            ),
        }
        atomic_write_bytes(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"PUBLISHED={output}")
    print(f"PROMPT_TOKENS={PREFILL}")
    print(f"EXPECTED_TOKENS={expected_ids}")


if __name__ == "__main__":
    main()
