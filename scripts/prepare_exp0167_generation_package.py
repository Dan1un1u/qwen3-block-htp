#!/usr/bin/env python3
"""Publish the EXP-0167 cache-native W4U8 generation package.

The selected EXP-0163 transformer ABI is retained.  This script adds a real
U8 embedding table, the original FP16 final RMSNorm gamma, a per-channel W4
LM head, precomputed integer-HMX bias/requant carriers, and the same 64-token
prompt used by the W4F16 boundary experiments.  Teacher token IDs are copied
only as a diagnostic; EXP-0167 does not gate model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from reference_w4u8_hmx import load_qparams_bin, round_half_away_from_zero


EXPERIMENT = "EXP-0167"
VOCAB = 151_936
HIDDEN = 2_048
HMX_CHANNELS = 32
HMX_BIAS_BYTES = 256
QPARAM_RECORD = struct.Struct("<32sfi2f")
FINAL_NORM_QPARAM = {
    "scale": 0.125,
    "zero_point": 128,
    "minimum": -16.0,
    "maximum": 15.875,
}
LM_HEAD_QPARAM = {
    "scale": 0.5,
    "zero_point": 128,
    "minimum": -64.0,
    "maximum": 63.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument(
        "--source", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0167/"
            "base_segmented_capacity80"
        ),
    )
    parser.add_argument(
        "--w4f16-boundary", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/"
            "w4f16_greedy16"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0167/"
            "w4u8_greedy16"
        ),
    )
    parser.add_argument("--embedding-row-chunk", type=int, default=2048)
    parser.add_argument("--bias-tile-chunk", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def model_tensor_location(model: Path, name: str) -> Path:
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    return model / index["weight_map"][name]


def export_embedding_u8(
    model: Path, output: Path, qparam: dict[str, object], row_chunk: int,
) -> None:
    if row_chunk <= 0:
        raise ValueError("embedding row chunk must be positive")
    name = "model.embed_tokens.weight"
    shard = model_tensor_location(model, name)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with safe_open(shard, framework="pt", device="cpu") as source:
        embedding = source.get_slice(name)
        if tuple(embedding.get_shape()) != (VOCAB, HIDDEN):
            raise ValueError(f"unexpected embedding shape {embedding.get_shape()}")
        with temporary.open("wb") as stream:
            for first in range(0, VOCAB, row_chunk):
                last = min(first + row_chunk, VOCAB)
                values = embedding[first:last, :].float()
                encoded = torch.round(
                    values / float(qparam["scale"]) +
                    int(qparam["zero_point"])
                ).clamp(0, 255).to(torch.uint8)
                stream.write(
                    np.ascontiguousarray(encoded.cpu().numpy()).tobytes()
                )
    os.replace(temporary, output)
    if output.stat().st_size != VOCAB * HIDDEN:
        raise ValueError("U8 embedding has the wrong byte count")


def export_generation_qparams(path: Path) -> None:
    records = bytearray()
    for name, qparam in (
        ("generation_final_norm_output", FINAL_NORM_QPARAM),
        ("generation_lm_head_output", LM_HEAD_QPARAM),
    ):
        encoded_name = name.encode("ascii")
        records.extend(QPARAM_RECORD.pack(
            encoded_name + bytes(32 - len(encoded_name)),
            float(qparam["scale"]), int(qparam["zero_point"]),
            float(qparam["minimum"]), float(qparam["maximum"]),
        ))
    atomic_write(path, bytes(records))


def export_lm_head_bias(
    packed_path: Path, scales_path: Path, output: Path, tile_chunk: int,
) -> None:
    if tile_chunk <= 0:
        raise ValueError("bias tile chunk must be positive")
    n_tiles = VOCAB // HMX_CHANNELS
    k_tiles = HIDDEN // HMX_CHANNELS
    packed_tile_bytes = k_tiles * 512
    scales = np.fromfile(scales_path, dtype="<f4")
    if scales.shape != (VOCAB,):
        raise ValueError("LM-head scale shape mismatch")
    ratio = np.asarray(
        np.float32(FINAL_NORM_QPARAM["scale"]) * scales /
        np.float32(LM_HEAD_QPARAM["scale"]), dtype=np.float32,
    )
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError("LM-head requant ratio is invalid")
    lower = np.asarray(np.float32(512.0) * ratio, dtype=np.float32)
    lower = lower.astype("<f2").view("<u2").astype("<u4")

    temporary = output.with_suffix(output.suffix + ".tmp")
    with packed_path.open("rb") as packed_stream, temporary.open("wb") as out:
        for first_tile in range(0, n_tiles, tile_chunk):
            count = min(tile_chunk, n_tiles - first_tile)
            payload = packed_stream.read(count * packed_tile_bytes)
            if len(payload) != count * packed_tile_bytes:
                raise ValueError("short LM-head packed-weight read")
            packed = np.frombuffer(payload, dtype=np.uint8).reshape(
                count, k_tiles, 512
            )
            codes = np.empty((count, k_tiles, 1024), dtype=np.uint8)
            codes[..., 0::2] = packed & 0x0F
            codes[..., 1::2] = packed >> 4
            signed = codes.astype(np.int8)
            signed[signed >= 8] -= 16
            sums = signed.reshape(
                count, k_tiles, 8, HMX_CHANNELS, 4
            ).sum(axis=(1, 2, 4), dtype=np.int32)
            first_channel = first_tile * HMX_CHANNELS
            last_channel = first_channel + count * HMX_CHANNELS
            tile_ratio = ratio[first_channel:last_channel].reshape(
                count, HMX_CHANNELS
            )
            offset = (
                -float(FINAL_NORM_QPARAM["zero_point"]) *
                sums.astype(np.float64) +
                float(LM_HEAD_QPARAM["zero_point"]) /
                tile_ratio.astype(np.float64)
            )
            upper = round_half_away_from_zero(offset)
            if np.any(upper < np.iinfo(np.int32).min) or np.any(
                upper > np.iinfo(np.int32).max
            ):
                raise OverflowError("LM-head HMX bias exceeds int32")
            tile_lower = lower[first_channel:last_channel].reshape(
                count, HMX_CHANNELS
            )
            carrier = np.empty((count, 2, HMX_CHANNELS), dtype="<u4")
            carrier[:, 0, :] = tile_lower
            carrier[:, 1, :] = upper.astype("<i4").view("<u4")
            out.write(carrier.tobytes(order="C"))
    os.replace(temporary, output)
    if output.stat().st_size != n_tiles * HMX_BIAS_BYTES:
        raise ValueError("LM-head HMX bias carrier has the wrong size")


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    source = args.source.resolve()
    boundary = args.w4f16_boundary.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing archive: {output}")
    layer0_qparams = load_qparams_bin(source / "layer0/qparams_u8.bin")
    embedding_qparam = layer0_qparams["block_input"]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.publishing-", dir=output.parent
    ))
    shutil.rmtree(staging)
    try:
        shutil.copytree(source, staging, copy_function=os.link)
        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})

        copied_names = [
            "generation_prompt_token_ids_u32.bin",
            "generation_expected_token_ids_u32.bin",
            "generation_final_norm_weight_f16.bin",
            "generation_lm_head_weight_w4_hmx.bin",
            "generation_lm_head_weight_w4_scale_f32.bin",
            "rope_cos_f16.bin", "rope_sin_f16.bin",
        ]
        copied_names.extend(
            f"generation_decode_rope_{kind}_{index:02d}_f16.bin"
            for index in range(15) for kind in ("cos", "sin")
        )
        for name in copied_names:
            destination = staging / name
            if destination.exists():
                destination.unlink()
            link_or_copy(boundary / name, destination)
            files[name] = file_record(destination)

        embedding_path = staging / "generation_embedding_weight_u8.bin"
        export_embedding_u8(
            model, embedding_path, embedding_qparam,
            args.embedding_row_chunk,
        )
        files[embedding_path.name] = file_record(embedding_path)

        qparam_path = staging / "generation_qparams_u8.bin"
        export_generation_qparams(qparam_path)
        files[qparam_path.name] = file_record(qparam_path)

        bias_path = staging / "generation_lm_head_bias_u32.bin"
        export_lm_head_bias(
            staging / "generation_lm_head_weight_w4_hmx.bin",
            staging / "generation_lm_head_weight_w4_scale_f32.bin",
            bias_path, args.bias_tile_chunk,
        )
        files[bias_path.name] = file_record(bias_path)

        manifest["experiment"] = EXPERIMENT
        manifest["execution_unit"] = (
            "real_token_ids_u8_embedding_layers0_27_cache_native_"
            "final_norm_w4u8_lm_head_greedy_feedback"
        )
        manifest["generation"] = {
            "prompt_tokens": 64,
            "generated_tokens": 16,
            "cache_capacity": 80,
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "embedding_dtype": "asymmetric_u8_layer0_block_input_qparam",
            "final_norm_gamma_dtype": "fp16",
            "final_norm_output_qparam": FINAL_NORM_QPARAM,
            "lm_head_weight": "signed_symmetric_per_output_channel_w4",
            "lm_head_output_qparam": LM_HEAD_QPARAM,
            "lm_head_output": "shared_u8_logit_domain_argmax_only",
            "full_logits_ddr": False,
            "teacher_token_ids": "diagnostic_only_not_a_quality_gate",
            "model_quality_gate": False,
            "implementation_gate": (
                "independent captured-hidden final-norm and integer-HMX "
                "argmax reference"
            ),
        }
        atomic_write(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2,
                        sort_keys=True) + "\n").encode("utf-8"),
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"PUBLISHED={output}")
    print(f"EMBEDDING_BYTES={(output / 'generation_embedding_weight_u8.bin').stat().st_size}")
    print(f"LM_HEAD_BIAS_BYTES={(output / 'generation_lm_head_bias_u32.bin').stat().st_size}")


if __name__ == "__main__":
    main()
