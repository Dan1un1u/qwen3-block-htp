#!/usr/bin/env python3
"""Publish the retained EXP-0169 long-generation overlay.

The 28-layer capacity-257 transformer package and the selected EXP-0168
generation package remain immutable.  This directory contains hard links to
the generation boundary tensors plus 192 decode-position RoPE carriers linked
from the verified EXP-0163 replay source.  Its manifest records both parents so
the device deployer can assemble the complete package without duplicating the
multi-gigabyte transformer archive on D:.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


EXPERIMENT = "EXP-0169"
PREFILL_TOKENS = 64
DECODE_STEPS = 192
TOTAL_STEPS = 1 + DECODE_STEPS
CACHE_CAPACITY = 257
KV_HEADS = 8
HEAD_DIM = 128
HMX_INPUT_CHANNELS = 32
HMX_OUTPUT_CHANNELS = 32
HMX_BIAS_BYTES = 256


def segmented_cache_bytes(capacity: int, kind: str) -> int:
    """Mirror the segmented-v4 cache ABI in include/block_protocol.h."""
    segment_count = (capacity - 1) // HMX_INPUT_CHANNELS
    segment_weight_bytes = HEAD_DIM * HMX_INPUT_CHANNELS
    tail_bytes = HMX_INPUT_CHANNELS * HEAD_DIM
    if kind == "k":
        per_head = (
            segment_count * (segment_weight_bytes + HMX_BIAS_BYTES)
            + tail_bytes
        )
    elif kind == "v":
        per_head = (
            segment_count * segment_weight_bytes
            + HEAD_DIM // HMX_OUTPUT_CHANNELS * HMX_BIAS_BYTES
            + tail_bytes
        )
    else:
        raise ValueError(f"unsupported cache kind: {kind}")
    return KV_HEADS * per_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer-package", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0163/"
            "candidate_segmented_capacity257"
        ),
    )
    parser.add_argument(
        "--generation-package", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0167/"
            "w4u8_greedy16"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0169/"
            "w4u8_greedy193_overlay"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    transformer = args.transformer_package.resolve()
    generation = args.generation_package.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace retained overlay: {output}")

    transformer_manifest = transformer / "manifest.json"
    generation_manifest = generation / "manifest.json"
    if not transformer_manifest.is_file() or not generation_manifest.is_file():
        raise FileNotFoundError("parent package manifest is missing")
    for layer in range(28):
        for kind in ("k", "v"):
            path = transformer / f"layer{layer}/kv_cache_{kind}_hmx_u8_segmented.bin"
            expected_bytes = segmented_cache_bytes(CACHE_CAPACITY, kind)
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    "capacity-257 segmented cache mismatch: "
                    f"{path}: expected={expected_bytes}, actual={actual_bytes}"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.publishing-", dir=output.parent,
    ))
    try:
        files: dict[str, dict[str, object]] = {}
        generation_names = (
            "generation_prompt_token_ids_u32.bin",
            "generation_expected_token_ids_u32.bin",
            "generation_embedding_weight_u8.bin",
            "generation_final_norm_weight_f16.bin",
            "generation_lm_head_weight_w4_hmx.bin",
            "generation_lm_head_weight_w4_scale_f32.bin",
            "generation_lm_head_bias_u32.bin",
            "generation_qparams_u8.bin",
        )
        for name in generation_names:
            source = generation / name
            destination = staging / name
            link_or_copy(source, destination)
            files[name] = file_record(destination)

        for index in range(DECODE_STEPS):
            for kind in ("cos", "sin"):
                source = transformer / (
                    f"replay_decode_rope_{kind}_{index:02d}_f16.bin"
                )
                destination = staging / (
                    f"generation_decode_rope_{kind}_{index:02d}_f16.bin"
                )
                link_or_copy(source, destination)
                files[destination.name] = file_record(destination)

        manifest = {
            "experiment": EXPERIMENT,
            "execution_unit": (
                "real_token_ids_M64_prefill_then_192_continuous_W4U8_"
                "decode_steps_to_valid_length256"
            ),
            "assembly": {
                "transformer_package": str(transformer),
                "transformer_manifest_sha256": sha256_file(transformer_manifest),
                "generation_package": str(generation),
                "generation_manifest_sha256": sha256_file(generation_manifest),
                "method": "device_symlink_overlay_no_parent_archive_mutation",
            },
            "generation": {
                "prefill_tokens": PREFILL_TOKENS,
                "decode_steps": DECODE_STEPS,
                "total_accelerator_passes": TOTAL_STEPS,
                "final_valid_length": PREFILL_TOKENS + DECODE_STEPS,
                "cache_capacity": CACHE_CAPACITY,
                "expected_token_reference_count": 16,
                "semantic_quality_gate": False,
                "timed_full_logits_ddr": False,
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"PUBLISHED={output}")
    print(f"OVERLAY_FILES={len(files)}")
    print(f"TOTAL_STEPS={TOTAL_STEPS}")


if __name__ == "__main__":
    main()
