#!/usr/bin/env python3
"""Replace EXP-0147 device-golden decode references with exact host references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from reference_w4u8_hmx import HmxU8Converter, load_qparams_bin
from verify_exp0152_w4u8_replay import run_layer_decode


M = 64
HIDDEN = 2048
KV_HEADS = 8
HEAD_DIM = 128
SUPPORTED_LENGTHS = (64, 256, 1024, 4096)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_package(root: Path, past_length: int) -> Path:
    suffix = "_v2" if past_length == 64 else ""
    return root / f"decode_l{past_length}_w4u8{suffix}"


def replace_array(path: Path, value: np.ndarray) -> None:
    path.unlink()
    np.ascontiguousarray(value).tofile(path)


def publish_one(
    source: Path,
    output: Path,
    past_length: int,
    converter: HmxU8Converter,
    converter_path: Path,
) -> dict[str, object]:
    capacity = past_length + 1
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    ))
    shutil.rmtree(staging)
    try:
        try:
            shutil.copytree(source, staging, copy_function=os.link)
            clone_mode = "hardlink"
        except OSError:
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(source, staging)
            clone_mode = "copy"

        input_u8 = np.fromfile(
            source / "reference_w4u8_block_input_u8.bin", dtype=np.uint8
        ).reshape(M, HIDDEN)
        k_initial = np.fromfile(
            source / "kv_cache_k_u8.bin", dtype=np.uint8
        ).reshape(KV_HEADS, capacity, HEAD_DIM).copy()
        v_initial = np.fromfile(
            source / "kv_cache_v_u8.bin", dtype=np.uint8
        ).reshape(KV_HEADS, capacity, HEAD_DIM).copy()
        k_reference = k_initial.copy()
        v_reference = v_initial.copy()
        cosine = np.fromfile(source / "rope_cos_f16.bin", dtype="<f2")
        sine = np.fromfile(source / "rope_sin_f16.bin", dtype="<f2")
        output_row, k_append, v_append = run_layer_decode(
            input_u8[:1], source, cosine, sine,
            k_reference, v_reference, past_length, converter,
        )

        qparams_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        output_zero_point = int(
            load_qparams_bin(source / "qparams_u8.bin")["block_output"][
                "zero_point"
            ]
        )
        output_reference = np.full(
            (M, HIDDEN), output_zero_point, dtype=np.uint8
        )
        output_reference[0] = output_row[0]

        replace_array(staging / "reference_kv_cache_k_u8.bin", k_reference)
        replace_array(staging / "reference_kv_cache_v_u8.bin", v_reference)
        replace_array(
            staging / "reference_w4u8_integer_attention_block_output_u8.bin",
            output_reference,
        )
        replace_array(
            staging / "reference_exp0147_cpu_block_output_u8.bin",
            output_reference,
        )

        manifest = qparams_manifest
        manifest.update({
            "experiment": "EXP-0161",
            "exact_reference_revision": 1,
            "exact_reference_source": "independent_host_HMX_HVX_simulator",
            "exact_reference_converter": str(converter_path),
            "exact_reference_converter_sha256": sha256(converter_path),
            "inherited_synthetic_cache_fixture": (
                "EXP0147 accepted M64 device-boundary history; only the "
                "current-token append and resulting block output are replaced"
            ),
            "source_package": str(source),
            "source_manifest_sha256": sha256(source_manifest_path),
            "clone_mode": clone_mode,
        })
        files = manifest.setdefault("files", {})
        for name in (
            "reference_kv_cache_k_u8.bin",
            "reference_kv_cache_v_u8.bin",
            "reference_w4u8_integer_attention_block_output_u8.bin",
            "reference_exp0147_cpu_block_output_u8.bin",
        ):
            path = staging / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest_path = staging / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        return {
            "past_length": past_length,
            "output": str(output),
            "clone_mode": clone_mode,
            "k_append_sha256": hashlib.sha256(k_append.tobytes()).hexdigest(),
            "v_append_sha256": hashlib.sha256(v_append.tobytes()).hexdigest(),
            "block_output_sha256": hashlib.sha256(
                output_reference.tobytes()
            ).hexdigest(),
            "manifest_sha256": sha256(output / "manifest.json"),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0147"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/exact_logical_v1"
        ),
    )
    parser.add_argument(
        "--converter", type=Path,
        default=Path(
            "/home/daniuniu/work/qwen3-block-htp/build/reference/"
            "qbh_hmx_u8_reference.so"
        ),
    )
    parser.add_argument("--length", type=int, action="append", choices=SUPPORTED_LENGTHS)
    args = parser.parse_args()
    converter_path = args.converter.resolve()
    converter = HmxU8Converter(converter_path)
    results = []
    for past_length in tuple(args.length or SUPPORTED_LENGTHS):
        source = source_package(args.source_root.resolve(), past_length)
        output = source_package(args.output_root.resolve(), past_length)
        result = publish_one(
            source, output, past_length, converter, converter_path
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"experiment": "EXP-0161", "packages": results}, indent=2))


if __name__ == "__main__":
    main()
