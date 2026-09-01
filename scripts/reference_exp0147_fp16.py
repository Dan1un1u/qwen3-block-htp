#!/usr/bin/env python3
"""Generate an independent dynamic-cell FP16 block reference for EXP-0147."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from export_exp0022_block import linear_half, rms_norm
from prepare_exp0042_block import unpack_w4_weight


PHYSICAL_M = 64
HIDDEN = 2048
INTERMEDIATE = 6144
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
Q_HEADS_PER_GROUP = HEADS // KV_HEADS
PROJECTION_SHAPES = {
    "o": (HIDDEN, HIDDEN),
    "gate": (INTERMEDIATE, HIDDEN),
    "up": (INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--recipe", choices=("f16f16", "w4f16"), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_fp16_weight(package: Path, name: str, n: int, k: int) -> torch.Tensor:
    packed = np.fromfile(package / f"{name}_weight_f16_hmx.bin", dtype="<f2")
    packed = packed.reshape(n // 32, k // 32, 16, 32, 2)
    logical = (
        packed.transpose(0, 1, 2, 4, 3)
        .reshape(n // 32, k // 32, 32, 32)
        .transpose(0, 2, 1, 3)
        .reshape(n, k)
    )
    return torch.from_numpy(np.ascontiguousarray(logical).copy())


def dynamic_attention(
    q_rows: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    past: int,
) -> torch.Tensor:
    query_rows = q_rows.shape[0]
    q = torch.from_numpy(q_rows.copy()).to(torch.float16)
    k = torch.from_numpy(k_cache.copy()).to(torch.float16)
    v = torch.from_numpy(v_cache.copy()).to(torch.float16)
    output = torch.empty(query_rows, HEADS, HEAD_DIM, dtype=torch.float16)
    scale = HEAD_DIM ** -0.5
    for group in range(KV_HEADS):
        first_head = group * Q_HEADS_PER_GROUP
        q_group = q[:, first_head : first_head + Q_HEADS_PER_GROUP].transpose(0, 1)
        scores = torch.matmul(q_group.float(), k[group].float().transpose(0, 1))
        scores = (scores * scale).to(torch.float16)
        for row in range(query_rows):
            valid = past + row + 1
            probability = torch.softmax(scores[:, row, :valid].float(), dim=-1).to(
                torch.float16
            )
            output[row, first_head : first_head + Q_HEADS_PER_GROUP] = torch.matmul(
                probability.float(), v[group, :valid].float()
            ).to(torch.float16)
    return output.reshape(query_rows, HIDDEN)


def block_tail(
    hidden: torch.Tensor,
    attention_output: torch.Tensor,
    weights: dict[str, torch.Tensor],
    post_weight: torch.Tensor,
) -> torch.Tensor:
    projected = linear_half(attention_output, weights["o"])
    post_residual = (hidden + projected).to(torch.float16)
    post_norm = rms_norm(post_residual, post_weight)
    gate = linear_half(post_norm, weights["gate"])
    up = linear_half(post_norm, weights["up"])
    middle = (F.silu(gate.float()) * up.float()).to(torch.float16)
    down = linear_half(middle, weights["down"])
    return (post_residual + down).to(torch.float16)


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scan = manifest["shape_scan"]
    logical_m = int(scan["logical_m"])
    past = int(scan["initial_kv_length"])
    capacity = int(scan["kv_cache_capacity"])
    chunks = int(scan["physical_chunks"])
    block_input = np.fromfile(package / "block_input_f16.bin", dtype="<f2").reshape(
        chunks * PHYSICAL_M, HIDDEN
    )
    output_path = package / f"reference_{args.recipe}_block_output_f16.bin"
    output = np.fromfile(output_path, dtype="<f2").reshape(
        chunks * PHYSICAL_M, HIDDEN
    )
    q_base = np.fromfile(
        package / f"reference_{args.recipe}_q_rope_f16.bin", dtype="<f2"
    ).reshape(PHYSICAL_M, HEADS, HEAD_DIM)
    k_cache = np.fromfile(
        package / "reference_kv_cache_k_f16.bin", dtype="<f2"
    ).reshape(KV_HEADS, capacity, HEAD_DIM)
    v_cache = np.fromfile(
        package / "reference_kv_cache_v_f16.bin", dtype="<f2"
    ).reshape(KV_HEADS, capacity, HEAD_DIM)
    if args.recipe == "f16f16":
        weights = {
            name: unpack_fp16_weight(package, name, n, k)
            for name, (n, k) in PROJECTION_SHAPES.items()
        }
    else:
        weights = {
            name: unpack_w4_weight(package, name, n, k)
            for name, (n, k) in PROJECTION_SHAPES.items()
        }
    post_weight = torch.from_numpy(
        np.fromfile(package / "post_norm_weight_f16.bin", dtype="<f2").copy()
    )

    if scan["mode"] == "decode":
        query_rows = 1
        first_output = 0
    elif logical_m > PHYSICAL_M:
        query_rows = PHYSICAL_M
        past = PHYSICAL_M
        first_output = PHYSICAL_M
    else:
        print(json.dumps({"package": str(package), "dynamic_reference": False}, indent=2))
        return

    attention = dynamic_attention(
        q_base[:query_rows], k_cache, v_cache, past
    )
    hidden = torch.from_numpy(
        block_input[first_output : first_output + query_rows].copy()
    ).to(torch.float16)
    reference = block_tail(hidden, attention, weights, post_weight)
    output[first_output : first_output + query_rows] = reference.cpu().numpy()

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=package)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        output.tofile(temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest["independent_dynamic_reference"] = {
        "method": "FP32 matmul/Softmax with FP16 recipe boundaries",
        "source": "package recipe-specific Q/K/V and logical weights",
        "file": output_path.name,
        "sha256": sha256(output_path),
    }
    manifest["files"][output_path.name] = {
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"package": str(package), **manifest["independent_dynamic_reference"]}, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    main()

