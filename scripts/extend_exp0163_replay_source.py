#!/usr/bin/env python3
"""Extend the accepted EXP-0162 W4U8 replay inputs through position 255.

Only layer-0 replay inputs and RoPE rows are extended from the original model.
The independent HMX/HVX simulator subsequently regenerates every block output
and cache reference, so this script never treats earlier device output as a
new numerical authority.
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
from tokenizers import Tokenizer

import export_exp0022_block as base
from reference_w4u8_hmx import load_qparams_bin


PREFILL = 64
CAPACITY = 257
FINAL_VALID = 256
DECODE_STEPS = FINAL_VALID - PREFILL
PHYSICAL_M = 64
HIDDEN = 2048
HEAD_DIM = 128
KV_HEADS = 8
LAYERS = 28


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_array(path: Path, value: np.ndarray) -> None:
    path.unlink(missing_ok=True)
    np.ascontiguousarray(value).tofile(path)


def token_ids(model: Path) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model / "qwen3-tokenizer.json"))
    repeats = 2
    while True:
        values = tokenizer.encode(
            base.PROMPT * repeats, add_special_tokens=True
        ).ids
        if len(values) >= FINAL_VALID:
            return values[:FINAL_VALID]
        repeats *= 2


def embedding_rows(model: Path, ids: list[int]) -> torch.Tensor:
    index = json.loads(
        (model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    key = "model.embed_tokens.weight"
    shard = model / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor(key)
        return weight[torch.tensor(ids, dtype=torch.long)].float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    prior = args.prior.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    prior_manifest_path = prior / "manifest.json"
    if not prior_manifest_path.is_file():
        raise FileNotFoundError(prior_manifest_path)

    manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_ids = [int(value) for value in manifest["token_ids"]]
    ids = token_ids(model)
    if ids[:len(prior_ids)] != prior_ids:
        raise ValueError("extended token sequence changed the EXP-0162 prefix")

    qparams = load_qparams_bin(prior / "layer0/qparams_u8.bin")
    input_qparam = qparams["block_input"]
    encoded = base.quantize_u8(
        embedding_rows(model, ids), input_qparam
    ).cpu().numpy().reshape(FINAL_VALID, HIDDEN)
    base.M = FINAL_VALID
    cos_tensor, sin_tensor = base.rope_tables(torch.float16)
    cosine = cos_tensor.detach().float().cpu().numpy().astype("<f2")
    sine = sin_tensor.detach().float().cpu().numpy().astype("<f2")
    identity_cos = np.ones((PHYSICAL_M, HEAD_DIM), dtype="<f2")
    identity_sin = np.zeros((PHYSICAL_M, HEAD_DIM), dtype="<f2")
    fill = int(input_qparam["zero_point"])

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    ))
    shutil.rmtree(staging)
    try:
        try:
            shutil.copytree(prior, staging, copy_function=os.link)
            clone_mode = "hardlink"
        except OSError:
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(prior, staging)
            clone_mode = "copy"

        changed: list[Path] = []
        for index in range(DECODE_STEPS):
            position = PREFILL + index
            input_row = np.full(
                (PHYSICAL_M, HIDDEN), fill, dtype=np.uint8
            )
            input_row[0] = encoded[position]
            step_cos = identity_cos.copy()
            step_sin = identity_sin.copy()
            step_cos[0] = cosine[position]
            step_sin[0] = sine[position]
            payloads = (
                (f"replay_decode_input_{index:02d}_u8.bin", input_row),
                (f"replay_decode_rope_cos_{index:02d}_f16.bin", step_cos),
                (f"replay_decode_rope_sin_{index:02d}_f16.bin", step_sin),
            )
            for name, value in payloads:
                path = staging / name
                expected = np.ascontiguousarray(value).tobytes()
                if index < 40:
                    if path.read_bytes() != expected:
                        raise ValueError(f"EXP-0162 replay prefix changed: {name}")
                else:
                    replace_array(path, value)
                    changed.append(path)

        empty_cache = np.zeros(
            (KV_HEADS, CAPACITY, HEAD_DIM), dtype=np.uint8
        )
        for layer in range(LAYERS):
            for kind in ("k", "v"):
                for prefix in ("kv_cache", "reference_kv_cache"):
                    path = staging / f"layer{layer}/{prefix}_{kind}_u8.bin"
                    replace_array(path, empty_cache)
                    changed.append(path)

        files = manifest.setdefault("files", {})
        for path in changed:
            files[str(path.relative_to(staging))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest.pop("exact_reference_revision", None)
        manifest["experiment"] = "EXP-0163"
        manifest["token_ids"] = ids
        manifest["clone_mode"] = clone_mode
        manifest["contract"].update({
            "cache_capacity_per_layer": CAPACITY,
            "decode_positions": [PREFILL, FINAL_VALID - 1],
        })
        manifest["source_extension_revision"] = {
            "kind": "layer0_embedding_and_RoPE_extension",
            "prior": str(prior),
            "prior_manifest_sha256": sha256(prior_manifest_path),
            "capacity": CAPACITY,
            "final_valid_length": FINAL_VALID,
            "decode_steps": DECODE_STEPS,
            "retained_exp0162_input_prefix_bytes_exact": True,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        print(json.dumps({
            "experiment": "EXP-0163",
            "output": str(output),
            "clone_mode": clone_mode,
            "decode_steps": DECODE_STEPS,
            "retained_input_prefix_gate": "PASS",
            "manifest_sha256": sha256(output / "manifest.json"),
        }, indent=2, sort_keys=True))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
