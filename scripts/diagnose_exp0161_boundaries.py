#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from reference_w4u8_hmx import (
    HmxU8Converter,
    exact_qk_norm_rope_u8,
    exact_rms_norm_u8,
    load_qparams_bin,
    project_w4u8,
)


M = 64
HEAD_DIM = 128
Q_HEADS = 16
KV_HEADS = 8


def unpack_feature_tiles(path: Path, heads: int) -> np.ndarray:
    physical = np.fromfile(path, dtype=np.uint8).reshape(
        heads, HEAD_DIM // 32, M, 32
    )
    return physical.transpose(2, 0, 1, 3).reshape(M, heads, HEAD_DIM)


def difference(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    return {
        "elements": int(delta.size),
        "mismatches": int(np.count_nonzero(delta)),
        "max_lsb": int(delta.max(initial=0)),
        "mean_absolute_lsb": float(delta.mean()),
        "rmse_lsb": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--old-audit", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual = {
        "q": unpack_feature_tiles(args.dump / "actual_q_tiles_u8.bin", Q_HEADS),
        "k": unpack_feature_tiles(args.dump / "actual_k_tiles_u8.bin", KV_HEADS),
        "v": unpack_feature_tiles(args.dump / "actual_v_tiles_u8.bin", KV_HEADS),
    }
    old = {
        "q": unpack_feature_tiles(args.old_audit / "actual_q_tiles_u8.bin", Q_HEADS),
        "k": unpack_feature_tiles(args.old_audit / "actual_k_tiles_u8.bin", KV_HEADS),
        "v": unpack_feature_tiles(args.old_audit / "actual_v_tiles_u8.bin", KV_HEADS),
    }
    cpu = {
        "q": np.fromfile(
            args.reference / "reference_w4u8_q_rope_u8.bin", dtype=np.uint8
        ).reshape(M, Q_HEADS, HEAD_DIM),
        "k": np.fromfile(
            args.reference / "reference_w4u8_k_rope_u8.bin", dtype=np.uint8
        ).reshape(M, KV_HEADS, HEAD_DIM),
        "v": np.fromfile(
            args.reference / "reference_w4u8_v_u8.bin", dtype=np.uint8
        ).reshape(M, KV_HEADS, HEAD_DIM),
    }

    qparams = load_qparams_bin(args.package / "qparams_u8.bin")
    input_u8 = np.fromfile(
        args.package / "reference_w4u8_block_input_u8.bin", dtype=np.uint8
    ).reshape(M, 2048)
    input_norm = exact_rms_norm_u8(
        input_u8,
        qparams["block_input"],
        np.fromfile(args.package / "input_norm_weight_f16.bin", dtype="<f2"),
        qparams["input_norm"],
    )
    converter = HmxU8Converter(args.converter)
    q_projection = project_w4u8(
        input_norm, args.package, "q", 2048, 2048,
        qparams["input_norm"], qparams["q_projection"], converter,
    )
    k_projection = project_w4u8(
        input_norm, args.package, "k", 1024, 2048,
        qparams["input_norm"], qparams["k_projection"], converter,
    )
    v_projection = project_w4u8(
        input_norm, args.package, "v", 1024, 2048,
        qparams["input_norm"], qparams["v"], converter,
    )
    cosine = np.fromfile(args.package / "rope_cos_f16.bin", dtype="<f2")
    sine = np.fromfile(args.package / "rope_sin_f16.bin", dtype="<f2")
    exact = {
        "q": exact_qk_norm_rope_u8(
            q_projection, Q_HEADS, qparams["q_projection"], qparams["q_rope"],
            np.fromfile(args.package / "q_norm_weight_f16.bin", dtype="<f2"),
            cosine, sine,
        ).reshape(M, Q_HEADS, HEAD_DIM),
        "k": exact_qk_norm_rope_u8(
            k_projection, KV_HEADS, qparams["k_projection"], qparams["k_rope"],
            np.fromfile(args.package / "k_norm_weight_f16.bin", dtype="<f2"),
            cosine, sine,
        ).reshape(M, KV_HEADS, HEAD_DIM),
        "v": v_projection.reshape(M, KV_HEADS, HEAD_DIM),
    }

    cache_k = np.fromfile(
        args.dump / "actual_kv_cache_k_u8.bin", dtype=np.uint8
    )
    cache_v = np.fromfile(
        args.dump / "actual_kv_cache_v_u8.bin", dtype=np.uint8
    )
    k_head_bytes = cache_k.size // KV_HEADS
    v_head_bytes = cache_v.size // KV_HEADS
    k_tail_offset = 2 * (4096 + 256)
    v_tail_offset = 2 * 4096 + 1024
    cache_current = {
        "k": np.stack(
            [
                cache_k[head * k_head_bytes + k_tail_offset:
                        head * k_head_bytes + k_tail_offset + HEAD_DIM]
                for head in range(KV_HEADS)
            ]
        ),
        "v": np.stack(
            [
                cache_v[head * v_head_bytes + v_tail_offset:
                        head * v_head_bytes + v_tail_offset + HEAD_DIM]
                for head in range(KV_HEADS)
            ]
        ),
    }

    report: dict[str, object] = {"row0": {}}
    for name in ("q", "k", "v"):
        report["row0"][name] = {
            "current_vs_exact_dsp_reference": difference(
                actual[name][0], exact[name][0]
            ),
            "current_vs_cpu": difference(actual[name][0], cpu[name][0]),
            "current_vs_old_device": difference(actual[name][0], old[name][0]),
            "old_device_vs_cpu": difference(old[name][0], cpu[name][0]),
        }
    report["full_physical_m64_vs_exact_dsp_reference"] = {
        name: difference(actual[name], exact[name]) for name in ("q", "k", "v")
    }
    report["cache_carrier"] = {
        "k_current_vs_attention_boundary": difference(
            cache_current["k"], actual["k"][0]
        ),
        "v_current_vs_attention_boundary": difference(
            cache_current["v"], actual["v"][0]
        ),
        "k_current_vs_cpu": difference(cache_current["k"], cpu["k"][0]),
        "v_current_vs_cpu": difference(cache_current["v"], cpu["v"][0]),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
