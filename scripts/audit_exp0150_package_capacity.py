#!/usr/bin/env python3
"""Audit the immutable lower bound for the EXP-0150 shared package.

The current Host ABI stores every projection carrier in one rpcmem allocation.
Android's rpcmem_alloc API takes a signed int byte count, so any recipe whose
immutable weight payload alone exceeds INT_MAX cannot reach the device mapping
smoke.  This audit intentionally uses a lower bound: metadata, references,
caches, alignment and runtime-generated carriers can only make the package
larger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYERS = 28
INT32_MAX = (1 << 31) - 1
UINT32_MAX = (1 << 32) - 1
PROJECTIONS = {
    "q": (2048, 2048),
    "k": (2048, 1024),
    "v": (2048, 1024),
    "o": (2048, 2048),
    "gate": (2048, 6144),
    "up": (2048, 6144),
    "down": (6144, 2048),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    elements_per_layer = sum(k * n for k, n in PROJECTIONS.values())
    f16_weight_bytes_per_layer = elements_per_layer * 2
    f16_weight_bytes = f16_weight_bytes_per_layer * LAYERS
    w4_weight_bytes_per_layer = elements_per_layer // 2
    w4_weight_bytes = w4_weight_bytes_per_layer * LAYERS
    w4_scale_bytes_per_layer = sum(n for _, n in PROJECTIONS.values()) * 4
    w4_weight_and_scale_bytes = (
        w4_weight_bytes + w4_scale_bytes_per_layer * LAYERS
    )
    report = {
        "experiment": "EXP-0150",
        "stage": "A_static_package_capacity",
        "gate": "fail",
        "current_abi": "one_rpcmem_allocation_one_shared_fd",
        "declared_layers": LAYERS,
        "projection_shapes_k_n": {
            name: [k, n] for name, (k, n) in PROJECTIONS.items()
        },
        "rpcmem_alloc_signed_int_max_bytes": INT32_MAX,
        "shared_header_uint32_max_bytes": UINT32_MAX,
        "f16f16": {
            "weight_bytes_per_layer": f16_weight_bytes_per_layer,
            "immutable_weight_lower_bound_bytes": f16_weight_bytes,
            "bytes_over_rpcmem_limit": f16_weight_bytes - INT32_MAX,
            "fits_current_rpcmem_allocation": f16_weight_bytes <= INT32_MAX,
        },
        "w4_lower_bound": {
            "weight_bytes_per_layer": w4_weight_bytes_per_layer,
            "scale_bytes_per_layer": w4_scale_bytes_per_layer,
            "immutable_weight_and_scale_lower_bound_bytes":
                w4_weight_and_scale_bytes,
            "fits_current_rpcmem_allocation_lower_bound":
                w4_weight_and_scale_bytes <= INT32_MAX,
        },
        "omitted_positive_terms": [
            "request_and_telemetry_header",
            "128_byte_alignment",
            "input_output_and_independent_references",
            "rope_metadata",
            "28_layer_norm_weights_and_qparams",
            "28_independent_kv_caches_and_cache_references",
            "w4u8_bias_and_streaming_bundle_carriers",
            "w4f16_scale_caches",
        ],
        "conclusion": (
            "F16F16 projection weights alone exceed the signed-int rpcmem "
            "allocation limit; the complete three-recipe package mapping "
            "gate cannot pass under the current single-buffer ABI."
        ),
        "required_action": (
            "stop EXP-0150 before package generation/device execution and "
            "approve a segmented persistent-mapping ABI; do not refill "
            "weights between layers or issue 28 Host RPC calls"
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
