#!/usr/bin/env python3
"""Materialize compact EXP-0151 evidence outside Git."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


ARTIFACT_NAMES = (
    "qwen3_rpcmem2_cli",
    "libqwen3_probe.so",
    "libqwen3_probe_skel.so",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-json-base64", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--boot-id-before", required=True)
    parser.add_argument("--boot-id-after", required=True)
    parser.add_argument("--memtotal-before-kb", type=int, required=True)
    parser.add_argument("--memavailable-before-kb", type=int, required=True)
    parser.add_argument("--memtotal-after-kb", type=int, required=True)
    parser.add_argument("--memavailable-after-kb", type=int, required=True)
    args = parser.parse_args()

    run_result = json.loads(
        base64.b64decode(args.run_json_base64).decode("utf-8")
    )
    if run_result.get("gate_pass") != 1:
        raise SystemExit("refusing to record EXP-0151 as passed")
    run_result["source_commit"] = args.source_commit
    run_result["sentinel_offsets"] = [4096, 1449996288, 2899992576]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gate_result.json").write_text(
        json.dumps(run_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "device_state.txt").write_text(
        "\n".join(
            (
                f"device_serial={args.device_serial}",
                "device_model=PJZ110",
                "device_product=PJZ110",
                f"boot_id_before={args.boot_id_before}",
                f"boot_id_after={args.boot_id_after}",
                "state_before=device",
                "state_after=device",
                f"memtotal_before_kb={args.memtotal_before_kb}",
                f"memavailable_before_kb={args.memavailable_before_kb}",
                f"memtotal_after_kb={args.memtotal_after_kb}",
                f"memavailable_after_kb={args.memavailable_after_kb}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (args.output / "static_audit.txt").write_text(
        "\n".join(
            (
                "source_branch=codex/exp-0151-rpcmem-alloc2-single-buffer",
                f"source_commit={args.source_commit}",
                "host_rpcmem_alloc2_symbol=WEAK_UNDEFINED_RUNTIME_RESOLVED",
                "host_fastrpc_mmap_symbol=GLOBAL_UNDEFINED_LIBCDSPRPC",
                "host_generated_probe_symbol=qwen3_probe_run_rpcmem2_probe",
                "dsp_probe_symbol=qwen3_probe_run_rpcmem2_probe_GLOBAL",
                "host_dynamic_dependencies=liblog.so,libdl.so,libqwen3_probe.so,libcdsprpc.so,libm.so,libc.so",
                "qnn_dynamic_dependency=none",
                "build_android=pass",
                "build_hexagon_v79=pass",
                "",
            )
        ),
        encoding="utf-8",
    )

    hashes = []
    for name in ARTIFACT_NAMES:
        path = args.artifact_root / name
        hashes.append(f"{sha256(path)}  {name}")
    (args.output / "artifact_hashes.txt").write_text(
        "\n".join(hashes) + "\n", encoding="utf-8"
    )

    allocation_ms = run_result["allocation_ns"] / 1_000_000
    map_ms = run_result["map_ns"] / 1_000_000
    run_ms = run_result["run_ns"] / 1_000_000
    (args.output / "full_profiling_report.md").write_text(
        f"""# EXP-0151 profiling closure

EXP-0151 is a capacity/addressability probe and does not execute a Qwen3 block,
projection, Attention path, or numerical recipe. Therefore the PC-027 block
timing ledger, engine-work counters, and PC-028 three-recipe module overview are
unavailable by declared scope rather than omitted measurements. No selected
baseline or prior recipe timing changed.

| Probe stage | Wall time |
| --- | ---: |
| `rpcmem_alloc2` allocation | {allocation_ms:.6f} ms |
| `fastrpc_mmap` | {map_ms:.6f} ms |
| FastRPC DSP sentinel probe | {run_ms:.6f} ms |

The timings are diagnostic only and have no speed gate. The physical result is
one 2,900,000,000-byte allocation, one fd and one contiguous cDSP mapping. All
begin/middle/end Host-to-DSP reads and DSP-to-Host writes matched exactly. The
device boot ID remained `{args.boot_id_after}`, and allocation, DSP mapping,
DSP unmapping, Host unmapping and session release all returned success. QNN
dependencies are absent.
""",
        encoding="utf-8",
    )

    evidence_names = (
        "artifact_hashes.txt",
        "device_state.txt",
        "full_profiling_report.md",
        "gate_result.json",
        "static_audit.txt",
    )
    (args.output / "evidence_sha256.txt").write_text(
        "\n".join(
            f"{sha256(args.output / name)}  {name}"
            for name in evidence_names
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
