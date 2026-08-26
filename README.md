# qwen3-block-htp

Standalone research runtime for executing one Qwen3 transformer block on a
Qualcomm Hexagon HTP without QNN. The project owns the block schedule, VTCM
layout, DMA transfers, HVX work, and HMX kernels.

The first experiment is a minimal FastRPC bring-up. It intentionally does not
implement a graph runtime or full-model inference.

Project constraints and experiment state live on the independent
`codex/qwen3-block-project-memory` branch.

## EXP-0001

The bring-up probe allocates one uncached rpcmem region on Android, maps it into
cDSP, acquires VTCM through the HAP compute-resource API, performs a deterministic
U8 vector transform entirely in VTCM, and writes a compact qtimer report back to
the shared region. It uses one FastRPC invocation and has no QNN dependency.

Build with the isolated toolchain:

```sh
scripts/build_exp0001.sh
```

The implementation structure and FastRPC session setup are adapted from
`htp-ops-lib` commit `85eb88edcafd35afff1a43606a4c47eec9a0ca0b`;
the build and resource APIs are checked against Hexagon SDK 6.6.0.0 examples.

## EXP-0002

The integer-HMX tile probe uses the same standalone FastRPC substrate to run a
64×32 asymmetric-U8 activation by a 32×32 signed-S8 weight tile on V79 HMX.
The DSP service owns the VTCM layout, K4/N weight packing, activation zero-point
correction, HMX power/resource lifecycle, and saturated-U8 output. HMX executes
on a dedicated QuRT worker; the FastRPC thread acquires HVX only after that
worker completes, then copies the result from VTCM to the shared RPC boundary.
QNN is not used.

Standalone execution must explicitly power HMX with `HAP_power_set_HMX` before
issuing matrix instructions. A successful `HAP_compute_res_hmx_lock2` alone is
not sufficient. Omitting the power vote lets the instruction sequence issue,
but the first consumer of the HMX-written VTCM output blocks indefinitely.

Build and run a correctness pattern with the isolated toolchain:

```sh
scripts/build_exp0002.sh
scripts/run_exp0002.sh identity 1
```

The four correctness patterns are `identity`, `signed`, `structured`, and
`boundary`. Each result is checked byte-for-byte against an independent
row-major CPU reference for `clamp(sum((activation - 128) * weight), 0, 255)`.
The static gate verifies the integer-HMX instructions, the HVX VTCM copy, and
the absence of QNN/QAIRT dynamic dependencies:

```sh
scripts/check_exp0002_static.sh
```

Formal device evidence and binaries are archived outside Git under the project
storage contract:

```sh
scripts/collect_exp0002_evidence.sh
```
