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
