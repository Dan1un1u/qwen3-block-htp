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

## EXP-0003

The projection probe composes twelve integer-HMX tile pairs into an exact
`M=64, K=128, N=96` asymmetric-U8 by signed-S8 projection. Four activation K
tiles are staged once and remain resident in VTCM. Three packed weight bundles
flow through two VTCM slots, so the third bundle must reuse slot 0 only after
the HMX worker releases it. A FastRPC-thread User DMA producer and a dedicated
HMX consumer synchronize with explicit QuRT semaphores; QNN is not used.

Build and run the first detached diagnostic without adding a process-kill
timeout:

```sh
scripts/build_exp0003.sh
QBH_DETACH=1 scripts/run_exp0003.sh identity 1
scripts/poll_exp0003.sh identity 1
```

The result is checked against an independent row-major Host reference for
`clamp(sum_k((activation - 128) * weight), 0, 255)`. Stage counters and qtimer
measurements expose activation DMA, weight DMA, HMX computation, producer and
consumer waits, pipeline duration, and output assembly separately.

## EXP-0004

The real-shape projection probe scales the validated EXP-0003 physical
contract to the two Qwen3 middle-block MLP shapes at native HMX spatial width:
`M=64, K=2048, N=6144` for gate/up and `M=64, K=6144, N=2048` for down. Both
perform exactly 12,288 logical 64x32x32 HMX tile pairs. Activations and the
complete tiled output remain resident in a fixed 1 MiB VTCM allocation, while
packed S8 weight bundles flow through two User-DMA slots.

K depth is split into at most 32 architectural tiles per continuous
`activation.ub ... :deep:cm` stream. The accumulator is initialized once and
retained across two gate/up streams or six down streams before the final U8
store. This is a substrate correctness experiment: W4 unpacking and
per-channel scales are intentionally deferred.

Build, statically audit, and run the first detached diagnostic without a
process-kill timeout:

```sh
scripts/build_exp0004.sh
scripts/check_exp0004_static.sh
QBH_DETACH=1 scripts/run_exp0004.sh gate_up identity 1
scripts/poll_exp0004.sh gate_up identity 1
```

Both shapes and all four correctness patterns are checked byte-for-byte
against an independent host reference. Formal evidence uses repeat 1 for
correctness and repeat 10 for timing:

```sh
scripts/collect_exp0004_evidence.sh
```

## EXP-0005

The W4U8 projection substrate keeps the EXP-0004 real Qwen3 shapes and HMX
schedule, but adds a packed signed-W4 Native DDR Boundary. Two W4 nibbles are
stored per byte. User DMA stages each compressed output-channel bundle into
one of two VTCM slots, then an explicit HVX nibble LUT applies one exact
positive integer scale per output channel and writes the S8 carrier into one
of two HMX-consumption slots. QNN is not used.

`expanded_s8_control` stores that identical S8 carrier in DDR. It is the strict
A/B control for `packed_w4u8`: both variants request the same 2 MiB VTCM plan,
use the same activation and output residency, execute the same integer-HMX
schedule, and are checked against
`clamp(sum_k((activation - 128) * w4 * channel_scale), 0, 255)`.

Build and statically audit the DSP image, then use a detached first run:

```sh
scripts/build_exp0005.sh
scripts/check_exp0005_static.sh
QBH_DETACH=1 scripts/run_exp0005.sh packed_w4u8 gate_up identity 1
scripts/poll_exp0005.sh packed_w4u8 gate_up identity 1
```

Formal evidence covers both storage variants, both shapes, four patterns, and
repeat counts 1 and 10:

```sh
scripts/collect_exp0005_evidence.sh
```
