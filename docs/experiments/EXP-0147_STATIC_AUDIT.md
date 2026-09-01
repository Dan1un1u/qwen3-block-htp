# EXP-0147 static shape and KV-cache audit

## Scope

EXP-0147 characterizes the immutable selected kernels at prefill
`M={16,32,64,128}` and decode `M=1`, `past={64,256,1024,4096}`.  It does
not retune any selected `M=64` schedule.  The scan adds a logical-shape layer
around the existing physical projection kernels and a declared persistent
KV-cache DDR boundary.

## Physical row contract

The selected U8 projection instruction consumes a 64-row activation tile.
The selected FP16 instruction consumes a 32-row tile and the block schedules
issue two such tiles for `M=64`.  Consequently:

| logical cell | physical query work | useful row utilization |
|---|---:|---:|
| prefill M16 | one 64-row U8 tile / two 32-row FP16 tiles | 25% |
| prefill M32 | one 64-row U8 tile / two 32-row FP16 tiles | 50% |
| prefill M64 | selected path unchanged | 100% |
| prefill M128 | two selected M64 chunks in one RPC | 100% |
| decode M1 | one selected M64 physical query chunk | 1.5625% |

Padding rows are never included in correctness or useful-token throughput.
They remain visible in physical HMX counts and wall time.  Prefill M128 is
causally exact only when the second chunk attends the first chunk through the
persistent cache; two independent M64 invocations are not an admissible
substitute.

## KV-cache boundary

The cache stores post-Q/K-Norm+RoPE K and post-projection V, grouped by KV
head, with a capacity stride supplied in the RPC header.  It is a declared
Native Boundary in DDR.  Each run overwrites the append range starting at the
declared past length, so repeat-ten measures the same logical cell rather than
growing the cache ten times.

For every GQA group the dynamic Attention phase DMA-stages one K head and one
V head into VTCM, packs them into the existing HMX RHS carrier, and reuses the
same phase memory across groups.  QK score and probability carriers remain in
VTCM and are never exposed as graph/package outputs.  Cache DDR traffic is
counted separately from input/output boundary traffic and from forbidden
intermediate DDR traffic.

## Dynamic Attention VTCM upper bounds

Let `P=ceil(total_kv/32)*32`.  Per GQA group, with two Q heads and head
dimension 128, the largest simultaneous phase is V staging + V HMX carrier +
probability:

* U8: `P*128 + P*128 + 2*64*P` bytes = `384*P` bytes;
* FP16: `2*P*128 + 2*P*128 + 2*2*64*P` bytes = `768*P` bytes.

At the largest requested cache (`total_kv=4097`, padded to 4128), these are
1,585,152 bytes for U8 and 3,170,304 bytes for FP16.  They fit inside the
phase-dead QKV/Attention/MLP carrier interval already reserved by each selected
8 MiB plan; no second VTCM allocation and no DDR workspace are required.  The
implementation must assert the actual pointer interval at runtime before using
the overlay.

## Source assumptions that must not leak into the logical contract

The following compile-time assumptions were identified and are intentionally
kept as physical-kernel constants rather than rewritten globally:

* `QBH_BLOCK_M=64` controls selected projection, RMSNorm, residual, SwiGLU,
  and tile-carrier loops;
* `QBH_ATTENTION_M=64` and `QBH_ATTENTION_SCORE_TILES=2` control the selected
  M64 Attention fast path;
* current causal Softmax uses `valid=row+1` and a fixed 64-column carrier;
* Q/K Norm and RoPE use one 64-position table resident in VTCM;
* host input, reference, output and audit files are fixed to 64 physical rows;
* current RPC has no persistent K/V offsets or cache telemetry.

EXP-0147 therefore adds a separate generalized Attention/cache path.  The
selected `(logical_m=64,past=0)` path remains byte-for-byte the original
EXP-0109/0140/0144 path.  Smaller shapes use physical padding; M128 uses two
M64 chunks and reloads the second RoPE slice; decode uses one padded physical
chunk at the declared absolute RoPE position.

## Hard failure checks

The implementation must reject: logical M outside `{1,16,32,64,128}`; a
decode request with M other than one; total KV beyond 4097; a cache range that
does not cover all eight KV heads; an Attention overlay larger than its actual
phase interval; any nonzero intermediate DDR/spill/fill counter; any VTCM
request/acquisition other than exactly 8 MiB; or any path that changes the
selected M64 output.
