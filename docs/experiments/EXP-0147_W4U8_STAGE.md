# EXP-0147 W4U8 shape-scan stage

## Scope

This stage generalizes the selected EXP-0144 W4U8 block without retuning its
`M=64` kernels.  Prefill scans logical `M={16,32,64,128}` and decode scans
`M=1` with persistent KV-cache lengths `{64,256,1024,4096}`.  The cache is an
explicit DDR boundary; all block intermediates, QK scores, probabilities, and
carrier workspaces remain in the single 8 MiB VTCM allocation.

The selected `M=64,past=0` path is unchanged.  `M=16` and `M=32` execute one
physical 64-row tile and compare only valid rows.  `M=128` executes two
selected 64-row chunks in one RPC; the second chunk uses generalized causal
Attention over both chunks.  Decode executes one padded 64-row projection
chunk and generalized Attention over the declared cache plus the appended
token.

## Formal repeat-one and repeat-ten result

| Cell | repeat1 (us) | repeat10/block (us) | useful tok/s | Attention (us) | cache read/block | cache write/block |
|---|---:|---:|---:|---:|---:|---:|
| prefill M16 | 2,097.448 | 1,676.365 | 9,544.463 | 125.083 | 0 B | 32,768 B |
| prefill M32 | 2,109.739 | 1,704.839 | 18,770.106 | 124.271 | 0 B | 65,536 B |
| prefill M64 | 2,259.739 | 1,812.234 | 35,315.520 | 121.661 | 0 B | 131,072 B |
| prefill M128 | 38,520.572 | 37,992.495 | 3,369.086 | 34,377.005 | 262,144 B | 262,144 B |
| decode L64 | 13,062.916 | 12,420.609 | 80.511 | 11,028.323 | 133,120 B | 2,048 B |
| decode L256 | 44,496.041 | 44,168.859 | 22.640 | 42,529.109 | 526,336 B | 2,048 B |
| decode L1024 | 172,979.427 | 170,169.172 | 5.877 | 168,592.016 | 2,099,200 B | 2,048 B |
| decode L4096 | 676,445.520 | 674,402.948 | 1.483 | 673,040.000 | 8,390,656 B | 2,048 B |

Every formal cell reports exactly 8 MiB VTCM, zero intermediate DDR,
zero spill/fill, one RPC Execution Unit, one HMX owner, zero cache mismatch,
and zero formal output mismatch.  Cache byte counts close exactly as
`2 * 8 heads * total_tokens * 128 * sizeof(U8)` for reads and
`2 * 8 heads * logical_rows * 128 * sizeof(U8)` for appends.

## Correctness provenance

The unchanged `M=64` path remains byte-exact with EXP-0144.  Cache append and
reload transforms are independently verified byte-for-byte from captured
post-Norm/RoPE K and post-projection V boundaries.  Generalized integer
Attention is independently recomputed from the actual accepted Q/K/V boundary;
its AV output is zero LSB for the audited prefill and decode cells.  The
unchanged downstream projection/residual/MLP path is then covered by a
two-run reproducibility golden.  The diagnostic independent CPU full-block
reference differs by at most 2 U8 LSB because its HMX/FP accumulation order is
not bit-identical; it is not substituted for the zero-LSB boundary proof.

## Characterization conclusion

The scan separates two different limitations.  Small prefill and decode do
not need a new projection algorithm to be correct, but the fixed 64-row HMX
contract gives only 25%, 50%, and 1.5625% useful row utilization for M16, M32,
and M1.  Conversely, M128 and long-context decode fit the 8 MiB VTCM contract,
but the generalized cache path serially stages and repacks each K/V head into
the HMX carrier.  At L4096, Attention occupies 99.8% of block wall time.  This
is characterization code, not a performance candidate.

The W4U8 evidence therefore points to two distinct future specializations:
a true decode-tail projection/Attention contract for M1, and an HVX-vectorized
or cache-native long-context K/V carrier that removes repeated scalar packing.
It does not support retuning the already selected M64 projection schedule.

