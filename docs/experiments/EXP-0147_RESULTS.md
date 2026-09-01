# EXP-0147 multi-M and KV-cache characterization

## Scope and selected identities

EXP-0147 characterizes the selected F16F16 EXP-0109, W4F16 EXP-0140, and
W4U8 EXP-0144 kernels at prefill `M={16,32,64,128}` and decode `M=1` with
`past={64,256,1024,4096}`.  It does not retune the selected `M=64` kernels.
The scan adds logical tail handling, an explicit persistent head-major KV
cache in DDR, and a bounded VTCM-only generalized Attention path.

The W4F16 source identity was merged explicitly rather than approximated.  Its
selected runtime controls remain four expansion workers, one claimed region,
the extra Gate/Up expansion and stream workers, subgroup four, and
`join_only_down`.  At `M=64`, the selected kernel modules reproduce the
EXP-0140 timing contract: QKV is 414.0 us versus 414.6 us, Attention is
139.2 us versus 140.1 us, Gate/Up is 803.4 us versus 798.2 us, and Down is
302.3 us versus 298.3 us.  The scan's additional 2,293.7 us boundary interval
is persistent-cache carrier extraction, not a retune of those kernels.

## Formal wall-time scan

All values are repeat-ten host wall time per complete block.  A positive speed
number means the lower-bit recipe is faster; it is computed from reciprocal
latency rather than from a subtraction of wall times.

| Cell | F16F16 | W4F16 | W4U8 | W4F16 vs F16F16 speed | W4U8 vs W4F16 speed |
|---|---:|---:|---:|---:|---:|
| prefill M16 | 2,580.719 us | 2,543.708 us | 1,676.365 us | +1.45% | +51.74% |
| prefill M32 | 2,818.609 us | 3,116.391 us | 1,704.839 us | -9.56% | +82.80% |
| prefill M64 | 3,311.641 us | 4,254.787 us | 1,812.234 us | -22.17% | +134.78% |
| prefill M128 | 27,754.214 us | 29,397.318 us | 37,992.495 us | -5.59% | -22.62% |
| decode L64 | 3,349.026 us | 3,060.667 us | 12,420.609 us | +9.42% | -75.36% |
| decode L256 | 4,095.875 us | 3,744.031 us | 44,168.859 us | +9.40% | -91.52% |
| decode L1024 | 6,956.427 us | 6,581.630 us | 170,169.172 us | +5.69% | -96.13% |
| decode L4096 | 18,608.646 us | 18,590.276 us | 674,402.948 us | +0.10% | -97.24% |

Every one of the 24 formal cells requests and acquires exactly 8 MiB VTCM,
reports zero intermediate DDR read/write, zero spill/fill, one FastRPC
Execution Unit, one HMX owner, zero persistent-cache mismatch, and zero formal
output mismatch.  KV-cache traffic is explicit and separate: U8 reads and
writes exactly half the FP16 bytes.

## Correctness interpretation

Formal zero mismatch is a two-run device reproducibility gate and must not be
confused with an independent mathematical reference.  F16F16 passes the
independent FP16 tolerance, while W4U8 has its separate zero-LSB integer
Attention boundary proof described in `EXP-0147_W4U8_STAGE.md`.

For W4F16, all independently recomputed Attention outputs pass the existing
`max_abs <= 0.0625` and `cosine >= 0.99999` rule.  The selected M64 final
output also remains within 0.0625.  However, the independent complete-output
comparison reaches 0.125 at M128 and decode L256/L1024/L4096, and the prefill
V-cache comparison reaches 0.125.  These are one FP16 ULP at the affected
high-magnitude values and retain cosine at or above 0.999999, but they exceed
the immutable absolute threshold.  Therefore EXP-0147's evidence is valid and
the characterization is complete, while the strict aggregate local
correctness gate is not fully passed.  No new baseline is promoted.

## What the scan reveals

The optimized W4U8 path generalizes well only while it can use the selected
64-token Attention schedule.  At M16, M32, and M64 it is respectively 51.7%,
82.8%, and 134.8% faster than the W4F16 scan path.  The fixed projection tile
still wastes rows at M16/M32, but it is not the present critical path.

M128 crosses into generalized cache-backed Attention.  W4U8 Attention alone
takes 34,377.0 us, or 90.5% of the complete block, and W4U8 becomes 22.6%
slower than W4F16.  The projection work is merely two selected M64 chunks;
the cliff is caused by serial K/V staging and packing plus a non-streamed
longer Softmax/AV path.

Decode is more decisive.  The current W4U8 generalized Attention consumes
11,028.3 us at L64, 42,529.1 us at L256, 168,592.0 us at L1024, and
673,040.0 us at L4096.  Its share of wall time grows from 88.8% to 99.8%.
Halving KV-cache bytes therefore cannot produce a speedup: the saved DDR
traffic is overwhelmed by scalar/cache-carrier conversion, repeated complete
cache scans, synchronization, and a physical 64-row HMX query carrier with
only one useful decode row.

The FP16 scan exposes a second, independent carrier issue.  Head-major cache
export is a phase boundary absent from the old no-cache block benchmark.
F16F16 spends about 963.7 us in the M64 boundary interval, W4F16 spends
2,293.7 us, and W4U8 spends 350.0 us.  W4F16 is worst because normalized K is
held as an HMX RHS carrier and V as grouped HMX output; converting both into
logical head-major cache rows serializes work that the selected Attention
kernel previously consumed directly.

## Recommended transition

The next performance experiment should not retune the selected M64 projection
schedule and should not begin with quantization-fidelity work.  It should build
a cache-native generalized Attention substrate, starting with decode because
that path isolates the failure and already owns almost the complete wall time.
The useful comparison is an M1 HVX dot-product path versus a padded HMX query
path, both using a K/V cache layout that can be consumed without scalar
repacking and an online tile-wise Softmax that never materializes a full score
or probability tensor in DDR.

Once decode establishes a fast cache carrier, the same K/V tiling and online
normalization should be extended to M128 as two 64-row query blocks sharing
the cache stream.  Only after generalized Attention is no longer dominant is
it useful to specialize M1 projection tails.  Quantization-accuracy repair
remains a separate track because this experiment deliberately fixes recipe
mathematics and measures physical generalization.

## Evidence

Formal result roots:

* `/mnt/d/llm_exp/results/qwen3-block-htp/exp0147/f16f16`
* `/mnt/d/llm_exp/results/qwen3-block-htp/exp0147/w4f16`
* `/mnt/d/llm_exp/results/qwen3-block-htp/exp0147/w4u8`

Model and device-golden packages are under
`/mnt/d/llm_exp/models/qwen3-block-htp/exp0147`.  Independent W4F16 boundary
audits are stored per cell under the W4F16 result root.
