# L32-0063: FP intermediate islands motivation diagnostic

Complete Llama-3.2-1B-Instruct, 16 layers, M64 prefill plus one fixed-token decode.
No rotation; unchanged per-channel W4, native integer QK/AV, uniform INT16 Down,
FP32 residual and RMSNorm. Package: L32-0062 control (unfolded AV-to-O); all
497 manifest files match locally and on device. Baselines and workbook unchanged.

## Result

The operator-boundary diagnostic measures FP SwiGLU with fused QDQ at
17.046 ms
(35.42% of complete Host wall),
and FP Softmax with fused QDQ at
3.154 ms
(6.55%).
Together: 41.97%. FP32 norm/residual are already present in
the retained runtime and are reported separately, not counted as newly restored islands.

This supports the narrower motivation that fast low-bit matrix products alone
leave substantial floating intermediate-processing cost in this configuration.
It does NOT establish a separate QDQ percentage, a universal floating penalty,
a fully integer residual/Norm implementation, or a measured all-integer speedup.
This is one model/shape. Qwen1.7 companion is not measured by this record.

## Implementation and scheduling

FP Softmax reuses vector QHL exponential/native U8 output. New FP32 SwiGLU uses
32-lane HVX, bounded exp range reduction/polynomial, Newton reciprocal, fused
input dequantization and ties-to-even INT16 output quantization, directly writing
the same two byte planes. No per-element scalar exp, CPU fallback, redundant
standalone conversion, or full floating DDR intermediate. Low/high Down matrix
passes, output scaling, remaining weights/parameters and cache are unchanged.
FP Softmax consumes raw integer scores with the existing score scale and emits
the original probability codes; low-bit nonlinear approximation is intentionally
replaced, so equivalence to that different mathematical function is not required.

Two schedules are retained. Production-style scheduling keeps worker overlap;
its formal M64 Host wall is 41.816 ms
(1530.52 token/s). Its summed Softmax worker timer exceeds its attention wall;
that sum cannot be used as an additive percentage.

For an additive diagnostic, QK -> Softmax -> AV are separated in batches of the
existing four HVX contexts; Gate/Up -> SwiGLU also has an operator boundary.
Within operators, vector workers, native matrix kernels and DMA double buffering
remain enabled. Softmax phase dispatch/join is included in its phase time. No
HMX work runs inside that isolated phase. Exclusive diagnostic M64 Host wall:
48.130 ms (1329.72 token/s). The plot MUST identify this
operator-boundary schedule; its fractions are not production critical-path shares.
The two schedules were not a paired speedup campaign and must not be presented
as such. QDQ is fused within these vector operators; splitting it would change
implementation/register traffic or introduce intrusive timing. Combine it with
nonlinear arithmetic in the proposed motivation plot.

## Correctness and physical evidence

- All 16 layers x 65,536 Gate/Up code pairs: 1,048,576 cases exactly match an
  independently implemented FP32 arithmetic/quantization reference. Difference
  from the double-precision mathematical formula is at most one INT16 code.
- Independent full transformer reference starts from original verified embedding
  and fixed token IDs, including integer matrices, floating intermediates,
  KV state, FP32 residual and final norm/head. Device audits at 1, 3 and 16
  consecutive layers pass exact layer output hashes and selected token/logit
  codes; full16 operator-boundary schedule passes the same reference.
- Five short checks then ten formal repeat10 rounds per retained schedule.
  Each formal schedule contains 100 prefill passes and 100 subsequent decode
  passes. One-step decode is a correctness/diagnostic scope, not a long decode
  benchmark. No PPL or readable-text/model-quality acceptance is claimed.
- 8,388,608 B VTCM requested/granted; peak 8,098,272 B; zero timed
  intermediate DDR/spill, DSP numerical/status success, exact additive ledgers.
- A SHA256 verification batch could overlap the initial phase-timing batch;
  all original files are retained and excluded from the final statistics.
  A fixed replacement five-short/ten-formal batch ran after hash checking ended.
  Use isolated-phases-* only for phase timing (timing-exclusion.json).
- Sources: pipelined 95e0ee3a630bdb87032cd22b2bf4973286ac03ce;
  phase diagnostic 2ac474ac485d16e33bb913493c3c4aadbf00321e.
  Sealed 1/3/16-layer binaries, input/output audits and per-run commands retained.

## Exclusive diagnostic prefill ledger

| Component | ms | Host wall share |
|---|---:|---:|
| I/O、metadata | 0.117 | 0.24% |
| Input RMSNorm | 1.310 | 2.72% |
| QKV＋RoPE | 3.854 | 8.01% |
| O projection | 1.587 | 3.30% |
| Post-attention residual＋RMSNorm | 1.280 | 2.66% |
| Down | 3.149 | 6.54% |
| Final residual | 0.002 | 0.00% |
| KV carrier conversion | 0.028 | 0.06% |
| KV append DMA | 0.112 | 0.23% |
| Block orchestration | 0.024 | 0.05% |
| Layer bookkeeping | 0.013 | 0.03% |
| Stage-boundary bookkeeping | 0.006 | 0.01% |
| DSP unattributed | 0.000 | 0.00% |
| Runtime setup/teardown | 0.089 | 0.18% |
| Embedding | 0.057 | 0.12% |
| Final model RMSNorm | 0.058 | 0.12% |
| LM head＋greedy（不含 final norm） | 3.327 | 6.91% |
| Host-DSP boundary | 1.515 | 3.15% |
| Gate/Up matrix pipeline | 5.016 | 10.42% |
| QK/AV and attention preparation | 6.385 | 13.27% |
| FP SwiGLU + QDQ | 17.046 | 35.42% |
| FP Softmax + QDQ | 3.154 | 6.55% |
| Total | 48.130 | 100% |

summary.json includes bootstrap intervals across ten round means (seed 63,
10,000 resamples), auxiliary single-decode data and overlapping work timers.
figure-b-llama1b.csv contains the additive plot data. MODULES.md retains both
full ledgers. Device profiling does not include loading/tokenization/ADB setup.
The retained float nonlinear path is opt-in QBH_FP_ISLANDS; no baseline promotion.
