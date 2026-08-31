# Frozen Public Baseline — EXP-0109

The user promoted EXP-0109 on 2026-08-31 as the Frozen Public Baseline.
`PROJECT_STATUS.yaml` remains the authoritative pointer; this document records
the runnable source, fair-comparison cells, fastest retained W4U8 cell and
formal evidence identity.

## Identity

- Execution Unit: real Qwen3-1.7B layer-14 complete middle block, `M=64`.
- Backend: standalone FastRPC/DSP implementation on SM8750 HTP V79; no QNN.
- Source branch: `codex/exp-0109-public-common-freeze-consolidation`.
- Source commit: `42e2a33012929ccbb05e8c2741124b6fd539552c`.
- Frozen fair tag: `baseline-fair-exp0109-public-freeze`.
- Fastest W4U8 tag: `baseline-w4u8-fastest-exp0109-public-freeze`.
- Formal evidence:
  `D:\llm_exp\results\qwen3-block-htp\exp0109\20260831T155519Z_42e2a3301292_formal`.
- Retained binaries:
  `D:\llm_exp\models\qwen3-block-htp\exp0109\artifacts\42e2a3301292\20260831T155519Z_formal`.

## Frozen plans

The fair matrix builds F16F16, W4F16 and W4U8 from the same source revision.
F16F16 uses the accepted EXP-0107 eight-output-tile Gate/Up interleaving while
preserving the prior arithmetic, Crouton carrier, one-HMX-owner and streaming
SwiGLU contracts. W4F16 and fair W4U8 retain their EXP-0106 runtime plans.
The separate fastest W4U8 cell retains Down batch4, four-row native Softmax
and residual carrier shuffles, and the quarter-tiled paired Q/K kernel.

EXP-0108 `qkv_norms` is not part of the frozen public plan. Its W4F16-local
benefit remains valid evidence, but its F16F16 Host regression makes it a
future recipe specialization rather than a common component.

## Formal gate result

All EXP-0109 gates passed. Every selected cell preserved its accepted output
hash with zero independent-audit mismatches and zero maximum LSB. The plan
requests exactly 8,388,608 VTCM bytes, records zero intermediate DDR and
spill/fill, uses one FastRPC invocation and one HMX owner, and contains no QNN
dependency.

The accepted F16F16 interleaving reduced Gate/Up by 8.36% at repeat one and
8.22% at repeat ten. Complete Host wall improved by 2.57% and 3.41%; paired
medians improved by 2.60% and 4.28%. This reproduces the causal EXP-0107
result without repeating identical W4 candidate/control cells.

| Frozen cell | repeat1 Host wall | repeat10 Host wall | Output hash |
|---|---:|---:|---|
| F16F16 fair | 2,761.042 us | 2,335.630 us | `704252c89780e695` |
| W4F16 fair | 2,665.261 us | 2,181.063 us | `f18b9abbe1487231` |
| W4U8 fair | 2,292.187 us | 1,956.099 us | `69f22eeb035e5ec5` |
| W4U8 fastest | 2,184.791 us | 1,749.349 us | `69f22eeb035e5ec5` |

The fastest W4U8 cell is 4.69% faster than fair W4U8 at repeat one and 10.57%
faster at repeat ten in this formal run. This is a hardware-execution and
local numerical-correctness result for one block; it is not a full-model
quantization-accuracy claim.

## Repeat-ten module wall-time

| Module | F16F16 fair | W4F16 fair | W4U8 fair | W4U8 fastest |
|---|---:|---:|---:|---:|
| I/O and metadata | 6.6 us | 7.6 us | 4.1 us | 4.1 us |
| Input RMSNorm | 17.2 us | 17.3 us | 19.0 us | 19.1 us |
| QKV + Q/K Norm/RoPE | 396.9 us | 437.4 us | 423.6 us | 410.2 us |
| QK-Softmax-AV | 139.2 us | 140.0 us | 195.9 us | 121.9 us |
| O projection | 198.0 us | 171.3 us | 170.8 us | 170.8 us |
| Post-attn residual + RMSNorm | 16.7 us | 16.8 us | 33.8 us | 22.9 us |
| Gate/Up + SwiGLU | 1,025.6 us | 969.9 us | 694.1 us | 686.9 us |
| Down projection | 457.6 us | 328.0 us | 318.9 us | 227.6 us |
| Final residual | 5.0 us | 5.0 us | 17.2 us | 6.7 us |
| Host/RPC and closure | 73.9 us | 83.0 us | 80.6 us | 81.3 us |
| Complete block Host wall | 2,335.6 us | 2,181.1 us | 1,956.1 us | 1,749.3 us |

## Governance consequence

The Public Common Layer is frozen at EXP-0109. New performance work branches
from this source and is recorded as a Recipe Specialization unless the user
explicitly approves a new public-layer contract. Fair three-recipe reporting
must continue to use the frozen cells; fastest-per-recipe reporting remains a
separate table and cannot silently redefine the fair baseline.
