# EXP-0302: Qwen3-1.7B C decode optimization

Only Qwen3-1.7B W4A8 and W4A8-SP2 C741+3 were optimized. Native A64+42 is an unchanged same-round reference. Other models, A16, B and D remain untouched. Fixed token fixtures, not named-dataset quality evaluation.

## Implementation

Original long option65439 remains the same-binary control. Candidate196511 adds131072: decode-only exact signed-byte centering and a 16-row staged K transpose. For x,z in [0,255], sat_s8((int8)(x xor128)-(z-128)) equals sat_s8(x-z). All65536 combinations checked independently. Full16-row blocks avoid per-row tail branches; only the tail uses bounded masks. Final shuffle/reduction is unrolled with separate sum chains. Existing pre-QK output plane supplies temporary storage, with no extra allocation or cache representation change. Native W4, SP2, all quantizers, softmax, FP32 residual, weights and HMX arithmetic stay fixed.

Candidate1 (staged16 alone,130975) reduced compiler stack frame3584 to40 bytes but did not show stable E2E gain. Candidate2 also simplifies centering and complete-tile control flow; frame384 bytes. Do not attribute its gain solely to register pressure. Failed/nonbeneficial engineering evidence is retained.

## Formal complete-model performance

Five short rotated paired rounds, then ten formal rotated paired repeat10 rounds, all samples retained. One device owner. Host wall includes embedding, every block, final norm/head/greedy and FastRPC, plus long frontend input/RoPE staging; excludes cold loading, tokenization and audit writes. Decode excludes the first token produced by prefill.

| Recipe | A prefill TPS | A decode TPS | C control prefill TPS | C candidate prefill TPS | C control decode TPS | C candidate decode TPS | C decode gain | C/A decode loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a8 | 1870.768731 | 47.933776 | 1881.396583 | 1880.904216 | 42.558523 | 44.432374 | 4.4030% | 7.3047% |
| sp2 | 1850.940985 | 47.708461 | 1861.826425 | 1861.814540 | 42.337074 | 44.285152 | 4.6014% | 7.1755% |

## Gates and confidence

Same-length candidate/control Host-wall ratio upper95%<=1.10 is a regression guard. C/A throughput lower95%>=0.90 is a distinct cross-context gate. Paired-round bootstrap20000 seed302302. Cross-context ratios compare the requested complete trajectories and different KV lengths; they are not an isolated causal estimate of prompt length.

| Recipe | Same-C prefill wall ratio (95% CI) | Same-C decode wall ratio (95% CI) | C/A prefill TPS ratio (95% CI) | C/A decode TPS ratio (95% CI) |
|---|---|---|---|---|
| a8 | 1.000262 [0.998902, 1.001442] | 0.957827 [0.955424, 0.960379] | 1.005418 [1.002547, 1.007667] | 0.926953 [0.923516, 0.930094] |
| sp2 | 1.000006 [0.998315, 1.001266] | 0.956011 [0.954604, 0.957762] | 1.005875 [1.005227, 1.006479] | 0.928245 [0.926462, 0.929919] |

Same-length guards: True. Cross-context prefill and decode gates: True.

## Attention attribution

Microseconds per decode RPC. K/softmax/HMX service totals sum concurrent workers and MUST NOT be added to wall time. Complete additive module accounting is in EXP-0302-MODULES.md.

| Recipe | Attention wall control | Attention wall candidate | K packing worker total control | K packing worker total candidate |
|---|---:|---:|---:|---:|
| a8 | 4739.223 | 3698.273 | 5993.339 | 1330.716 |
| sp2 | 4740.880 | 3701.021 | 5992.824 | 1331.754 |

Formal timed RPCs: 14600. All selected tokens/logit codes and physical telemetry checked. Full audits: 8 runs, 2548 exact layer outputs, 2155905024 exact valid KV elements, poisoned future unchanged. Candidate C audited both recipes; A8 additionally at65/128/129 and final control741. Candidate1 exact audits retained.

Exactly8MiB VTCM grant, bounded nonaliasing overlay, one HMX owner, no intermediate DDR spill. Paired HMX command counts, cache DMA counts/bytes and overlay sizes are identical. No PPL or model-quality acceptance, no relaxation of historical floating-reference failures, no automatic selected-baseline promotion.

## Provenance

Source branch codex/exp-0302-qwen17-c-decode; measured HEAD 43c075fd881236df9b5d1b0a641ca3bcd485703c. Runtime seal, commands, input verification, immutable parent references, audits, all raw samples and analysis scripts archived under exp0302. Source parent22fba3060abd606ec33efb7fabb942b33e14a20e. Inherited EXP0298/0301 model and reference identities verified before execution.

Only Qwen1.7 A8/SP2 measured A/C rows are eligible for workbook refresh; B, all other rows and curated D preserved. A references are freshly measured; compare older B measurements with their own recorded references rather than retroactively treating them as paired.
