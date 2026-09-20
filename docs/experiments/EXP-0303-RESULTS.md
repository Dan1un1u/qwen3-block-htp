# EXP-0303: Qwen3-0.6B A16 long-prefill completion

Scope: W16A16 and W4A16 B536+46/C741+3 prefill. Native A64+42 and long64+3 are unchanged mathematical companion references. All other model/recipe table rows and curated D preserved. Fixed token fixtures, not named-dataset quality evaluation.

## Implementation and selection

First remove the prior experiment-scope exclusion of Qwen0.6 from the existing generic row-parallel FP16 softmax and bounded KV look-ahead DMA. Main plus three existing HVX workers own disjoint rows. Per-context scratch is3328bytes in phase-dead expanded-weight VTCM, disjoint from the reusable KV slot after19968bytes. Cache prefetch still requires its capacity guard. No new pool, matrix-owner, cache representation or intermediate DDR. Option4 is the retained original control, option7 is the initial migrated candidate. Native64 and decode arithmetic/scheduling are unchanged.

The second engineering candidate (option15) uses the same row arithmetic, applies causal masks only on partial tail vectors, and clears padding with row ownership rather than serially clearing all live output bytes. All FP32 reductions, vendor vector exp, FP16 exponential rounding and normalization are retained. It is a generic vector-loop change, not a model-specific tile/kernel redesign. Its exact long741 audits and all engineering timings are retained regardless of final selection. Option15 was about6-7% slower in the bounded same-binary engineering check and was removed from final source. Selected option: 7.

## Formal performance

Five short rotated paired repeat1 rounds, then ten formal rotated paired repeat10 rounds, all samples retained. The unchanged native M64 and candidate long64 reference are both measured; the fastest eligible same-round prefill reference defines the10%gate. Decode comparisons across contexts are descriptive; same-length candidate/control decode is the regression guard. No reference slowdown or numerical tolerance change.

| Recipe | Shape | Control prefill TPS | Candidate prefill TPS | Prefill gain | Host wall ms | Decode TPS | Long/reference prefill ratio | lower95% | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| w16 | 536+46 | 1108.181028 | 1579.675930 | 42.5467% | 339.310101 | 25.023771 | 0.934378 | 0.932634 | True |
| w16 | 741+3 | 1007.051345 | 1559.401327 | 54.8482% | 475.182358 | 24.056729 | 0.922385 | 0.921169 | True |
| w4 | 536+46 | 1085.791597 | 1538.489490 | 41.6929% | 348.393670 | 24.748657 | 0.883702 | 0.882322 | False |
| w4 | 741+3 | 983.670592 | 1510.851018 | 53.5932% | 490.452064 | 24.212819 | 0.867827 | 0.867110 | False |

| Recipe | A64+42 prefill TPS | A decode TPS | Long64+3 prefill TPS | Selected reference |
|---|---:|---:|---:|---|
| w16 | 1690.618031 | 27.353994 | 1660.969430 | A-o4 |
| w4 | 1740.959628 | 27.133946 | 1711.676197 | A-o4 |

## Gates, accounting and correctness

Paired-round bootstrap20000 samples seed303416. Long/reference prefill TPS lower95%>=0.90; same-length prefill/decode wall upper95%<=1.10. Repeat1 is auxiliary only. All formal timed head tokens/logit codes, HMX work and physical additive ledgers validated. Reported Host wall includes embedding, every block, final norm, LM head/greedy, FastRPC and long-input/RoPE staging. Cold load, external tokenization and untimed audits excluded. Decode excludes the first prefill-produced token.

All four prefill gates pass: False. Same-length guards pass: True. Formal RPCs: 37400.
Final selected audits: 16, 374 full-model RPCs, 8498 capture files byte-exact to immutable parent. Native64, long64/65/128/129/536/741 and poisoned future KV included for both recipes. Independent actual-operand Norm/head and original softmax component checks retained. A8/SP2 native/long smoke checks pass on the shared binary. Two poisoned741 audits reuse the initial selected implementation: all four rebuilt final binaries are SHA256-identical, verified in selected-build-reproduction.json. Raw command/source identities remain intact.

W16 FP32 residual retains original full-M64 decode Norm/residual workload; W4 retains FP16 residual. Neither recipe receives new valid-row decode specialization. Historical full-floating-reference failures remain failed. No PPL/model-quality acceptance or automatic selected-baseline promotion. Any remaining throughput gate miss is reported without forcing acceptance.

## Provenance

Measured source d8e0576d61133090a1cbc25add11698d2e80f2ae on codex/exp-0303-qwen06-a16-prefill. Initial parent43c075fd881236df9b5d1b0a641ca3bcd485703c; inherited EXP0299/0301 packages/references checked read-only against authority-bound manifests/ledgers. Device binaries and consumed model payload hashes checked. Frozen reports, commands, all rounds, runtime seals and failed engineering attempts are retained in exp0303; complete additive module tables in MODULES.md.

Only six measured Qwen0.6 A16 ABC rows (four targets and two A references) may be refreshed in the workbook; unrelated data and D stay unchanged.

## Prefill attention attribution

Milliseconds summed across all prefill chunks. These attribution counters are not a replacement for the additive Host-wall module ledger.

| Recipe | Shape | Attention control ms | Attention candidate ms | Softmax control ms | Softmax candidate ms | Exposed cache DMA control ms | Exposed cache DMA candidate ms |
|---|---|---:|---:|---:|---:|---:|---:|
| w16 | B | 226.343280 | 81.909271 | 180.945096 | 44.038959 | 9.303563 | 0.673134 |
| w16 | C | 394.633931 | 134.303417 | 323.886130 | 75.857659 | 14.672593 | 1.162868 |
| w4 | B | 227.793944 | 83.335797 | 180.941256 | 44.140585 | 9.481292 | 0.759961 |
| w4 | C | 397.965662 | 137.080271 | 323.882262 | 76.097293 | 16.192044 | 1.331274 |
