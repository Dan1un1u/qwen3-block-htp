# EXP-0240 results and handoff

Completed 2026-09-07 under PC055. Evidence valid; local speed gate FAIL. Adoption pending, no baseline promotion. Full-model execution was stopped before starting because both complete single-layer Host wall regressions exceed the user-selected 10% threshold. Other recipes remain frozen. Next action is discussion with the user, not another experiment.

## Scope and result

Fresh original Qwen3-1.7B layer14, all seven projections, AIMET LPBQ compressed4/decompressed8, K group32, one FP32 base scale per output channel. Compressed W4 and packed integer group multipliers are transferred within timed projections, reconstructed to S8 in VTCM by HVX, then consumed by U8xS8 HMX. Control is existing per-channel direct-W4 W4U8. Five short rounds and ten alternating formal pairs completed for both repeat1 and repeat10. M64 prefill is followed by eight M1 decode positions with self-computed persistent KV; each repeated replay resets state. No outlier deletion or extrapolated token throughput.

| Complete Host wall, repeat10 | Per-channel us | LPBQ32 us | Paired regression | Regression 95% CI |
|---|---:|---:|---:|---:|
| M64 prefill | 1676.1277 | 4220.43755 | +151.7675% | +149.5059% to +157.8751% |
| M1 decode, mean of 8 positions | 1020.72104375 | 3692.571275 | +262.3722% | +256.0509% to +268.7582% |

Latencies are medians of ten round means; regressions use medians of paired round ratios, with 50,000 bootstrap resamples, seed240. Repeat1 independently shows +145.85%/+260.48%. Same-format scheduling diagnostic (five unpaired repeat10 rounds) is 1743.4218/1169.12425 us; this is not the primary paired comparison. O and MLP dominate the observed added cost. This result measures the current project-owned unpack/scale/schedule implementation, not the theoretical limit of LPBQ or Qualcomm QNN kernel performance.

## Verification and physical evidence

Original checkpoint and frozen control hashes verified. Official AIMET grouping/scale oracle pinned at761ef454c39b51cd127d3104ca1d975116bedb61. All seven fresh packed weights and independently derived bias tables pass. The independent integer accumulation plus SDK native HMX conversion comparison passes63 projection/step checks,1,474,560 values,0 mismatches,0 LSB error. Separate int64 reductions cross-check integer products. SIMD/scalar complete layer outputs and prefill KV are exact; unit-multiplier LPBQ/direct-W4 outputs and prefill KV are exact. All final short/formal output hashes match frozen references. No PPL/quality measurement was performed.

Requested/acquired VTCM8,388,608 bytes; peak plan6,682,752 bytes. Zero intermediate DDR reads/writes and zero spill/fill. One FastRPC block invocation per step, no QNN. Actual weight DDR including bias25,329,664 bytes control and26,116,096 bytes LPBQ. Multiplier metadata786,432 bytes,3.125% of W4 code storage. Expanded S8 stays in VTCM. Every formal additive DSP ledger closes exactly, with zero unattributed ticks.

Initial projection audit failed due to audit-output cache publication, repaired before final exact checks. LPBQ batch telemetry was subsequently corrected and all short/formal collections repeated with prefix final_. Earlier attempts are retained as diagnostics and excluded from primary timing. Historical comparison fields in the single-layer runner are inapplicable; the external independent audit is the correctness authority.

## Reproduction and continuation

Formal runtime source633e9da1b77ee3336a334013e10329b6baf34c74; reporting sourcee02d01c0d40603bb80fcc9e292d106a7d72be0b9 on codex/exp-0240-w4a8-lpbq32-layer-gate. Exact host/DSP artifacts and hashes are retained in final_formal_provenance.json and artifact_manifest.json, with immutable copies under /mnt/d/llm_exp/models/qwen3-block-htp/exp0240/artifacts/633e9da1b77e. Host binary provenance is recorded separately from the DSP telemetry build.

Results: /mnt/d/llm_exp/results/qwen3-block-htp/exp0240. Models: /mnt/d/llm_exp/models/qwen3-block-htp/exp0240. Evidence ledger covers709 files, including failed attempts, frozen references, provenance, commands, all timing rounds, independent report check and source patch. Ledger SHA256:25a99c604a6d378d1162b3c8cefe607bbe2da81e587957467f8c6a2299eca69b. All709 file sizes/hashes reverified at closure.

Current ignored build caches android_ReleaseG_aarch64 and hexagon_ReleaseG_toolv19_v79 have QBH_EXP0240_SINGLE_LAYER=ON. Before any future approved full-model build, explicitly turn it OFF or use a separate build directory. Do not mistake these binaries for full-model artifacts. No full-model test is authorized to continue after this failed speed gate without further user direction.

[Complete profiling report](EXP-0240-PROFILE.md) includes repeat1/repeat10, complete module/counter ledgers, provenance, physical evidence, correctness and the frozen-recipe N/A overview.

E2E token/s: N/A; full model not run after the single-layer gate failed.
