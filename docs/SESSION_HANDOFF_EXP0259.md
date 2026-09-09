# EXP0259 completed handoff

Source codex/exp-0259-dense-r3-pipeline-optimization 4fad27dc0a50a6d541f1d23ecf15018ec1099df6. Active none, next260. No mllm work. Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0259; seal 2f891f9aa35e7fa5bece5180e8e3a81fe4fd1253dc3749565dd226b462541a07, 1231 files. Read EXP-0259-RESULTS.md and PROFILE.md. Runtime manifests and exact package/prefix hashes in ARTIFACT_PROVENANCE.json. Current staged full28 ABI121.

QBH_R3_OPT0 original,1 constant/vector/fusedK,2 same plus per-head streamed preparation over five workers. Original dense HMX arithmetic/sqrtf/FP16 rounding, native per-channel W4, wideNR64 and EXP0258 head lifetime fix unchanged. Static32KiB sign matrix, vector four-row layout moves, quantization-fused Koperand and safe dead scratch relocation. Worker time overlaps QKV and must not be added to Host wall. No butterfly, model fitting, calibration or mixed precision.

Layer0 same-input M64+8M1:108 full-file comparisons plus independent dense component pass. First four-arm5short10formal preserved; r1decode CIupper1.10959 was uncertain. Fixed extra10pair off/stream batch before any new timing, no tuning/selection, combined20 CIs all within1.10; total7920layer timedRPC. Then correct offlineEOS prefix three-layer54 full-file comparisons exact to sealedEXP0258. Full28 smoke token IDs exact; fresh5short10formal paired off/stream repeat1/10 yields5280RPC,147840layerledgers and5280selected token/code checks. Same8MiB, zero timed intermediateDDR/spill/audit, oneRPC per complete model pass.

Warm R3 repeat10: prefill64 1703.358835 token/s; decode15 47.869072;16-output generation loop 43.994826. Paired noR3: 1705.317688, 48.357248, 44.342794. Fullmodel speed eligibility True. Historical EXP0258 oldR3 loop25.461939 token/s is nonpaired; use new paired control for gate.

Exact implementation equivalence passed; idealFloat64 whole-layer R3 numerical gate remains failed. No new PPL or quality acceptance, no baseline promotion. All recipes/weights frozen. Next discuss quality alignment/longer context evidence; no independent new experiment authorized.
