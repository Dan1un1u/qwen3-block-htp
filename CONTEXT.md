# Llama 3.2 context

Read this authority after PROJECT_CONTRACT.md and PROJECT_STATUS.yaml, then
experiments/index.yaml. Do not infer current state from chat or old source
status files. This is standalone HTP, not mllm. No mllm legacy work is required.

The user froze Qwen3 and approved two Llama development branches. Both keep
F16F16 EXP0218 and per-channel C64 W4F16 EXP0260 OPT2. W4U8 OFF and dense R3 OPT2
are the defaults respectively; dense R4 OPT6 remains optional on the rotation
branch. Explicit startup references are in recipes/, branch policy in config/.
Code/runtime build/weight/report identities and Qwen3 historical speed are
pinned together in baselines/qwen3-frozen/manifest.json.

Current native code still implements Qwen3 semantics. The model descriptors are
porting boundaries only, and tools/recipe.py prints plans without execution.
No Llama checkpoint, weights, calibration, device result, PPL or throughput yet.
Never pass a Qwen3 package/prefix through the Llama model selector. Research
scripts keep their old paths for historical provenance; they are not Llama
launchers. Model version/revision selection precedes a registered port experiment.

Qwen3 W4A8 ideal-R3 whole-layer failure remains failed. R4 latest speed fixture
uses freshly folded RTN Down without R4-specific calibration and did not answer
the recorded France-capital prompt. No quality acceptance is implied by adopting
these implementations as migration starting points. R3 is the speed-oriented
default; R4 needs new Llama-specific math/weights/quality evidence if pursued.

Historical complete warm M64+15 speeds (prefill/decode tok/s): F16 EXP0218
793.1375/9.1241; W4F16 OPT2 EXP0260 1040.2401/15.2963; W4U8 OFF EXP0259
1705.3177/48.3572; paired R3 EXP0259 1703.3588/47.8691; R3+R4 EXP0265
1581.0687/46.0619. F16 used ten independent RPC-repeat1 sessions; other entries
formal repeat10. Cross-campaign figures are historical nonpaired references.

Initial organization is owned by Qwen3 authority EXP0266. After its closure,
new Llama work registers L32-0001 and runs this authority's preflight on the
chosen owning worktree. Freeze data before calibration/selection/evaluation.

## Organization completed

Both branches/worktrees and independent authority passed closure checks. Read docs/SESSION_HANDOFF.md. Qwen3 source restored at freeze. No active experiment and no Llama model implementation yet. Do not repeat organization or Qwen3 profiling.
