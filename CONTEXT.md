# Completed L32-0072: four-model HVX R4 side comparison

R3 excluded by user due to Q/K confound. Llama1B retains L32-0070; Qwen0.6/1.7 and Llama3B freshly paired against ordinaryA8 at fixed64+42, five short/ten formal repeat10. Exact own-reference, padding, KV/finalNorm/hidden and8MiB/no spill checks pass. No quality/default promotion. Results appended as final M_R4蝶形消融 in rebuilt workbook; original12 tabs unchanged. Source 90353c4725ae8e53431944bd1cdb7dfbf457335d; ledger 6360edebb3e79524599a47275255b21bf857303849edaa44b2f6abe664b96139. Device released. Read docs/experiments/L32-0072-RESULTS.md.

# Completed L32-0068: matched Table A Down precision
Full-model fixed64+42, five short and ten formal alternating repeat10, same binary/trace/shared optimizations. Exact own-reference1/3/full-layer and padding checks plus8MiBVTCM/zero intermediateDDR passed. Main rebuilt workbook A six scoped rows updated across L32-0068/EXP0308; H/I/J/K appended paired module evidence. Other rows/tabs frozen. No quality or Selected promotion. Source 399cc083f6c586e91695edef2d64c4105e3c4da6; ledger 651f24415c1b96de3bf60d3a91d0101f680fac8a40fe2bf472c9ca6e1bc87fc2; device released. See docs/experiments/L32-0068-RESULTS.md.

# Active L32-0068: Table A matched Down precision pair
Read docs/experiments/L32-0068.md. Latest shared production path, exact independent arithmetic, full1B64+42 pair; serialized device.

# L32-0067 completed: residual boundary fusion
Llama1B full16 fixed64+42, uniform INT16 Down, FP32 residual/Norm, no rotation. F/N/R/NR same-binary five short/ten formal repeat10; exact numerical/physical and padding checks passed. Both splits retain native layouts, tile/row granularity and common buffering; do not describe as unfused whole-layer/DDR controls. Combined NR/F wall ratios: prefill1.009487, decode1.005097; use formal CI before asserting individual gains. H/I/J appended to desktop workbook; main A-G unchanged, no promotion/quality claim. Device released. Sourcea0d6d8f504d360009333c0d0b7c649e49ce69aa2. See docs/experiments/L32-0067-RESULTS.md.

# Active L32-0067: residual boundary fusion four-arm experiment
Read docs/experiments/L32-0067.md. User approves only Llama1B64+42 N/R/NR versus fused, exact arithmetic and fair vector controls. Device reserved; no other experiments.

# 2026-09-22 workbook archive and matched A refresh completed

User authorized existing matched results only; no new hardware run. Desktop HTP workbook saved and read back. A64+42 final rows use actual uniform INT16 Down EXP-0304, EXP-0307 I, L32-0065 M, L32-0066 folded. Qwen retains independent AV RQ. B/C speed numbers and historical SP2 identities unchanged, explicitly pending refresh; historical relative-A percentages still refer to their original paired A. Other A recipes and D/E/G unchanged. F uses fair valid-row comparisons and retains superseded L32-0062 data in its history section. H/I/J archive138 paired phase comparisons,89 configurations,3570 module rows. Optimized historical pipeline source is EXP-0282/L32-0038; do not regress to older campaign reports. No PPL acceptance, source change or model-artifact baseline mutation.

See docs/PAPER_ABLATION_WORKBOOK_20260922.md and .json for scope, provenance and workbook receipt. Evidence/authoring snapshot: /mnt/d/llm_exp/results/paper-ablation-workbook-20260922. Workbook SHA256 f5d823bbf70a0f39ec5313f765857fac39895c5e012ed449cbd3058772649237. No active experiment or device work. Historical contexts below remain historical.

# L32-0066 completed: Llama3B AV folding supplement
M64+42, uniform INT16 Down, FP32 residual, no rotation. Five short/ten formal paired repeat10. Zero AV clipping across18,235,392reference values; own1/3/28-layer exact checks. Prefill wall reduction1.0970%, decode0.3193%. Preserve rounding/output differences, no quality/baseline promotion. Device released. See docs/experiments/L32-0066-RESULTS.md.

# Active L32-0066: Llama3B AV folding replication
Read docs/experiments/L32-0066.md. User requests matched3B I/M supplement; numerical eligibility precedes timing. Device owned; prior failures preserved.

## 2026-09-22 completed paired integer/fusion ablations

L32-0065 and EXP-0307 complete; both sources clean/synced at report closure and device released. Fixed M64+42, uniform INT16 Down, no rotation, FP32 residual/Norm, production pipeline. F/I prefill wall reductions: Llama33.019%,Qwen33.462%; decode0.305%/0.833%. Llama fair valid-row AV fusion adds0.408%decode gain; no confirmed prefill gain. Qwen unconditional AV fold fails original saturation semantics and is not timed/promoted as a valid comparison. Full own-reference checks pass for all eligible arms. Do not claim universal no-RQ, whole-block no-floating-point, or PPL acceptance. E/workbook untouched. Combined report: /mnt/d/llm_exp/results/paper-integer-fusion-20260922/REPORT.md. No further hardware work queued; await user discussion. See latest experiment record below and PROJECT_STATUS for hashes/results.

# Active L32-0065: complete integer/fusion ablations
User authorizes continuous completion; read docs/experiments/L32-0065.md. Llama1B64+42 first, device owned. Prior SP2/INT16 shared-backend ablations retained; E untouched. Qwen companion not started. No quality/baseline promotion.

# L32-0064 completed: Llama3B FP-island motivation
28-layer M64+one decode own reference exact; five short/ten formal repeat10. FP Softmax+QDQ4.920879%, SwiGLU+QDQ35.000214%, combined39.921093% complete Host wall. Read docs/experiments/L32-0064-RESULTS.md and MODULES.md. Device released to Qwen0.6 companion. No promotion/quality claim; diagram a unchanged.

# Active L32-0064: Llama3B FP-island diagnostic
Read docs/experiments/L32-0064.md. Match L32-0063 M64 phase accounting; device reserved for Llama, Qwen companion waits. No speed/quality promotion.

# L32-0063 complete: FP intermediate motivation
Read docs/experiments/L32-0063-RESULTS.md and MODULES.md. Llama1B full16 M64 + one decode exact independent reference; vector FP Softmax/SwiGLU, unchanged INT16 Down and FP32 residual. Diagnostic operator-boundary shares: Softmax+QDQ6.553%, SwiGLU+QDQ35.417%, total41.970%; NOT pipelined critical-path fractions or QDQ-only overhead. Production-style run retained, overlapping timers not additive. Five short/ten formal repeat10 each; initial diagnostic batch excluded for possible concurrent read-only hash checking, fixed isolated replacement retained. No baseline/quality/workbook change. Source 14d7858900244f18ac475c27df2fd8fa134ef956, device released, active none,next64. Qwen companion not measured here.

# Active L32-0063: FP intermediate motivation profiling
Read docs/experiments/L32-0063.md. Preserve native matrix/runtime; vector floating intermediates; correctness and additive profiling, no speed/quality gate. Device reserved after cross-project owner check.

# L32-0062 complete: fullmodel AV-O folding, F archived
Llama1B64+42 uniform INT16 Down/FP32 residual/no rotation. Source5486a5f772641a6717fdc05cba9aae2d40c0837a, native src/include unchanged. Each arm matches own independent full16 reference688layer outputs/43heads,preKVexact; no AV saturation6,946,816samples. BUT not old-output-equivalent: layer11 norm142.5 vs142.4999847 causes143vs142code; finalrelativeL2.2605 and5/43greedy token changes. No PPL/quality claim or baseline promotion. Ten paired repeat10: prefill2319.56->2317.91TPS(no gain),decode44.6586->46.4437TPS(+3.997%). Decode control includes fullM64RQ invalid rows, not minimalrow comparison. Timed12900boundaries,8MiB/no intermediateDDR. Workbook F added and A-E preserved. Read docs/experiments/L32-0062-RESULTS.md. Device released; no active experiment. Discuss accumulator-side factor restoration before any new folding adoption.

# Historical registration L32-0062: fullmodel AV-to-O scale folding
Read docs/experiments/L32-0062.md. Llama1B64+42 INT16 Down FP32 residual, one paired configuration; append F only.

# L32-0061 complete: AV-to-O scale folding probe
One Llama1B layer0 M64+1, uniform INT16 Down, FP32 residual. Metadata-only candidate skips AV HVX RQ and folds multiplier9 into O scale/zero128. Hardware outputs exact to each own reference; actual AV saturation0/133120. Small FP32 reassociation differences between arms, no old-bitwise-equivalence claim. Five short/ten formal repeat10: Host wall -1.109% prefill (CI crosses1), -5.489% decode. Current1B short decode RQ visits M64, so gain partly removes redundant row work; not a row4-optimized control. No E2E/PPL/baseline promotion. Read docs/experiments/L32-0061-RESULTS.md. Source820254505b42f6da6cddf821ac0809fa831ebaae, native code unchanged. Device released.

# Active L32-0061: AV-to-O scale folding probe
User approves one bounded Llama1B M64/M1 configuration. Read docs/experiments/L32-0061.md. Regenerate only needed retired inputs; no baseline promotion or model-quality claim.

# D-drive artifact retirement (2026-09-21)
User authorizes deletion of derived weights/builds/captured tensors, retaining original models, reports/raw logs/reproduction parameters and existing hash ledgers. Cleanup completed; D free space 264.63 GiB, net reclaimed 249.54 GiB. All 43 original file SHA256 values match. Reports/logs and historical ledgers remain retained; referenced derived payloads are intentionally unavailable, not a new hash failure. See docs/D_DRIVE_CLEANUP_20260921.md and D:/llm_exp/results/maintenance-cleanup-20260921. No source/gate/baseline/result changes.

# E sheet single-layer Down latency added
L32-0060 E A23:J35 now contains single-layer average Down latency in microseconds: formal fullmodel module sums divided by16/28layers, decode per token. Prefill536 includes all nine M64 chunks per layer. Original E E2E and A-D preserved. Workbook SHA256 99c76a1a523ac1515e07ce1c87cfafe05ec4713514d5518466b3bfab5df419be. Receipt: docs/experiments/L32-0060-WORKBOOK-E-DOWN.json. No hardware rerun or independent single-layer benchmark claim.

# L32-0060 comparison archived in workbook E
Four full-model HMX/HVX comparisons added to desktop main hardware workbook E sheet, with original TPS, formula-driven wall ratios and measurement scope. Original A-D parts and existing style entries preserved. Workbook SHA256 80fabddb5f4698adc76b1678c59e3f877e92966c6672fee6366fb618b311e539. Receipt: docs/experiments/L32-0060-WORKBOOK-E.json. No hardware rerun or baseline change.

# Latest closure: L32-0060

Four Llama1B/3B short/long full-model configurations completed. All independent and paired hardware arithmetic checks pass. Five short and ten formal paired repeat10 rounds retained. HVX shift/add Down is slower in all measured E2E cases; native HMX remains default. No baseline or quality promotion. Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0060; ledger SHA256 db12f374fc3f09a3b0b306404cc2eaeb1dc9b8c72302d4f08697d39be9337fc2. Source heads: {"1B": "5ec72782c410fbbe85ec5d92e031c8fc8fd14797", "3B": "4806798f979834f6913cd97f09c401e476ad709d"}. Results: docs/experiments/L32-0060-RESULTS.md; additive modules: docs/experiments/L32-0060-MODULES.md. Device released; active experiment cleared.

# Active L32-0060: HVX SP2 full-model generalization
Read docs/experiments/L32-0060.md. Llama1B/3B short and long; prior0059 remains component-only. Hardware ownership serialized, device reserved.

# L32-0059 complete: HVX SP2 shift-add Down
User selected full Down component only, not E2E. Llama1B8192->2048 layer7; four HVX workers and native-W4 gather unpack/K-vector shifts, exact outputs. Five short/ten formal repeat10,600 validated samples. HVX/HMX complete component wall6.336x prefill and4.859x decode; core217.523x/23.307x. Independent component copies/preparation differ from in-model fused endpoints: do not call these fullmodel or production Down ledger timings. Retain HMX default; no quality or baseline promotion. Read docs/experiments/L32-0059-RESULTS.md and MODULES.md. Source 2b80267b2e086ffe01dfb180db096c284233a9fc; evidence ledger 06af2b075751c4eaec9902fc1516e44ae9fbc3c7f29696753e6a53cbf5fcf890. Device released.

# Active L32-0059: SP2 HVX shift/add comparison
Read docs/experiments/L32-0059.md. Device owned; full Down component on Llama1B, exact same SP2/W4 contract. Historical baselines immutable.

# L32-0058 completed: SP2 / uniform INT16 Down
Read docs/experiments/L32-0058-RESULTS.md and MODULES.md. Both arms use identical nativeW4 pipeline and FP32 residual; only Down LUT/scale changed. Own full CPU reference exact,5short/10formal repeat10. No model-quality or baseline promotion. Device released. Ledger cd1b2da3bf6d48055aff0d32a40c2ba613cd0a075e289b597b8a8b89b4ca37f9.

# Active L32-0058: Llama3B SP2/INT16 comparison
Read docs/experiments/L32-0058.md. Offline only until EXP0304 releases device; fixed741+3, unchanged archived3B runtime. No measurements yet.

# L32-0057 completed: head64 parallel long decode
Read docs/experiments/L32-0057-RESULTS.md and MODULES.md. Selected long option159, same-binary prior31. Exact frozen0054 full-layer/KV/head/padding checks,5short and10formal repeat10. All10percent confidence gates pass. Six Llama1B A8/SP2 ABC rows updated; A16,3B,Qwen,D unchanged. No quality claim or automatic baseline promotion. Source 0e796912fa562bd5ab9797b40fec894d18aba862; evidence ledger de8bfe8a33bfa5a4badbbddcacd5318299e2ca510a85e0d7222c2b8172286f6a. Device released; next58.

# L32-0057 active: Llama1B A8/SP2 long decode repair
Read docs/experiments/L32-0057.md. Preserve numerical/physical gates and frozen other recipes/models. Existing head64 long decode is sequential; test parallel admission and exact vector preparation. Device reserved for this experiment.

# L32-0056 completed: generic A16 long-context repair

Read docs/experiments/L32-0056-RESULTS.md and MODULES.md. Qwen1.7B and Llama1B/3B were processed sequentially, excluding Qwen0.6. Five short and ten formal repeat10 paired rounds with unchanged native/long64 references and same-length controls. Generic row-parallel softmax and cache-DMA overlap retain exact per-row arithmetic; selected A16 long flag3, original control0; all original A16/A8/SP2 capture fields exact. Extra final-hidden capture fields independently verified. No per-model specialization or baseline degradation. Confidence gates all pass: True. Source closure 68d9401265b8f77aa6de9d1d2135d5608b61e9db.

Desktop workbook: refresh only18 affected A16 ABC rows across these two experiments; preserve Qwen0.6, A8/SP2 and curated D. Historical floating-alignment failures remain failures; no PPL/quality claim or automatic baseline promotion. Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0056; ledger 0d94ed728ba1b6499c9a5a19ebac6b05f87ba5b77f89e5117c9704125fc55b2a. Device released; next 57.

# L32-0056 measurements complete; workbook save pending

All three models completed numerical validation, five short and ten formal repeat10 rounds. All long-prefill and equal-length decode confidence gates pass. Read docs/experiments/L32-0056-RESULTS.md and MODULES.md. Llama3B final round spans two ADB interruptions; all completed slow samples retained. User restored charge/cooling; only missing final C arms were added. Device released. No automatic baseline promotion or quality acceptance.

Desktop HTP hardware workbook remains unchanged: export hit EBUSY. User asked to close workbook. Prepared18 rows; do not confuse interim reports with a saved workbook update. Re-run workbook_bundle.py prepare, long-a16-update.mjs, review/render/readback, archive, seal, authority_finish.py close. Operation marker already ran once successfully for this edit; do not repeat on retry. Full scripts/checkpoint: /mnt/d/llm_exp/results/qwen3-block-htp/exp0300/PROGRESS.json. Keep active experiment until workbook/evidence closure.

# L32-0056 active: generic long-context speed repair
User excludes Qwen0.6B. Read docs/experiments/L32-0056.md; retain10% long/M64 throughput gate, unchanged math and basic A16 fairness. Device serialized Qwen1.7 then Llama1B/3B.

# L32-0055 completed: generic A16 ABC baselines

Read docs/experiments/L32-0055-RESULTS.md and MODULES.md. Two models x W16A16/W4A16 x A64+42/B536+46/C741+3 completed, five short and ten formal repeat10 paired rounds per model. Fixed fixtures, not named datasets. Basic vector and DMA implementation only; no new per-model specialization. Implementation/physical checks pass; historical teacher-alignment failures remain failures and no quality/PPL claim or automatic baseline promotion is made. Source closure 5a818c744e3aa63b9f5c6401248ec17983e6a86e.

Desktop workbook A/B/C now contains 48 formal rows across four models and four recipes; the 24 A16 rows were completed/refreshed, prior A8/SP2 and eight curated D rows preserved. Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0055; ledger 6465a9e0404d9b47af9641f141c6e17eaedc48afb06233f143ea953d823c9497. Device released, no active experiment; next 56. Parent weight payloads, residual policy and original full-M64 Qwen0.6 W16 decode scheduling preserved. A8/SP2 long/M64 throughput gate remains a separate result.

# L32-0054 ABC ordinary A8/SP2 paired measurements completed

Four-model ABC campaign completed with the separately governed companion experiment. This project contributes two models, three shapes and both recipes: 64+42, 536+46, 741+3. Five short and ten formal repeat10 rounds per model, same binary per model, rotated order; all samples retained. Numerical/physical checks pass; long/M64 lower95% >=0.90 gate remains separately reported, with small-model optimization deferred. No model-quality claim or automatic promotion.

Read docs/experiments/L32-0054-RESULTS.md. Source ab9d7c2d0bf1061d10a945757589fb6df0ef64fc. Desktop workbook A/B/C updated; all A16 rows and curated D history preserved. No active experiment or device owner. Only the long API whitelist changed to admit ordinary A8; restore original ordinary middle qparams/LUT, keep W4 and other boundaries frozen.

# L32-0053 long-context throughput loop completed
Source 04673543a6c847809bcdc16621d9db2d5aa5723d; workspace build 3B/28 layers. Candidate option 8095. Fullmodel six-shape checks,5short and10formal repeat10 cycles complete. Numerical/physical pass; unchanged long/M64 parity+decode guard: fail. Same-binary prior opt31 control, fastest measured64-token reference; no slowed baseline. Long decode now admits parallel GQA. Failed engineering evidence retained. See docs/experiments/L32-0053-RESULTS.md and docs/experiments/LONG_CONTEXT_LOOP_20260918.md. Historical accepted baselines and quality status unchanged. No active experiment/device owner. Desktop workbook C:/Users/35961/Desktop/HTP硬件实验总表.xlsx must preserve exact-length A/B/C results and prior rows in D.

# L32-0053 active: long-context throughput loop
Read docs/experiments/L32-0053.md. CPU work only until Qwen EXP0296 releases device. Frozen Llama0052 fixtures/math; no quality/baseline promotion.

# L32-0052 completed — migration implemented, parity separately reported
Read docs/experiments/L32-0052-RESULTS.md and IMPLEMENTATION.md. Source closure 0360df475b91bc2e33d43c3f01dde02e457e8a9e. Numerical/physical checks pass; inherited long/M64 parity `fail`. Five short/ten formal repeat10 complete, all samples retained. No tolerance relaxation, slowed control, named-dataset result, PPL acceptance, or automatic baseline promotion. Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0052; ledger 5cdc62d51f5aaa0788e0f1088bc50df2eeefd931532b6cfe8a216dd2131447a0. Device owner released; no active experiment. Build 3B/28 layers. Long frontend prompt<=768/decode<=63/cache832; one-row long-cache decode remains sequential dynamic attention and has further optimization room. Cross-model summary: docs/experiments/LONG_PREFILL_MODEL_MIGRATION.md. Historical selected paper baselines unchanged.

# L32-0052 checkpoint: formal timing complete, closure pending
Source0360df4, measured5cedece, 3B28 build. Final six-shape arithmetic checks pass (2520 layer outputs, all KV/head/padding). Ten formal repeat10 cycles complete: M64 1167.866209 TPS;536+46 1103.851729/16.284528;741+3 1106.973578/14.745304. Long/M64 parity FAILS (~0.945/0.948), matched decode guards pass. Read results/l32-0052/RESULTS.md and profiling_summary.json. Closure head scope guard preserves1B; all loadable sections equal measured binaries, debug metadata differs. No resampling/gate change. Device now belongs to Qwen EXP0295. closure_campaign.py waits for Qwen1.7 formal summary before staging closure and audit64. Do not duplicate device work. Experiment still active until evidence ledger/authority closure.

# L32-0052 migration checkpoint (active)
Source 5cedece, branch codex/llama32-no-rotation, 3B28 build. Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0052; read IMPLEMENTATION.md and formal-protocol.json. Final six-shape audits64/65/128/129/536+46/741+3 all exact (2520 layer outputs, all KV/head/padding); old scalar RoPE repair compiler FMA defect fixed with explicit separate rounding, failed evidence retained. Five short rounds done; ten rotated five-arm repeat10 campaign is running under campaign.py. Do not duplicate device work. Source/parameters frozen until campaign completes. Qwen EXP0295 is separately registered, CPU references complete, its campaign waits for this profiling_summary.json before acquiring device. Completion/promotion not yet recorded; short3B long/M64 parity is below1 and is not a pass claim.

# L32-0052 active: migrate long prefill to Llama3B
User authorizes remaining-model migration. Read docs/experiments/L32-0052.md. Device owned by Llama; Qwen migration follows separately.

# L32-0051 completed: long-prefill throughput parity passed
Read docs/experiments/L32-0051-RESULTS.md, MODULES.md and IMPLEMENTATION.md. Current source/build is Llama1B/16layers, e31a286fac5b43cfccd93706bcc21a00400920ce, archive binaries-a10, opt-in QBH_LONG_OPT=31. Four engineering rounds; no weight/calibration/math changes, no rotation, FP32residual/SP2 retained. Other models/recipes are not migrated.
Frozen ten rotated five-arm cycles x repeat10: M64 prefill2305.962150TPS; optimized536+46=2505.950463/37.713353prefill/decodeTPS, optimized741+3=2499.129407/34.297589. Matched original0050 controls536=638.318843/23.014299,741=518.944727/19.381251. Parity PASSES: long/M64 throughput536=1.086727[1.081447,1.091901],741=1.083769[1.078151,1.089085]. Both decode10%slowdown guards pass. All samples included; no formal resampling or M64 baseline degradation.
Final measured binary independent128,741poisoned,536+46 checks:1200/1200layer hashes bit-exact,595607552KV elements/head IDs/codes exact, future cache untouched. Earlier same-family64/65/129 checks preserved separately (one historical signed-zero-only65 difference).14,400timedRPC ledgers exact; peakVTCM8098272/8388608, maxparallel overlay2494464/3145728,no intermediateDDR/spill. No new PPL/text acceptance or automatic paper-baseline promotion.
Method: native four-query softmax vectors with cross-KV reductions; two-row K horizontal sums/Vregister LUT; four GQA HVX clients on disjoint overlay slots, serialized original HMX/DMA owner; pooled cachedQKRoPE; reused score carrier and shared immutableV LUT. Shared LUT MUST remain beyond all four live slots. Parallel attention subcounters are overlapping worker durations, not additive wall components.
Failed predicate128 zero-mask and sharedLUT/up-alias candidates retained; final repairs pass, no relaxed thresholds. Longfrontend remains bounded prompt<=768/decode<=63/cache832; fixed project-owned fixture, not named dataset. D counts decodeRPC steps; first selected token comes from prefill. Host input/RoPE staging included, coldstartup/audit/logging excluded.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0051; ledger559e75c2eaae9811bee974e40f98f65ba9cf42c3c8f1485ff5ebe2776f5cad83,9260files/4505934845logicalbytes hashed. Frozen model manifests2682entries/792unique payloads,11archives and565parentreference/control files reverified. Device idle, experiment owner released; next52. Historical short-paper baselines remain unchanged.

# L32-0050 completed: Llama 1B W4A8-SP2 chunked long prefill
Read docs/experiments/L32-0050-RESULTS.md and MODULES.md. Opt-in M64 chunks, persistent KV and absolute RoPE, valid tail rows; head only on final prompt chunk. Frozen SP2/FP32-residual weights and integer arithmetic; no rotation. Current workspace build is 1B /16 layers.
64/65/128/129/536/741 and 536+46 independent hardware checks pass. Across seven checks, 1631 layer outputs are bit-exact and one differs only as -0 versus +0; no nonzero tolerance relaxed. All-layer KV/head and future-cache/padding poison checks pass. Audit-only copies are excluded from speed runs.
Five auxiliary single-replay checks and ten formal repeat10 processes per shape: 536+46 =638.957990/23.035885 prefill/decode token/s; 741+3 =519.037582/19.387292. Host input/RoPE staging included; cold loading/session preparation, tokenizer, logging excluded. Decode counts are RPC steps; prefill also computes the first selected token. Fixed project-owned trajectory, not named datasets. All 7000 timed RPC ledgers reconcile; VTCM peak8098272/8388608, no timed intermediateDDR/spill.
741 attention consumes79.13%prefill/69.11%decode Host wall; long attention is supported but not deeply optimized. Other sizes/recipes are not migrated here. No PPL, quality claim, cross-shape slowdown gate or baseline promotion.
Failed second-chunk attempts retained. Fixes: rebind layer0 metadata each chunk, RoPE all live query rows, initialize MLP worker context without relying on attention-pool side effects. Partial physical M64 preserves logical rows.
Source/measure fad2d40fa909c2c42f251b04a1c1eed64ad2af1c; evidence /mnt/d/llm_exp/results/llama32-htp/l32-0050; ledger 385b0d52f667264895d2f1ac99f191c3ec5cc25f7f047240b551967742e808f9, 5827 files, 2502716962 bytes rehashed. Models and all eight archived binary sets separately reverified. Experiment/device owner released; next51. Prior short-baseline/history entries below retained.

# L32-0049 completed: generic3B W16A16 via resident segmented weights
Read docs/experiments/L32-0049-RESULTS.md and MODULES.md. Fresh original BF16->FP16, seven768MiB backbone shards, resident embedding/full FP16 head; single session, bounded delayed DSP mappings withinoneRPC/token. All weight loading precedes timing; map/unmap included. Generic HMX/HVX/basic DMA, no specialization search.
M64+42/cache128/full28, five short/ten formal repeat10: 409.770413/6.785250 prefill/decode token/s. Peak VTCM7852800/8388608, zero timed intermediateDDR/spill. Current workspace3B28.
Selected0/13/27+chain3 composition_v2 pass;20 segmented/monolithic files byte-exact. Untimed28 single-layer monolithic composition matches full-stack M64+1 final hidden bitwise and56 prefill KV tensors. Final guard-only build reproduces all full43 audit bytes. Independent row-major KV reconstruction reproduces all43 hidden/norm outputs and IDs/logitbits exactly. 1B W4A16 and3B A8/SP2 regressions pass.
Full43 mathematical alignment remains FAILED: maxNRMSE 0.004798236>.003; no thresholds relaxed. Prefill KV mincos 0.999990956 passes; no mixed violations. Implementation evidence is separate from mathematical alignment/PPL. No model-quality claim or automatic baseline promotion.
Fixed3B capacity issue and FP16 Gate/Up producer linear-vs-ring addressing mismatch; faileda01/a02 retained. No multisession/CPU inference fallback. Prior W16 deferral resolved for runtime support.
Source/measure dba32d5ebd0bf73175c6d93e09d606e490a14385; evidence/mnt/d/llm_exp/results/llama32-htp/l32-0049, ledger4647bca725189e7dffc9bcb5bc4b69cd5e3e4a0135e46ec9dc95108266d8d9fa, 6118files rehashed. Experiment/device owner released; next50. Prior entries below are historical, including obsolete restrictions.

# L32-0048 completed: >4GiB delayed mapping works on current V79
Read docs/experiments/L32-0048-RESULTS.md. Maximum6.73388671875GiB data allocated and fully touched, two768MiB DSP windows plus751.5MiB pinned. Fifteen independent large processes x20 sweeps,2200 mappings allpass; smallcontrol12 maps also pass. 33 distributed pages/buffer, bidirectionalDMA/epoch checks, host readback, retained window/pin checks, map/unmap/cleanup allpass. Addresses reused; bounded test, not exhaustive full-buffer validation or fullmodel inference.
Mean mapping+unmapping per7/7/8-buffer sweep:0.699967/0.601545/0.697083ms. Wholeprobe wall includes diagnostic loops, notE2E. 8MiBVTCM; noHMX. No recipe/kernel changes, weight export, PPL, quality claim or baseline promotion.
Mechanism FASTRPC_MAP_FD_DELAYED host registration + HAP_mmap2/HAP_munmap2 insideoneRPC. Doesnot remove32bitDSP/monolithicuint32ABI; next step requires segmented resident FP16 weights and buffer-ID+local-offset lookup, bounded KV/control mappings. No needmultisession atthisstage; integration remainsnextdiscussion.
Measured4a5fbbfda4dc14ea4b3937e711b01922bc868c59, closuree9995b7fb0e0a93193a20b39af6715e33c6ac2d8. Evidence/mnt/d/llm_exp/results/llama32-htp/l32-0048; ledger08d8312bdcd7ae1eb8ca1c73e3a5b06564d0f19ed985409061e815bab1847c89, 100files rehashed. Source/memory synchronization checked atclosure; activeexperiment/deviceowner released. Initialcontrol-a01 linkerfailure retained, fixedLD_LIBRARY_PATH; notdrivermappingfailure.
Prior headings below are history, including obsolete active-state notices.

# L32-0048 active: delayed mapping probe
Read docs/experiments/L32-0048.md. User approves independent mapping/correctness/cost probe; no full3B W16 implementation or multisession yet. Previous results frozen.

# L32-0047 completed: generic3B W4A16; missing recipes filled except user-deferred W16
Read docs/experiments/L32-0047-RESULTS.md and MODULES.md. M64+42/cache128/full28, five short/ten formal repeat10. 493.223307/7.929781 prefill/decodeTPS. No specialization tuning. FP16 residual/KV/nonlinear baseline, same0041 W4/head/scales; no SP2/rotations.
Selected0/13/27+chain3 floating gates, RoPE/W4 component checks, 1B bounded regression,3B A8/SP2 full audits pass. Native KV inherited one32-token tail bug fixed: native vs row-major all43 hidden/norm and IDs/logitbits exact.
Full mathematical alignment remains FAILED (max boundaryNRMSE 0.006557147>.003, some cache cosine<.99999). Do not claim full FP16 alignment or model-quality/PPL acceptance. Old failed runs retained. Timing has audit disabled.
Measuredd08ea545270fff12ee5bc08218c606a3760a2b91, source closure38b001a6127d7083d7d20f0b552ba23f20fde4ec; runtime R/f16-l28c, whereR=/mnt/d/llm_exp/results/llama32-htp/l32-0047. Current workspace rebuilt3B28 (latest host audit admission only); use frozen measured runtime for archived speed.
Evidence ledgerfa3e1db2638a6573c2aeb1addb3c6e6fb1597d602af0fb6062a7a1a98783eb50, 8104files rehashed. No active experiment/device owner; next48. No auto baseline promotion.
Ordinary3B A8 L32-0046=1206.676361/23.726358; matchedSP2control1186.505056/23.717208. Qwen0.6 ordinaryA8 EXP0294=3074.858385/93.465647. All preserve latest sharedA8 optimizations while only removingDownSP2.
3B W16A16 unsupported/user-deferred:5.25GiB FP16 backbone exceedsuint32 shared-offsetABI. No weight paging authorized. Named-dataset48-row table stillpending; completed shapes alone do not establish HellaSwag data.
Prior headings below are historical, including obsolete active-state notices.

# L32-0047 active: generic3B W4A16
Read docs/experiments/L32-0047.md. User requests generic vector/basic DMA overlap, no specialization tuning. Ordinary3B A8 completed (L32-0046), preserved0045 SP2 and all shared optimizations. 3B W16A16 remains deferred by explicit user choice. L32-0046 reports/ledger retained; no quality or baseline promotion.

# L32-0046 active: missing recipe completion
Read docs/experiments/L32-0046.md. Ordinary3B A8 only removes SP2, latest0045 optimizations preserved. Device after EXP0294 closure. W4A16 follows; W16A16 user-deferred for >4GiB ABI.

# L32-0045 completed: 3B no-performance-gate pipeline loop
Latest 3B no-rotation W4A8-SP2mode8, FP32 residual: 1184.718073 prefill / 23.640810 decode token/s, M64+42/cache128/full28.
Paired frozen0044 control 1136.492048/22.446407; gains +4.243411%/+5.321132%. Five short + ten formal AB/BA pairs repeat10; all retained, performance gates disabled by user. CI in SUMMARY/results, not an acceptance threshold.
Four iterations: QKV8, QKV32 with phase-dead Gate bias storage, Norm/QKV DMA lookahead, final Up/Down DMA lookahead. Retained combined D, same arithmetic/payloads,980 fewer HMX submissions and1960 fewer weight DMA descriptors pertoken. Tile pairs/weight/cache bytes unchanged; peak8360416/8388608; no intermediate DDR/spill.
Measured ff213089c6f5cfc07c620c9509b22a5e00486e80; source closure 9ddf524f70f057069c8c4d03cd5cbe1b2277f028; frozen control 4686765b98939a34fec218af950eeba573087c71. Read docs/experiments/L32-0045-RESULTS.md, MODULES.md, IMPLEMENTATION.md and latest_llama3b_w4a8_sp2_optimization.
Launch AV_REQUANT_ROWS=4/head32/SP2mode8/FP32residual/WIDE_SCORE8/no rotation/masks0 via execute_llama32_3b_nogate.py. Frozen measured runtime /mnt/d/llm_exp/results/llama32-htp/l32-0045/final-l28/runtime.json. Current workspace build restored to3B28; do not accidentally use1B regression binaries.
87 runs/23678 token boundaries; all candidate selected0/13/27, chain3/28, full64-step true-greedy exact; bounded1B layer7 exact. Additional 6108160 captured FP32 words bitwise equal. No quality/PPL claim or automatic paper-baseline promotion. M64+63 is untimed correctness only in0045.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0045; ledger ad93526362d76ecac38d0a379ce2033020d8dc2f6262d4ee33cf935b7e581bc5, 2026 files/2077215106 bytes verified. Prior0044 ledger all2137 files rehashed. No active experiment/device owner.
Further opportunities: persistent native KV storage/full-prefix preparation; not implemented. Do not repeat0044 rejected RMS/K-transpose probes. Prior entries below are history, including obsolete active/build-state notices.

# L32-0045 active: no-performance-gate 3B optimization
Read docs/experiments/L32-0045.md. Latest user removes performance stopping thresholds; correctness/physical/evidence unchanged. Frozen0044 control, original0041 payloads. Primary M64+42/cache128. Prior entries retained as history.

# L32-0044 completed: longer3B decode optimization
Latest3B W4A8-SP2 FP32 residual,no rotation:
- PrimaryM64+42/cache128/full28:1138.295039prefill/22.538183decode tps. Paired0043arithmeticcontrol1130.655081/21.633357;decode+4.182553%,prefill+.675711%.5short+10formalrepeat10,95%CI gatespass.
- SupplementM64+63:1132.026119/22.126467 vs1124.985320/21.210176;5pairsrepeat10,separatefromformal.
- ExternalQuant.npu Table12 64+42 decode28.04 remainsunmet(~19.6% lowerthroughput); matchlength only,notsame dataset/completequantization.
Read latest_llama3b_w4a8_sp2_optimization, docs/experiments/L32-0044-RESULTS.md / MODULES / IMPLEMENTATION.
IMPORTANT candidatefastflags: QBH_W4U8_DECODE_AV_REQUANT_ROWS=4 plus QBH_W4U8_DECODE_LM_HEAD_GROUP_TILES=32. Newexecutor maps QBH_3B_HEAD_TILES=32. Expectedtokenbuffer64/cache128; QBH_3B_DECODE_COUNT=42or63. Do not use oldhardcoded16record launcher forlongfixture.
Finalnative784e754,measuredfullbuild4686765,closure6f488b36293d85b2b7f22f70bc78e461e81f816c. Control488e5f7 isparent0043 arithmetic rebuiltonlyforboundedlong-eval support; old16stepexactconfirmed. VregisterLUT/exactdivision andhead32retained; Norm/Ktransposeattemptsrejected/restored. Same594weight/scale/bias/LUT/qparamfiles.
Selected0/13/27,chain3/full28exact;64-stepindependentFP32hidden/IDs/logitcodes andprefillKV exactforcontrolfixed/candidatefreegreedy. 1Bboundedlayer7exact.67validatedruns20222boundaries,8600primaryformal+6400supplement. Peak8360416/8388608,zerotimedintermediateDDR/spill;physicalHMXtilepairs/weightbytesunchanged,375fewerheadcommands/token.
Evidence/mnt/d/llm_exp/results/llama32-htp/l32-0044;ledger735c9aa7a68a4c9121e583c50f32446c64d41fc91bc804334dc4850589369582,2137files/4135058140bytesrehashed. Bothsource/memorysynced. Noactiveexperiment/deviceowner;noqualityclaim orautomaticpromotion. Currentbuilddirectory contains1Bregressionbinary: usefrozenopt2c-l28runtime or rebuild3B28 explicitly.
Further directionsneednewprotocol: persistentnativeKV toavoidremainingfull-prefixpreparation, overlap nextstage/layer weightpreparation. Do notrepeat completedRMS/scattertransposeprobes. Prior entries below retained ashistory.

# L32-0044 active
Further3B decode optimization, M64+42 primary andM64+63 supplementary/cache128. Read docs/experiments/L32-0044.md. Frozen0043control/0041weights; exact/physical gates unchanged. No quality claim.

# L32-0043 completed: research-grounded3B target achieved
Latest3B1139.982409prefill/22.234309decode token/s,M64+15/cache80/full28. Paired0042control1040.690614/21.200743; +9.540952/+4.875143percent.5short/10formalrepeat10,95%CI speedgatespass;target1100/22pass.
Read latest_llama3b_w4a8_sp2_optimization and docs/experiments/L32-0043-RESULTS.md / MODULES / RESEARCH.
Same0041weights/scales/goldens,SP2mode8,FP32residual,no rotation. IMPORTANT fastestrecipe adds QBH_W4U8_DECODE_AV_REQUANT_ROWS=4; originallauncherdefault64 misses thisgain.
NativeGQA3batchedprefill reduces3584submissions;co-packedlive3decodequeryrows removes10752paddingtilepairs/token;boundedAVrow4. Vectorclippingcountnoindependentspeedclaim. Source3Bguardsleave1Bmath/instructionsunchanged;1Bboundedlayer7exact.
Allselected0/13/27,chain3/full28FP32hidden+KVexact,fullfixedtrajectory andtruegreedyIDs/logitcodesexact. Peak8360416/8388608VTCM,zerointermediateDDR/spill,sameweightbytes.57validatedruns5528boundaries,3200formal.
Sourceclosureb6a0d8d2cb4cb205b7c773922950ea8e4893209c,measuredd69d8f9ca9103c1888b1426ab482e58f2ed60dda. Evidence/mnt/d/llm_exp/results/llama32-htp/l32-0043;ledger505c583c195a05fa5e4cffbf865088b5a1365e42dd51fe834191c8c0ef722127,751filesrehashed. Bothbranchesclean/synced. Noactiveexperiment/deviceowner; noPPL/generalqualityclaim orautomaticpromotion.
Newcompactionrestricted3Bdecode1,rowmajorKV,paddedKV<=128. Followupsneednewprotocol; no repeatofcompletedbatch/rowlayoutwork.

Prior checkpoints below retained as history.

# L32-0043 active
Research-grounded3B optimization loop; read docs/experiments/L32-0043.md. Target1100prefill/22decode tps, exact numerical/physical gates unchanged. Frozen0042control/0041payloads. Device serialized. Prior entries below historical.

# L32-0042 completed: optimized3B pipeline/layout
Latest3B W4A8-SP2 FP32 residual,no rotations: **1038.737970 prefill /21.152098 decode token/s**, M64+15/28layers/cache80. Five short and ten paired formal cycles,repeat10. Paired original512.024356/18.679376; wall ratios .49292928/.88309800,both95%CI upper<=1.10.
Read latest_llama3b_w4a8_sp2_optimization and docs/experiments/L32-0042-RESULTS.md / L32-0042-MODULES.md.
Direct head128 vector RoPE/native gather/store; native-KV whole-vector transpose; K exact integer sum; V gather/vdeal. Frozen0041 weights/scales/model oracles. All selected0/13/27,chain3/full28 hidden/KV exact; full16 generated IDs/logit codes exact.1B frozenlayer7 regression exact; compiled1B DSP differs only in diagnostic source-line immediate.8MiB peak8360416,zero intermediate DDR/spill,identical matrixwork/weightbytes.
Source closure 9624a5cdd05d11eff3c1275d099efdc9e4940a68,measured 76418bbbb504dc30608441b17f49a774c10b72a2; clean/pushed. Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0042; ledger 918471ea1aa5fd70f4a9e9f139578837a8115739606d1dd2b4d89158955575d0,531files rehashed. NoPPL/general quality claim or automatic promotion. Default1B unchanged;3B supports W4A8-SP2 only. Device released,no active experiment.
Further3B optimization/configuration needs a new approved protocol; do not repeat diagnosed small-memcpy/head64-splitting work. Prior checkpoints below retained as history.

# L32-0042 active: 3B pipeline/layout optimization

Read docs/experiments/L32-0042.md. Frozen0041 weights/oracles and control binaries; exact arithmetic/physical gates, paired full-model profiling. L32-0042 owns device after checking no concurrent work. No new PPL, calibration or memory campaign.

# L32-0041 completed: Llama 3.2 3B W4A8-SP2

Original-derived3B port completed. Read latest_llama3b_w4a8_sp2_port and docs/experiments/L32-0041-RESULTS.md / L32-0041-MODULES.md. Source closure c990326d949d56778cf5990d5b4b246f1a247710; measured binary source 6f911883fa98a426a5d8482bbaeceebc6a2d7d40. E2E M64+15,5short/10formal,repeat10: prefill 512.141778 / decode 18.638267 token/s. Selected0/13/27,chain3/full28 exact hidden+KV; full16 generated IDs/logit codes exact;1B bounded regression exact. FP32 residual,SP2mode8,no rotations,no spills,peak8360416/8388608VTCM. NoPPL/general quality claim or baseline promotion. Original1B fields/artifacts remain1B;3B does not yet support W16A16/W4A16. Memory optimization deferred. Device released; no active experiment.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0041; ledger afe4cce6a0f52c3a11ffa674fb198e156c6ab3720da8bef23dc35f157eea8cf0, 370 files. Fresh weights/package under /mnt/d/llm_exp/models/llama32-htp/l32-0041. Build with QBH_LLAMA_MODEL_SIZE=3B and layer count28; default remains1B. Preserve failed shape/arena attempts; do not repeat diagnosed faults. The short greedy prompt answers Paris correctly, but fixed16-token benchmark continues after EOS and is not quality acceptance.

Prior checkpoints below are history.

# L32-0040 completed

Uniform INT16 Down vs A8/SP2 complete and sealed. Read latest_uniform_int16_down_result. Shared mode8 native kernel unchanged; full16 hidden/IDs/logit exact own reference. Fixed primary and greedy supplementary5short/10formal complete, all10percent gates pass, no quality claim. Qwen EXP0284 owns remaining device work; no Llama experiment active.

# Active L32-0040

Uniform INT16 Down experiment approved and registered. Read docs/experiments/L32-0040.md. Llama owns device first; Qwen local preparation. Float refresh complete, frozen. Latest A8/SP2 controls must retain shared optimizations. No quality claim.

# Current checkpoint L32-0039 complete

Both W16A16 fairness refreshes completed and sealed. Llama OPT3 974.174930 prefill /17.472296 decode token/s, M64+15, original964.478420/12.956326. Exact528 full-model files and256 layer hashes. Read latest_f16f16_fair_baseline_refresh and L32-0039 results. No active experiment/device owner; no PPL or Selected promotion. Next: discuss uniform INT16 Down control, no implementation yet. Prior checkpoints below are history.

# Current checkpoint L32-0039

Registered W16A16 fairness refresh under latest user approval. Read docs/experiments/L32-0039.md. Qwen owns device first, Llama local preparation only until release. Previous paper ablations sealed. New formal same-binary M64+15 control/candidate required.

Latest cross-model paper ablation synthesis: /mnt/d/llm_exp/results/paper-latest-ablations-20260915-a02/UPDATED_CONCLUSIONS.md; ledger a8b755b64da9691a1346af509d780f6bdca5961a533ebbc445a28bacab43033f. Generalization remains supplementary and was not rerun.

# Current checkpoint

L32-0038 completed and sealed. Latest optimized A0/A1/A2/A3/A5/A6/A9 rerun complete. All controls received applicable valid-row optimization. Read current latest_paper_ablation_campaign; historical pre-optimization findings below are superseded only by new paired measurements. Generalization is supplementary, excluded and NOT rerun. No active experiment; no quality/baseline promotion.

# Current checkpoint

L32-0038 registered for latest optimized paper ablation rerun. Read docs/experiments/L32-0038.md. Generalization excluded, not rerun. Qwen owns device first; Llama local preparation only until explicit release.

# Current checkpoint

L32-0037 completed and sealed; native decode optimization, Llama exact log2 optimization and bounded generalization finished. No active experiment. Latest working flags and results are in PROJECT_STATUS latest_decode_native_optimization, latest_exact_log2_optimization (Llama) and latest_bounded_generalization. No selected-baseline/quality promotion; no factorial rerun.

# Current checkpoint

L32-0036 completed and sealed. Latest working runtime native-row1 plus exact corrected-NR log2 mode8, no rotations. L32-0037 registered CPU preparation; Q EXP0281 owns device until release.

# Current checkpoint

L32-0035 completed and sealed: optimized native row1 retained. L32-0036 registered for exact log2 optimization; owns device next. Q EXP0281 only CPU preparation until release. No full factorial or PPL.

# Active L32-0035: decode optimization

Read docs/experiments/L32-0035.md. QwenEXP0280 owns device; Llama local preparation only. New user-approved sequence decode,log2,generalization. Old campaign sealed; no full factorial rerun.

# Both-model paper ablation campaign complete

All hardware work finished throughQwenEXP0279/LlamaL32-0034. No deviceowner. Consolidatedreport /mnt/d/llm_exp/results/paper-no-rotation-ablation-20260915/REPORT.md; campaignledger 5c050ce12226744cbe198c280bf6ff97a090e8581f969ad8176593e983b15447. Complete module-tablecompendium andPNG/SVG/PDFfigure. No remaining approved campaign experiment; noquality/baselinepromotion. Llamasourceunchangedat573c721 closure.

# Llama paper ablations complete through L32-0034

Source573c721b26fb3b370a4f8e6dea8b086613023776, measured5b54cccb9de4c415bbe4943228bd89a7298abdc7. Branchcodex/llama32-no-rotation clean/pushed; no Llama process active, QwenEXP0279 ownsdevice. ActiveNone,next35. L32-0034 ledger6bbd9ddfa2d750b9f55877fb2277bdb0ef90c3a2e0c5fa1acb9b1aaab1711da6,222files267230894bytes,10200timedprofiles; /mnt/d/llm_exp/results/llama32-htp/l32-0034. Nevermodifysealedresults.

M64+33cache128 independentall16-layer34-stepgreedyoracle exact, originalfirst16token/codeexact. FFNdisable/all wallprefill1.07442139CI[1.07230315,1.07648175],decode1.02165071[1.02046795,1.02282521]. ActualvalidKV97checkedallrounds,<=8MiB/no tensorDDRspill/identicalHMXweightwork. Originalwide0log2/SP2/FP32 retained, onlyhostgreedyrepeatlimitextension, noDSPweightchange.

A0/A1:0029;A2/A6:0030;A3:0031;A5:0032;A9:0033;A8:0034 allfinished, A7citeQwen272, excludedW4toS8/residualcomparisonsnotrun. TracecompleteN/A, QKVringmaskinactiveinLlama. Noquality/PPLacceptance orpromotion. Finalcombinedpaperreport pendingQwen279closure. SeeeachstageRESULTS fullmodule/CIs/E2E.

---
# Active L32-0034: approved longer-KV appendix

Read docs/experiments/L32-0034.md. Qwen EXP0278/vsum owns device; Llama local preparation only until release. L32-0033 complete and sealed. No promotion or PPL.

# L32-0033 complete; A8 remains

Source 0dfebbd6f72b145574c66d22d7dd13dfb3175883, measured 78560f4c12c63d2d1ac07405baca84c4ed275b8d; branch codex/llama32-no-rotation clean/pushed. All Llama A9 device work finished. Qwen EXP0278/vsum owns device: do not deploy/hash/profile Llama until release. Ledger c647b984f5cc875a366f6829589c66a6b5428961b867ff1159cc4f9813866904, 1497 files / 300912966 bytes under /mnt/d/llm_exp/results/llama32-htp/l32-0033; immutable.

FP vector softmax full fixed wall vs retained log2: prefill .94964867 CI[.94791243,.95164266], decode .98853313 [.98712262,.98995377]; no universal log2 advantage. Llama retains original per-layer division, selected7 EXACT; Qwen NR64 differs. Probability/AV approximation changes documented, all own selected/chain3/full16/frontend references pass, no quality acceptance/promotion. No vector stack spills, 8MiB, no tensor DDR spill.

Remaining approved A8: one already supported longer KV all-on vs representative FFN disable2, same SP2/FP32 no rotations. Register L32-0034 then preflight; CPU preparation allowed while Qwen owns device. Propose M64+33 (34 outputs, cache128), crossing KV96 tile boundary, unchanged original log2 wide0. Freeze fixtures and reference before timing. Numerical/physical gates unchanged, no runtime redesign for sweeping. A1/A2/A3/A5/A6 complete, A7 citeQwen272.

---
# L32-0032 closed; A9 remaining

Sourcef91f3a881c6eff7988218bfcf5b055a2297d2708,codex/llama32-no-rotation clean/pushed; measured0e406c088b3b188d40e213ed1e4ecffcb935292d. Active none,next33. No Llama processesactive. QwenEXP0278 ownsdeviceforA9component/fullintegration. Do notdeploy/profile/hashdeviceconcurrently.

A5allcomplete: ordinaryA8/SP2 fixedFP32residual1, originalweight/scales. Selected7/chain3/chain16independenttwo-step exact; ownfull16alllayergreedy/fixed CPUteachersbeforetiming,alltoken/logitcodesexact. 9600timedprofiles,5short10formalrepeat10eachtrajectory,repeat1aux. Fixed SP2/A8wallprefill1.01336440 CI[1.01015261,1.01638875],decode1.00035005[.99778268,1.00294924]; greedy1.01475488/.99923879. Decode no measurableextraSP2cost,extra tilepairs0;prefill+262144,weightbytesidentical. Samevector/readiness/FP32epiloguesprovidedboth. NoPPL/qualityacceptance orpromotion.

Evidence/mnt/d/llm_exp/results/llama32-htp/l32-0032, ledgera11ee12fd162617993b247e4b60074a815493205050517fa0643d8af0343f557, 583files/266094775bytes. Reports docs/experiments/L32-0032-RESULTS.md andsource docs/L32_0032_A8_SP2_COST_RESULTS. NevermodifysealedR. Build16atmeasuredsource,reportclosurelater; freshbuildsealafternewnativechanges. OwnA8packages /models/llama32-htp/l32-0032,reference tools/llama32_a8_fp32_reference.py andprepare_llama32_a8_fp32_cost.py. Fullreferences R/greedy-a8-teacher.json andfixed-a8-teacher.json. Mode3forcedtokens has noNLL, samefullheadwork asgreedy2.

RemainingapprovedA9log2versusFP32HVXvectorsoftmaxsameU8prob/AV, thenA8extraM/KV. RegisterL32-0033understandingapprovalbeforestatefulwork; no repeated userpermission. A1/A2/A3/A5/A6finished,A7citeQwen272. LlamaQKVringmaskinactive(no-op control),FARFfulltimelineN/A,no moretransportrepairs inthiscampaign. Preservefailedrotations/qualityscope.

---
# Active L32-0030: pipeline mechanisms and layer scaling

Read docs/experiments/L32-0030.md. QwenEXP0275 owns device now; local preparation only until it releases device. L32-0029 closed with exact fullmodel factorial, failures preserved. No rotation/PPL/other recipe change.

# L32-0029 completed: full-model factorial

Source f839f81d8a8d8641673992b4038c69b3f2c7edbd,codex/llama32-no-rotation; measurednative64717025e8c86031764b53a41c8ab55313ea3bf9 ABI131. Active none,next30. No promotion or quality acceptance. Fixedoriginal0018 SP2mode8+FP32residual1,no rotation,M64+15,cache80,16layers.

Independent0/7/15,chain3,chain16 outputs byteexact tosealed0016/0018; frontend generatedids+selectedlogitbits exact. Full5short10formal balanced4arms repeat10primary,one repeat1aux each. B00 1910.353/44.2801;B10 1924.934/43.7524;B01 2153.719/45.3116;B11 2161.744/45.1889token/s. B11/B00walls .88370916/.97988825;pipeline mainbenefit. Nativeformat notuniformly faster,decode slightlyworse;prefillinteractionCI crosses1. Historical0018 speednonpaired,not newkernel optimizationdenominator. Module tables/CI in docs/experiments/L32-0029-RESULTS.md.

All9600mainprofilesHMXcommands identicalbytoken,8MiB peak8098272,no tensorDDR/spill. Evidence/mnt/d/llm_exp/results/llama32-htp/l32-0029,816files294387837bytes,ledgerbfef7a0973d7d41eb00061eba218e2a94b75c083bced81be1e29fe755e34c0ee. Three initialB11decode crashes retained. Cause nullattention_header innewSwiGLU predicate,notBAD_HEADER(status2=RUNNING). Boundcurrentcontextbeforeworkerdispatch in6471702; no arithmetic/gate change. Initialfairnessrepair restricts modularNormdecodeconsumer to1live row; no Llamaformal beforethisfix.

All A1 complete; remaining Llama individualpipeline,decodeSP2copack,ordinaryA8/SP2FP32,layer/shape,FPvectorsoftmax incomplete,alreadyapproved. QwenEXP0275 currentlyownsdevice forindependentpipeline/layerscaling; serializeddeviceownership. Source tools execute_llama32_factorial/report_llama32_factorial/plot_llama32_factorial archived; exclusive pathscompleted,do notrerun. Freshbuildsealneededfornewphase; otherrecipes/rotation/PPLfrozen.

## L32-0029 continuation checkpoint (2026-09-15)

Current source6471702 on codex/llama32-no-rotation. Initial A1 method port preserved0018 arithmetic but first layer0B11 prefill exact followed bydecode RPC0x8000040d. IMPORTANT: dsp_status=2 means DSP_RUNNING, NOT BAD_HEADER. Prior narration of header rejection was incorrect; debug rejection lines never triggered. Device crashlog resolves fault to qbh_w4u8_swiglu_stream_worker_run format predicate, BadVA0xe798, null attention_header. Llama decode bypasses attention pool initialization; new predicate mistakenly depended on it. Fix binds current header/buffers in SwiGLU start after worker-idle checks and before release. No arithmetic/validation change. First attributable scheduling-context fix; diagnostic-only commits and all failures retained. Rebuilding layer1, then re-run selected gates under new attempt tags. Results /mnt/d/llm_exp/results/llama32-htp/l32-0029, decode_crash_diagnosis.json and diag-logcat-a02.txt record cause.

No new Llama timing accepted. Reused0018 ledger181files and six model packages verified. Source ABI131. Current runner execute_factorial.py in resultroot. Avoid existing l0-B11-audit tag; preserve failures and use corrected attempt tags. Next selected0/7/15 allarms,chain3,full16/frontend then fixed5short10formal. Qwen owns separate EXP0274; device execution strictly serialized. No quality/PPL/rotation/promotion or other recipe work.

# Active L32-0029: full-model no-rotation factorial

User approved actual full-model paper ablations. Read docs/experiments/L32-0029.md. Fixed selected L32-0018 SP2mode8+FP32residual1, original Llama weights, no rotation. Qwen EXP0274 owns its separate active phase; device execution serialized. Other approved items remain pending, not complete. All-on must reproduce sealed0018 outputs before timing.

# L32-0028 closure: no-rotation paper speed baseline selected

User selects L32-0018 W4A8 SP2mode8+FP32 residual,R3/R4 OFF, historical
2069.7025566101083/42.51010980249135tok/s as primary Llama paper speed baseline.
Supersedes0027 rotated primary selection; rotation acceptance/failures retained
as supplementary evidence. This does not promote fixed-scale text/PPL quality.

Source0ae07a92e4d99fd3fd23c89adc3ce4a2b44f7b4f,codex/llama32-no-rotation,clean/pushed; native src/include unchanged
from0f7d083. New source docs/PAPER_NO_ROTATION_BASELINES.md/json pin original
measured0018e3d065,exact command,evidence hashes and scope. Five hot files match
0018; later changes guarded rotations or restored trials. Rotation branch common
commits patch-equivalent except branch default; no missed validated OFF kernel.
All1810018+1511Qwen0268 evidence entries rehashed. New Qwen kernel changes are
not timed on Llama and do not inherit this historical throughput.

Active none,next29. No new Llama model/hardware/native/PPL. Qwen ownsEXP0269:
FP32 singlelayer reference/KV passes,final short speedgate fails; no formal or
fullmodel expansion. Original Qwen paper2030.24/48.17 retained. Reports and audit
in/mnt/d/llm_exp/results/qwen3-block-htp/exp0269;ledgerSHA25620d3ddb9e1152e35e714f45b159f26312e5a79177cae907c31b7952a8af26ff5. Next discuss Qwen prefill FP32 Norm/O cost or draft
paper from selected historical results; no automatic new Llama experiment.

---
Historical context follows.

# L32-0027 closure: paper revision; original fast path accepted with known rounding

User accepts the original fast HMX implementation for paper preparation and
explicitly pauses further software/hardware numerical alignment. This replaces
the prior next-step suggestion of H512 integer-plane repair. Do not resume it,
run new PPL, build models, or profile based on the old0026 handoff.

Source /home/daniuniu/work/llama32-htp, codex/llama32-no-rotation,
HEAD 0f7d083dc99af7e2983ecd426fca03e9d57a0a07. Native src/include unchanged from cb8be962f3f31a1b89c8f8cd2bebb9cceef049a7.
Only docs/PAPER_STORY_RUNTIME_V2.md and its EVIDENCE.json were added.
Active none; next L32-0028. No hardware/model/build/native work in0027.

Read the V2 report for the current paper story: consumer-native physical
interfaces and bounded-VTCM scheduling; exact SP2 integer decomposition to
native packed-W4; phase-specific decode row packing/prefill overlap; paired
cross-model attribution plus FP32 residual and dense rotation extensions.

Qwen EXP0268 fair optimized-U8 control gives SP2 fullmodel latency +2.7099%
prefill / +0.8185% decode. Llama0012 gives +1.8851% / -0.0234% (decode CI
crosses zero). Llama0018 FP32 residual without rotations fullmodel is
2069.70/42.51token/s, paired latency -3.2309%/+8.1282%. Llama0022 fastR3R4
on FP32-residual/SP2 has only singlelayer speed +6.2074%/-0.9298% (decode
CI crosses zero); no combined fullfrontend E2E or PPL. Do not compose these
separate rates into an unmeasured combined result.

Original fastfull160023 independent cosine .5051155954/.7882501259 FAIL
remains preserved. User acceptance is paper/performance working scope only,
not numerical gate/quality pass. Later0026 rejected candidates remain rejected;
mode3 exact dense is diagnostic only. Do not relabel failures, measured source
heads, binaries, or historical quality/default flags. Broad PPL usable-quality
claims and all-A8 floating-boundary claims remain unsupported.

Report SHA256 02f0b458fd2d79e73c711bfa16b741c40fb151217548da0baa6a8baad91b6904.
Evidence index SHA256 b2444718b14fffcbf57475a5bced441ca3ae2624b8df19030bfcba3069093379.
Checked all local document links, five Qwen frozen report hashes, existing0018
and0026 ledger hashes, and0018 speed calculations. Index records hashes of
consumed authority at registration commit47c5df3; those snapshots are historical,
not assertions that mutable authority files stay byte-identical after closure.
No new PPL/performance result. All frozen source branches remain unchanged.

Next work is paper drafting/figures from existing evidence. Further native or
hardware work requires a newly authorized bounded experiment and preflight.


---
## Historical L32-0026 context (superseded next action; retained findings)

# L32-0026 closure: original fast HMX retained

User prioritizes speed and correct implementation on original fast pipeline.
PPL/text quality deferred; exact scalar dense mode3 diagnostic only. Preserve
independent numerical/physical gates and10% speed gate.

Source /home/daniuniu/work/llama32-htp, branch codex/llama32-no-rotation,
closure cb8be962f3f31a1b89c8f8cd2bebb9cceef049a7; clean/pushed. Active none; next L32-0027.
Read docs/LLAMA32_H512_FAST_ROUNDING.md and docs/experiments/L32-0026.md.

Two bounded candidates: A scales H512 matrix by16 and converter1/16, producing
identical six phase/layer outputs/raw/stage1/finalR4 captures to original.
B reverses sixteen K32 tiles using same accumulator and one final conversion.
Both local0/7/15 components, exact actual tails, cosine pass. B first-factor
prefill halfword differences31/27/6 ->27/21/4, but SP2 differences1/1/2 ->2/2/0.
Layer15 output exact; other layers not consistently improved.

B chain3 passes: prefill cosine.9999999999245163,NRMSE1.06697e-7,7927changed;
decode exact, KVexact. Full16 fails: prefill cosine.609073601365145,NRMSE
.2967764437,116736changed; decode.7699031028646253,.6505855918,2048changed.
KV value differences352930/363758, append-structure checks0. No formal/E2E/PPL.
Do not promote, select order by layer, or mistake local pass for full16 pass.

All changed native src/include restored to parent7478516 via
f1162649cdbff56b134dbc8eab9a2517a0c35fec. Fresh layer0 audit identical to original.
Only result-ID tooling and report changes remain. N1 build seal atf116264
differs from docs closure: rebuild after successful preflight before device.
9CLI/18RPC,1exit0/8original ideal-exact exit1 retained,5successful builds,
no crashes.8MiB VTCM,peak8229344,ordinary intermediateDDR/spill0; auditcaptures
untimed.63prior files,5model manifests/876payload entries verified.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0026;
ledger 599d86cf654b2b41526690c76ad1c0a251b5f206815a57ce4ae12b0024f94ac9,120files. No processes active.

Next possible bounded fast option: exact binary-lattice decomposition of FP16
inputs into integer byte planes, native HMX +/-1 dense Hadamard dot products,
wide integer merge then single normalization/RNE. Not implemented or timed.
Observed per-row plane counts mostly3; whole-LUT fixed coverage needs4/4/5
planes for0/7/15. Cannot assume SP2 two-plane construction directly applies.
First validate exact representation and isolated cost; no full scalar dense
recompute. Need cover signed digits,overflow,full frozenLUT range and rounding.
Local v79 header exposesFP16 conversion; noFP32 output established. Bundled
v81 PRM37-bit accumulator description is not a v79 guarantee. Offline plain
FP32 double rounding explains only8/68 observedH512errors, not full contract.
Do not repeat H16 placement, exponent-shift or arbitrary order tuning.
Historical fast0022 passes chain3 and singlelayer speed; full160023 fails.
Exact0024full16 passes but enormous cost, retained diagnostic only. No new
quality conclusions or Qwen/otherrecipe work.

## Paired INT16 Down campaign closed

Companion Qwen EXP0284 completed with full896 layer-output hashes exact and formal speed gates passed; device released. L32-0040 sealed evidence unchanged. Combined report /mnt/d/llm_exp/results/paper-int16-down-20260916/REPORT.md. No active experiment, quality claim or baseline promotion.

# Current L32-0055 — user-approved generic A16 ABC completion
Read docs/experiments/L32-0055.md. A8/SP2 and historical D frozen; no per-model specialization. Fixed fixtures only.

## Current L32-0069
User-authorized ordinary-A8 HVX butterfly R4 diagnostic active. Read docs/experiments/L32-0069.md. No E2E requested; preserve exact scope and do not assume expected slowdown.

## L32-0069 closure (2026-09-23)
Component-only full8192 HVX butterfly after SwiGLU/before ordinaryA8 passes24 exact FP32/U8 audits and5short/10formal repeat10. No E2E run. Single-layer M64 core642.443us one-worker or165.773us four-worker; M1 10.035us. Direct16-transform core10.2749/2.64931ms M64 and0.160664ms M1. Optimized core/historical ordinaryA8 fullmodelwall9.93%/0.744%; nonpaired reference only, not E2E slowdown. Synthetic Gate/Up with original qparams, not real trajectory. No model folding or quality claim; no promotion. Source cc20e9afcfb9d375cac1eb3574979a01c46321d2; measured a43398ba64760da3756a006ba2dde99638873cd1. Read docs/experiments/L32-0069-RESULTS.md. Active none/device released.

## Current L32-0070
User requests fullmodel64+42 paired E2E for HVX butterfly R4 vs ordinaryA8. Read registered protocol;0070 owns device.

## L32-0070 closure (2026-09-23)
Full16 Llama1B ordinaryA8+HVX H8192 R4 passes exact1/3/16-layer and padding checks. Fresh original Down fold/per-channel RTN, other weights unchanged; FP32 residual, noR3/SP2/INT16 Down. Five short/ten formal repeat10 pairs on64+42: A8 prefill2398.611tps vsR4 2018.536tps, wall+18.829%; decode46.383 vs45.629tps, wall+1.653%. Core16 prefill2.61616ms; perdecode0.159800ms. Fullboundary4.21130ms/0.265678ms; Gate/Up producer changes outside these windows. No claim of butterfly slower than fullmodel, no quality/default promotion. Read docs/experiments/L32-0070-RESULTS.md; evidence/mnt/d/llm_exp/results/llama32-htp/l32-0070; ledger57c71782848c3ed05593a43577e38d54a82571dd663da704a97a7d78c192128d. Source closurecbc3ddc1b55ae8000645a77c7e3afac24f410c2b; activeNone/device released; next0071.

## Current L32-0071
User-confirmed HVX butterfly R3 added to0070 R4, same64+42 E2E. Side experiment, no deeper optimization. Read protocol;0071 owns device.

## L32-0071 closure
User-confirmed HVX R3+R4 on ordinaryA8, full16 64+42 complete. Five short/ten formal repeat10 paired R4/R3R4; own-contract1/3/16-layer, KV/hidden/norm/padding exact; no HMX rotation,8MiB/no intermediateDDR. See docs/experiments/L32-0071-RESULTS.md for speeds and mandatory boundary-contract caveat: inherited R3 FP16/vector quant path differs from nonrotated SF32/division repair, so negative difference is not butterfly itself accelerating computation. No quality/default promotion or extra R4 optimization. Source closure 2fcab92e71f6ba3deac39e55f96cf4ba76ba2968, measured f291128c946e4e1124f0d8af2288fbb05b27296d; ledger 8ddf5e0a483eccede1945df58d85517a1e99ac8fabe95322dea3967b03c96c6e; activeNone/device released,next0072.

## 2026-09-23 L32-0072
User approves R4-only four-model E2E expansion and new final workbook sheet. R3 combination excluded from controlled ablation. Active L32-0072; Qwen counterpart EXP-0309.

2026-09-23 device handoff: Llama3B formal10 and padding/boundary checks complete; release L32-0072, acquire EXP-0309 for two Qwen sizes.
