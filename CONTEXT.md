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
