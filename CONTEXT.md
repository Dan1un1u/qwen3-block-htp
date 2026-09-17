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
