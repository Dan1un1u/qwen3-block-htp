# L32-0043: 3B scale target and exact native-layout optimization

Completed2026-09-17. User-approved research/optimization loop, no quality claim or automatic baseline promotion.
Source branch codex/llama32-no-rotation. Final measured native d69d8f9ca9103c1888b1426ab482e58f2ed60dda; launcher/campaign982053d7315040f88b50bca51a83898407e54c63.
Frozen control76418bbbb504dc30608441b17f49a774c10b72a2 (L32-0042); frozen0041 original3B weights/scales/references.

## Research and gate
Qualcomm ai-hub-models v0.62.2 provides matched1B/3B NPU end-to-end metrics onSnapdragon8Elite:
- GenieX W4:1383.589333->672.519690prefill;31.243782->12.888389decode.
- GenieX W4A16:3410.332241->1256.182775prefill;65.745355->27.651214decode.
- Genie W4A16:58.379513->28.378454decode;1Bprefill missing, not inferred fromTTFT.
Pinned primary sources:
https://raw.githubusercontent.com/qualcomm/ai-hub-models/v0.62.2/src/qai_hub_models/models/llama_v3_2_1b_instruct/perf.yaml
https://raw.githubusercontent.com/qualcomm/ai-hub-models/v0.62.2/src/qai_hub_models/models/llama_v3_2_3b_instruct/perf.yaml
Our historical1B2335.236148/45.740369 projected by those ratios yields860–1135prefill and18.87–22.23decode.
Context/precision/runtime differ, so these are scale guides, not hardware limits or directrankings. Quant.npu Table12 lacks pairedLlama1B.
Absolute engineering target fixedbeforecandidates:1100prefill/22decode token/s, full28M64+15. Existing exact/physical gates and bothphase95%CI upper<=1.10 retained.

## Three bounded directions
1. Batch three query heads sharingK/V into existing single-owner HMX submissions duringprefill. Perfullmodel commands6605->3021, same physicaltilework.
2. Vectorize exact V clipping telemetry using monotone LUT thresholds. Isolatedinitialpilot shows no materialbenefit; no separate speedclaim.
3. Co-packthree live decodequeryrows into oneM64, using two existingGate scratchslots; remove10752padding tilepairs/token.
The original launcher still selectedAVrows64, so earlyOPT1/OPT2 andOPT3 pilots didnot activate the new rowbound. FinalOPT3a explicitly selects QBH_W4U8_DECODE_AV_REQUANT_ROWS=4; selectedlayers andchain/fullchecks rerun.
Finalspeed is combined; no isolated speedclaim forco-packing vsrow4.
All editscompile-guarded3B.1B .text differs only diagnosticline immediate0x575f->0x57a0; arithmetic andschedule instructions identical.

## Formal full-model result
M64+15/cache80/28layers,5short+10fixedpairedAB/BA cycles,repeat10. Noearlystopping; fixedtoken trajectory.
Includes embedding,allblocks,finalnorm,head,greedy,RPC; excludes coldload/session setup andexternal tokenizer.

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| Paired L32-0042 control |1040.690614|21.200743|
| OPT3a |1139.982409|22.234309|
| Throughput gain |9.540952%|4.875143%|
| Absolute target |1100 PASS|22 PASS|

Latencyratios .91290059 (CI .91083894–.91472673), .95351479 (CI .95224536–.95466752).
Bothspeedgates pass. Historical1B ratio .48816/.48610 approximately; cross-size nonpaired reference.
Attentionprefill11.9147->6.6449ms, decode9.5118->7.3246ms/token. GateUp/Down unchanged withinnoise.
See MODULES.md forfullstable additive module table, SUMMARY.json forroundmetrics/20000pairedbootstrap.

## Validation and scope
Selected0/13/27,chain3/full28 FP32hidden+KV exact independent0041goldens; fixedtrajectory generatedIDs/logitcodes exact.
Separate truegreedy16step invocation (mode2) exact; noPPL/generalquality acceptance.
Frozen1Blayer7prefill/decode exact;1Bmath unchanged as above.
All3200formalprofiles reconcile. Peak8360416/8388608VTCM;oneHMXowner,oneRPC/token,zero intermediateDDR/spill.
Weightbytes1613512704/token unchanged. Decoderphysicaltilepairs3153408->3142656; logicalvaliddotproducts unchanged.
Prefix/KV persistentformat unchanged. Newcompactpath limitedto3B,decode1,rowmajorKV,paddedKV<=128; fallback preservedotherwise.
Gate scratchdead duringattention,align16KiB+32KiBconsumes<48KiBinside512KiBallocation. Scoreoverlay<=73728Bwithin196608Bscoreallocation; noaliaslivebuffers.
Prior0042evidence531filesrehash verified. All attempts preserved, including source-edit preconditionfailure beforeanywrite andone redundantunstagedOPT2build.
Reproduce with tools/execute_llama32_3b_loop.py and tools/profile_llama32_3b_loop.py; frozenruntime seals andexact commandlines ineachrun.
To reuse fastest3B recipe, retain existingSP2mode8/FP32residual/no-rotation/WIDE_SCORE8 flags AND AV_REQUANT_ROWS=4.
No further optimization needed for this registeredtarget. Next opportunities: GateUp/Down weightDMA/HMX coverage andnormpreparation; newshapes/longKV need newboundedvalidation.
