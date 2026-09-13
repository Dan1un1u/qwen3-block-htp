# L32-0013 active: fresh C native-contract alignment

Owner /home/daniuniu/work/llama32-htp, codex/llama32-no-rotation.
Latest source 403848e3cd219a3e1c9942bdc521cef72f347f9c, clean/pushed. Latest change is report-only.
Authority protocol: docs/experiments/L32-0013.md. Source report:
docs/LLAMA32_C_RTN_SP2_ALIGNMENT.md. Do not use its historical checkpoint as live status.

## Live job: do not duplicate

Durable pipeline RUNNING in unified exec session20420, controllerPID48793.
Entry tools/run_llama32_c_pipeline.py; results/l32-0013/pipeline-a01 contains
owner, plan, per-stage logs/status. Stage08 integer_ppl is RUNNING, about12s
per sample,128 samples total. Read integer-ppl-a01/sample-*.json for progress.
Do not relaunch stages or overwrite artifacts. Process continues without a chat
connection. Controller stops at first failure; preserve attempt before repairs.

Initial/session11933 and C100 training are COMPLETE. B only initialized SW, no
optimizer updates. Initial source3809e2611323dc5a03f316be0e113c5fa997f79b,
B/C source6309a8a42c6aee13cecddace6e6631ecf15b4686. Export/calibration/oracle and
software evaluation sourcefbae5c537ed4e31d608406c41315936d89e4a70a.

Completed pipeline stages:00 Binit,01 Ctrain,02 export,03 calibration,04 package,
05 independent oracle,06 fixtures,07 teacher+input-only control PPL.
After08 integer PPL: build1 and C layer0/7/15; build3/consecutive3; build16;
deploy, exact generation/physical gate, device PPL, compare. All C device gates
remain PENDING. Baseline transport gates are not substitutes for new C gates.

## Fresh artifacts and contract

Models root /mnt/d/llm_exp/models/llama32-htp/l32-0013.
Results root /mnt/d/llm_exp/results/llama32-htp/l32-0013.
Fresh C quant-a01, calibration-a01, frontend-a01 and gates-a01 are completed.
Original source/model weights read-only. rotation-quant committed reference
snapshot d9a636ba273fa812b1d6098e9f9df0f8a33eb154; no old trained weights reused.
Initial100 and C100 actual histories verified. C112SW/376832channels allpositive,
none at floor.96 SA sites share48 parameters.17 rotation matrices maxerror6.509e-7.
Training-only R1 FP32 exceptO FP64, allR2 FP64; finalgamma/R1/R2fold FP64 with
reference BF16 casts and embeddingcentering. Centering not claimed RMSNorm exact.
C weight codesnative[-7,7], one scale peroutput; no groups/GPTQ/R3/R4.

SP2 mode9 exactly implements187signedvalues by v=low+257*high-32770, scalealpha/32768.
Includes +/-32768. Native packedW4 twoU8planes; fusedproducer andmode8pipeline.
Oldmodes retain241levels. Hardwarepriority retainsU8residual,nonlinear,KV,embedding,
head/projectionoutput contracts. Shared SA replacesindependent96scale semantics.
No floatingcontrol/Qwen changes orquality/defaultbaselinepromotion.

## Completed quality and important limitation

Full WT2 validation2048 including948tail,252728targets:
original BF16 teacher13.640870538532468;
fresh C input-only BF16 control17.29825809973266.
Same M64+16 bridge128windows/2048targets:
teacher23.85653710580871; input-only control31.456631990730184.
Integer/native scoring NOT COMPLETE. Never use partial PPL as final.
Historical17.6423999759 is not a required checkpoint reproduction and is not
comparable with the device bridge's context. Full2048 device evaluation is not
implemented; report scope explicitly. The software control keeps BF16 residual,
nonlinear,KV,head,embedding, and is not the hardware deployment.

Independent native greedy text is unusable (repeated commas/word fragments).
Read-only carrier statistics showlayer1 (secondlayer) output98.50235%zero,
30of64rows entirelyzero, U8step0.3745098. Evidence native-prefill-carrier-statistics.json
and oracle-a01 arrays. This is a localization clue, not yet a causal ablation or
device finding. SP2Down input does not protect its U8output/residual boundary.

## Hardware evidence already established (old sealed stimuli only)

Transport layers0/7/15 eachM64+1 exact; peak8229344/8388608VTCM, nospill/intermediateDDR.
Raw S32 probes5/6:4352outputs exact, all187levels and +/-1879048192 endpoints.
Total successfulDSPprocesses5,modelboundaries6,probeRPCs4. No C hardware execution yet.
CPU SP2 roundtrip/ties/dots passed. Input-only SP2 matches pinnedreference on790932
FP32/BF16/FP16elements. Batched offlineattention matches existingintegeroracle;
actualC config/shape cross-checks enabled. Existing SDKconversion library retained.
Per-targetcode equality and NLLatol5e-5; noA8 model-quality threshold. Physical
and integergates unchanged. Preserve failedFP32Oaudit and stoppedFP64initialattempt.
