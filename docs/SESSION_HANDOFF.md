# L32-0013 completed: native C/SP2 arithmetic aligned, quality unusable

No active experiment or live job. Next experiment number14; no new run approved.
Owner /home/daniuniu/work/llama32-htp, codex/llama32-no-rotation.
Source closure bc5fdc1a8b044d4442ace7e2122ebc47cf0937d2; tested native e2b3b91c78b3bd9667d14d16484a129223ce6174.
Source and authority are committed/synchronized at closure. The frozen rotation
branch remains 252aee5ecfbefd4a2df6cd5a5e4b9193d935f86b; Qwen3 andfloatingcontrols unchanged.
Read docs/LLAMA32_C_RTN_SP2_ALIGNMENT.md and docs/experiments/L32-0013.md.

## Established result

Fresh original Llama1B-Instruct -> initial100 R1/R2+SA -> B initialSW only ->
C100 jointR/SA/SW -> finalFP64fold withreferenceBF16casts -> nativeW4[-7,7].
112backboneLinears,376832learnedchannelSW,96inputsites with48sharedSA.
16DownSP2 sites use187levels, exactradix257 nativeW4twoU8planes (mode9),
v=low+257*high-32770, scalealpha/32768. Old241-levelmodes unchanged.
Hardware-priority keepsnativeinteger residual/nonlinear/KV/embedding/head.
No old trained/folded weights reused. rotation-quant source read-only.

Full WT2 validation2048+948tail,252728scoredtokens:
BF16 teacher13.640870538532468; freshinput-onlyBF16Ccontrol17.29825809973266.
This control keepsBF16residual/nonlinear/KV/head/embedding. It is not the original
historical17.6424checkpoint. New training and hardware-contract adaptations are explicit.

Matched M64+16bridge,128windows/2048targets:
teacher23.85653710580871; input-onlycontrol31.456631990730184;
nativeintegerreference138395.59496349443; device138395.59980512303.
All2048targetcodes equal; max pertokenNLLerror1.4944266819583163e-6 <5e-5.
Singlelayer0/7/15,consecutive3,andfull16generation/PPL allpassarithmetic/physical.
Text is unusable: repeatedcommas andwordfragments. Quality/defaultNOTpromoted.
Full2048hardwarePPL wasNOT run; nevercomparebridgePPLdirectlywithfull17.6424.

## Localized bottleneck / next discussion

On the BF16 control trajectory, applying onlylayer1(secondlayer) Down-outputU8
quantization makes99.99845% ofordinary-positionentrieszero (NRMSE0.999913).
First-positionRMS15.167 vsordinary0.03463; sharedstep0.37402. BlockoutputU8
makes99.89382%ordinaryentrieszero. Nativefirstprefilllayer1has98.50235%zero
entries overall,30of64rowsentirelyzero. SP2protectsDowninput butnot itsfollowing
U8output/residual. GlobalNRMSEhidesordinary-tokenloss becausefirstpositiondominatesenergy.
Diagnostic is isolatedboundaryquantization,not acumulativePPLablation.
Nextdiscussseparatingfirst-positionandordinary-tokenDownoutput/residualscales
under nativeW4. Do not keepretuningrotationbeforeaddressingthisknownboundaryloss.
No suchcontractchange/newexperimenthasbeenstarted.

## Evidence and recovery

Models /mnt/d/llm_exp/models/llama32-htp/l32-0013/frontend-a01.
Results /mnt/d/llm_exp/results/llama32-htp/l32-0013; summary.json,SUMMARY.md,closure.json,
pipeline-a01/complete.json,device-e2e-a01/alignment.json,
float-carrier-diagnostic-a01/result.json andevidence-ledger.json.
Ledger SHA256 b7f08359a7f76d374fafaa10b2e311758ef50bd48ecf7a15b7eb6ab6d8b599d2.
520resultfiles,57externalartifacts and1092modelmanifestentriesverifiedatsealing.
All20pipeline stagescomplete. Sessions11933,20420,4499,95212finished; do notrerun.
11successfuldeviceCLIexecutions,2078recordedmodelstepboundaries,4rawprobeRPCs.
PeakVTCM8229344/8388608bytes,zero intermediatespill/DDR,packedW4maintained.
AuxiliarysinglefunctionalM64+15speed:2075.9773prefill,46.0603decode token/s.
Noformalprofilingor10%speedtest; L32-0012 remainslastformalspeedbaseline.
KeepfailedallFP32OauditandstoppedslowFP64initialattempt. FinalFP32R1exceptO/allR2FP64
trainingauditpassed; finalfoldFP64; initialandC100updatesverified, noSWfloorchannels.
