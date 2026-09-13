# L32-0013 active: fresh C-RTN/SP2 native-contract adaptation

Owner /home/daniuniu/work/llama32-htp, branch codex/llama32-no-rotation.
Latest committed source fbae5c537ed4e31d608406c41315936d89e4a70a, clean/pushed.
Read docs/experiments/L32-0013.md and source tools/llama32_c_rtn_train.py.
Source docs/LLAMA32_C_RTN_SP2_ALIGNMENT.md documents contract/evaluation limits.

## Live job and continuation

Initial R1/R2+SA100-update training is COMPLETE; session11933 finished normally.
Stage source3809e2611323dc5a03f316be0e113c5fa997f79b; log
/mnt/d/llm_exp/models/llama32-htp/l32-0013/initial/run.log.
Actual100 callback updates verified. Complete manifestSHA256
b7f5f1d682bde5d75cb263a51a7765c74ee67b2269ba6ba1036f2307f471d66d.
initial-training-audit.json verifies96 SA sites/48sharedparameters and17rotation
matrices maxorthogonalityerror6.9098e-7. Do not relaunch or overwrite it.
Automatic continuation is RUNNING in unified exec session20420, PID48793.
Source tools/run_llama32_c_pipeline.py, controller launch source6e8ae6a.
B initialization is COMPLETE (zero optimizer updates), and C100 training is
RUNNING at about36.5s/update, launch source6309a8a42c6aee13cecddace6e6631ecf15b4686.
C log: models/llama32-htp/l32-0013/c/run.log. After C completion it runs
export, calibration, package/oracle/fixtures, software evaluation, native layer
gates and full16 deployment/generation/device PPL. Do not duplicate its stages.
Inspect results/l32-0013/pipeline-a01 per-stage logs and completion records.
The controller stops at first failure; preserve that attempt before repairs.
B stage only initializes112 weight scales; no unused B training/GPTQ. C trains100
updates. Single GPU accumulation8 preserves effective batch8. QKV andGate/Up
share48 learned SA parameters for96sites. W4 training/export usesnative[-7,7].
Training-only FP32 denseR1 exceptO retainsFP64; allR2 FP64. FinalfoldFP64.
12-matrix selectiveFP64 precisionaudit passedNRMSE<=1e-4. EarlierallFP32O
failed and is retained. OriginalrepeatedFP64training was deliberatelystopped
beforecompletion and preserved in initial-fp64-a01; no shortenedtraining.

## Already validated

Reference snapshot is pinned GitHub d9a636ba273fa812b1d6098e9f9df0f8a33eb154
under source build/l32-0013/reference, manifest in results/reference-snapshot.json.
Original rotation-quant source and local experiments are read-only inputs.
Fresh training uses original BF16 model; no old rotated/quantized weight reuse.

Native opt-inmode9 exactly implements187-levelSP2 via
v=low+257*high-32770, scalealpha/32768. This includes bothsigned endpoints
that oldradix256 cannot jointly represent. Existingmodes retain241-levelcontract.
CPU all187levels,376tie/endpointcases and2048dotoutputs passed.
Transport-only sealedhistoricalstimuli layers0/7/15 eachM64+1 passzerooutputcode
mismatches, nointermediateDDR/spill. These arekernelproofs,NOTnewCquality.
RawS32probe modes5/6 passall187levels and+/-1879048192 worstintegeroutputs,
4352elements,2processes4RPCs. Production retains its strictersigned24partialgate.
Totaldeviceprocesses5,modelboundaries6,probeRPCs4. NoPPL orformalprofilingyet.

## Implemented but waiting for C artifacts / not yet validated end to end

1. export_llama32_c_rtn.py: original->FP64gamma/R1/R2fold, learnedSWRTN,
   packednativeW4; referenceBF16rounding/embeddingcentering recorded; freshnative
   quantizedhead andembedding (hardwarewins conflicts, noFP16controlchange).
2. calibrate_llama32_c_rtn.py freeze completed. data/calibration.bin=train32x2048
   disjointwindows; validation.bin252852tokens; bridge128deterministicM64+16
   windows2048targets. Completehistorical2048protocolhas252728targetsincludingtail.
   Calibrate stage pending: newweights, learnedsharedSA; freshnative-onlyboundary
   minmax andSP2responseMSE33+17logalphafit ontrainonly.
3. prepare_llama32_c_package.py prepare/oracle/fixtures; usesindependentinteger
   referenceandSDKlibnativeconversion, optionalinteger-exactFP32dotacceleration.
   All three stages pending. Never label skeletoncode as completedmodelvalidation.
Software-control SP2 matches pinnedreference for790932 FP32/BF16/FP16 elements.
   Batched offlineattention matches originalintegerreference on random/endpoint/constant
   stimuli; see attention-oracle-proof.json for exact count; actualC stimuli also cross-checked per
   config/shape. Deviceattentionunchanged.
4. New C singlelayer0/7/15, consecutive3thenfull16device gates remainpending.
   run_llama32_stack.py acceptsL32-0013mode9 withmarker/manifestguard.
   Buildbeforeeachdeclaredshape;newHEADrequiresfreshbuildseal; archiveoldbinaries.
5. evaluate_llama32_c.py teacher/integer andrun_llama32_c_device.py
   deploy/generate/ppl/compare pending. Teacher stage also evaluates a fresh C
   input-only software control with BF16 residual/nonlinear/KV/head/embedding
   against the same native C backbone112, sharedSA48 and fittedSP2 alpha.
   Both floating models get full2048 validation and matchedbridge; initial
   hardware/softwarebridge isM64+16,notcomparabletohistorical17.6424.
   Per-tokenNLLatol5e-5,exacttargetcodes;noA8modelqualitythreshold.

Do not call this the originalCcheckpointorclaim17.6424reproduction. The user
explicitly superseded that dependency: new training, existingintegerresidual,
nonlinear,KV,head/embedding takeprecedence. Full2048hardwareevaluation isnotyet
implemented; do not silentlytruncateandclaimfullWT2coverage. Noqualitypromotion.
Qwen3 andfloatingcontrolsunchanged. Preserveallfailedandstoppedattempts.
