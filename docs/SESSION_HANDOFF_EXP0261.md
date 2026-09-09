# EXP0261 completed R4 cost handoff

Source codex/exp-0261-w4u8-dense-r4-cost fbc601d72377cc57fa54cd3fcd831f870ee59fe9; active none,next262. Native single-layer ABI123 runtime dabe2b6fc9a5511a919828d63dda3e6c8973d85f, staged /data/local/tmp/qwen3-block-htp/exp0261-l1_attempt1. Bootstrap before future work. No mllm.

Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0261; seal db991e2b5c848afea5d53d55fded71b37d3d1573a37406cc70ad6e73756057a4, 371files. Read docs/experiments/EXP-0261-RESULTS.md and PROFILE.md. ARTIFACT_PROVENANCE.json records original/candidate package hashes, native files and binaries; local/remote verified. Other recipes/Selected baselines unchanged.

Full6144 R4=PaleyH12 tensor SylvesterH512, explicit dense HMX factors, no butterfly. Before middle A8, FP16 SwiGLU LUT from existingGate/UpU8. Down folded anew from original checkpoint then RTN[-7,7] peroutputscale, allotherC64weights frozen. Static middle range unchanged; this is an exploratory speed fixture, not PPL or quantizer-quality acceptance. FP16 normalizations and interstage rounding declared. VTCM scratch phase-dead, no allocation growth or fullmatrixDDRtransfer; first implementation uses scalar transpose/output indexing.

Export/hash/orthogonality/densefactor/xW invariance/independentW4/LUT65536 checks pass. HMX versus independent Float64 factors all9steps <=1ULP+minnormal; middlequantization and SDK nativeDown/residual0LSB. Fullblock vs scalarR4 max1LSB,mincosine0.999773 passes original2/.999. PreR4live values and prefillKV exact. Known idealR3 wholelayerfailure remains unchanged; no newPPL/text claim.

5short10formal rotatedpairs,repeat1/10,2970timedRPCs independently reconstructed; alloutputhashes/8MiB/zero timedintermediateDDR/spill/audit/oneRPC/completeledger checks pass. No extra rounds or outlier removal. Repeat10 prefill control1518.570us,R4 31899.607us,paired20.990140 CI[20.558788,21.473788]; decode control946.401us,R4 1583.942us,paired1.668882 CI[1.649037,1.695214]. Bothstable>10percent. Fullmodel neverstarted; E2E N/A,noextrapolation. Localgatefail is speed only.

R4 timing us: prefill prepare426.456,GEMM45.852,layout11933.557,finish17950.143; decode50.144,4.618,186.473,333.754. Dominantcostlayout/finish, notdenseGEMM. Decode losesSwiGLUstream overlap; included. This is not an intrinsicR4lowerbound. Discuss vector/fusedlayout andstreaming next; no automaticnewexperiment or baselinepromotion.

Retained failures/recoveries: double sigmoid backends differ at2halfmidpoints -> Decimal80 generation+mpmath100 intervalcertificate (export.log/repairedlog). MissingPathimport -> revalidateunchangedrawcapture. Checker incorrectlydecodedrowmajorresidual and comparednonlivepadding -> correctedactualsemanticlayout withsamecaptures; no native arithmetic fix, thresholdrelaxation or hashreplacement. Initialfailedlogs preserved. Do not rerun completedexport/build/audits/profiling.
