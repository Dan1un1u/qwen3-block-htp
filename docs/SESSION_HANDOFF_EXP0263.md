# EXP0263 completed parallel R4 prefill handoff

Source codex/exp-0263-w4u8-r4-parallel-prefill 01b42be717fbed33fe413009f40f9427f4574106; active none,next264. Native singlelayer ABI126 runtime05cbf364085d009c5c5acea6d7f45d3e7008612d at exp0263-l1_attempt2. Bootstrap before future work. No mllm.

Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0263; seal 1404333eb0e87d0442d86c05be20df1e66d35c332373c58f6640c3ab10409b09,483files. Read source EXP-0263-RESULTS.md/PROFILE.md and result MODULE_TABLES.md. All source/memory committed/pushed. Frozen EXP0261 original/R4 packages unchanged,144files each verified localremote; entire EXP0262 491file and EXP0261 371file seals reverified. audit_a2 readonlylink to EXP0261 scalaroracle. No model exports/calibration/qualityclaims.

OPT3 main+two existing HVXworkers split preparation192tiles and finish12groups; private256Bscratch/context uses existing768Ballocation. H12/H512 FP16normalizations/roundings/nativeW4Down unchanged. Stage2 uses inherited8batch doublebuffer9R4HMXcommands; decode inheritedOPT2stream unchanged. No workercreation or VTCMallocationgrowth. OPT4 correct Up-readiness two-worker stream tried once;154us finaljoin canceled overlap, complete smoke1765us vsOPT3 1619us. route.json frozeOPT3 before fixedprofiling. Both options retained; no further tuning.

145exactfilecomparisons across5capturegroups vssealedEXP0262, independentFloat64R4 and SDKDown/residual gates pass. All liveFP16/U8/hidden/capturedKV exact. Address ownership partitions and independent scratch nonoverlap verified. KnownidealR3 fullblock failure unchanged. Otherrecipes/Selected unaffected.

5short10formal rotatedpairs repeat1/10,2970timedRPCs; alloutputhashes,pipeline/taskcounts,8MiB/no reportedintermediateDDR/spill/audit/oneRPC and completeledgers independentlypassed. OPT3prefill9paralleldispatches,192preparetiles,96finishgroupvisits; eachjoin before reuse.

Formal repeat10pre1528.89310us->1703.39845us,pairedratio1.11874007 CI[1.07643566,1.14731754]; decode955.9762875us->992.8672125us,pairedratio1.03463332 CI[1.01341578,1.04343464]. PrefillCI crosses10percent: eligibilityINCONCLUSIVE, not confirmedstable>10 slowdown, but also not confidencepass. No optionalextra samples or fullmodel. localgatefail denotes eligibilitynotestablished, evidencevalid,numericalpass. E2E/PPLN/A. HistoricalnonpairedpreviousR4 2223.5182us/1006.8245375us.

Prefillprepare153.794us,finish200.690us,layout62.120us,exposedHMX25.737us. Decode9.723/16.318/.972/4.320us. Remainingcost primarily outputconversion and preparation; investigate repeated quantization parameter setup/gather and worker scheduling in a separately registered experiment if user approves. Do not claim extra thread overhead or totalDDRtraffic from incomplete counters; aggregateworker and pipelinework overlap wallledgers. No automaticbaselinepromotion.

No native or numerical collection defects in this experiment. OPT4 slower smoke retained as research result, not discarded failure. Do not rerun completedexport/audits/profiling.
