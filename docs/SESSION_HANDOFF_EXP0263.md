# EXP0263 checkpoint before fixed profiling

Source codex/exp-0263-w4u8-r4-parallel-prefill 672d742a302f480a56b4ecd3c76b638ee04d2cb0; activeEXP0263. NativeABI126 source05cbf364085d009c5c5acea6d7f45d3e7008612d, stagedexp0263-l1_attempt2. ReadPC077/protocol. Frozen EXP0261 packages, exactnativeW4/R4 rounding; otherrecipes unchanged.

OPT3 main+two existing poolworkers eachprivate256Bscratch splits prepare192tiles and finish12groups. OPT4 adds two-workerUpreadiness inputprep, preserved correct but154usjoin cancelsbenefit. route.json freezesOPT3 before timing: smokeparallel1619us,previous2201us,stream1765us. No further tuning. 145exactfilecomparisons across5audits vssealedEXP0262; independentfactor/Down/residual gates pass. Both491fileEXP0262 and371fileEXP0261 seals and144files/package localremote checked. WholeidealR3failureunchanged, noPPL.

Run scripts/device_exp0263.py short thenformal,5/10fixedpaired rounds repeat1/10,noR4 vsOPT3. Allcounters/physical/hashgates enforced, nofullmodel orE2E extrapolation. Then report_exp0263.py (exclusiveJSONwrites; runonce), commitreports, finalize_exp0263.py andmemoryclosure. Use numerical_gate.json,exact_native.json,layout_address_proof.json,route.json. Do not rerun oldexport/audits. Current resultdir /mnt/d/llm_exp/results/qwen3-block-htp/exp0263 unsealed.
