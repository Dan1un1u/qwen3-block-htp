# EXP0261 R4 overhead running handoff

Bootstrap current source; active EXP0261, branch codex/exp-0261-w4u8-dense-r4-cost.
Source HEAD 0cb3de8ce32b5c58137f195304bed50ff978345a. Native binary built from dabe2b6fc9a5511a919828d63dda3e6c8973d85f, ABI123, single layer0; subsequent edits are scripts only. Source/memory clean/pushed before resume. Results /mnt/d/llm_exp/results/qwen3-block-htp/exp0261, new models /mnt/d/llm_exp/models/qwen3-block-htp/exp0261/r4. No mllm.

Full6144 R4=PaleyH12 tensor SylvesterH512, explicit dense HMX factors, no butterfly. FP16 SwiGLU LUT from existing Gate/Up U8, FP16 inter-stage rounding; Down W R folded anew from original then per-output-channel RTN[-7,7]. Other C64 weights and static middle qparams frozen. This is hardware-cost exploration, not accuracy/PPL acceptance. HMX2 factors, prefill5commands/decode2, phase-dead VTCM scratch reused. Initial implementation uses scalar transpose/output indexing; those costs count in full wall and are not claimed intrinsic algorithm cost.

Export completed: source original shard hashes verified, full orthogonality/invariance, W4 unpack, LUT65536 entries correctly rounded Decimal80 and independently certified mpmath100. Original double backends disagreed at2half-midpoint entries; preserved export.log, repaired export_repair1.log; export_audit.json finalpass. Device package hashes verified.

Audit_a0/a1/a2 captures done: baseline optimizedR3, HMXR4, untimed independent scalarR4 after identical HMX state transition. All9steps R4 component <=1ULP+minnormal, quantized middle exact to ownreference; Down integer SDK reference and final residual0LSB. Fullblock HMX versus idealR4 max1LSB,mincosine0.999773, passes unchanged2/.999. AllpreR4live data exact, prefillKV exact. Known idealR3 gate failure unchanged.

Retained checker failures: missingPathimport repaired with same rawsuccessfulcapture; wrong residual layout and comparing dead decode padding repaired in84201e7, same rawcaptures rechecked, numerical_gate_repair1.log finalpass. No native numerical fixes needed after dabe2b6. No threshold changes or deletion.

Five short paired rounds completed (990timedRPCs), repeat1/10, all output hashes match auditcontrols; physical8MiB/nointermediateDDR/spill, fulladditive ledgers pass. Short repeat10prefill ratio21.0167,decode1.63656; primary overhead scalar layout/finish, denseGEMM small. Tenformal paired rounds are next/current; inspect live process/log before running. No fullmodel authorized withinEXP0261 and stable>10percent confirms stopbeforefullmodel.

Use existing Python /home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python from source:
 scripts/device_exp0261.py formal > results/formal.log
Do not rerun export/build/audits/short. Retained validated.json cells may resume after interruption, never skip failedcells. Finish independent report/reconstruction, complete PC027 tables/counters and provenance/hash seal, then close memory/source clean+synced. No baselinepromotion. New R4 E2E N/A; do not extrapolate.
