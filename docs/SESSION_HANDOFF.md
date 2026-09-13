# L32-0011 closed: full-model SP2 pipeline gate passed

No active experiment or running jobs. Source3575edfdf2cb5d433be79b88aaaa3295844e69dc,clean/pushed.
Tested source8331c9ef7496b70e3d71d57505c53b22d1f5f3da;current build16/seal8331c9e. Closure adds onlyreport.
Future deployment requires rebuild/reseal under a new approvedexperiment,
not relabeling old binaries. Rotation252aee5 andQwen48eb1ea frozen.

# L32-0011 completed: full-model SP2 pipeline gate passed

Candidate A01 is the only native candidate and only formal attempt. New mode5 streams prefill SP2 production from completed Up32-tile batches using three existing persistent HVX workers. Low plane moves from q alias to dedicated middle allocation so it cannot overwrite still-live Up input. Each worker owns disjoint tiles/private scratch. Flags publish after HMXwait; abort/join on failure; all workers join before Down. Down and decode arithmetic/schedule remain mode4. Existing genericU8 path unchanged. Native packedW4,241-level frozenSP2 table,alpha,Q31 and original qparams/weights unchanged. No added VTCM allocation.

Historical L32-0010 detailed prefill: Gate/Up5043.48us(U8)/5041.49us(SP2);activation2822.98us/6371.32us. Exposed activation explained the main module increase. New candidate overlaps HVX activation with Up HMX/DMA. New Gate/Up+SwiGLU fullmodel9792.2us vs originalU87892.4us;Down3554.8us vs2494.2us. Previous SP2 combined11412.8us is a non-paired historical reference. No separate proof of bank conflicts or specific microarchitectural stall cause.

Fixed ten AB/BA M64+15 same-build fullmodel pairs: prefillU82032.35998 /SP21859.20319token/s,latencyratio1.093134944,95%CI[1.088463022,1.097847090],PASS. DecodeU846.142635 /SP246.123949token/s,ratio1.000405124,CI[.994373023,1.006668006],PASS. Upper prefilloverhead9.7847% leaves only0.2153percentagepoints under10%; acceptance applies to this frozen prompt/context,not all sequence lengths. No unchanged formal repeats or selection among multiple runs.

Singlelayer0/7/15 outputs/KV exact,399360 outputcodes;consecutive3layers exact;16-step fullmodel greedyIDs/logitcodes match immutable independent integerreference forbotharms. All26successful deviceprocesses /360tokenboundaries (320formal) preserve8MiB,oneHMXowner,zero intermediateDDR/spill and weight expansion. All360ledgers additive. MaxVTCM8212960B. Assembly:producer/streamworker no vectorstack access;Down unchanged256B zero/255constant masks only,no tensors. Physical/algebraic layoutownership audit and all16LUT reconstruction checks retained.

Tested source8331c9ef7496b70e3d71d57505c53b22d1f5f3da. Layer1/3/16binaries,seals,assembly,immutable commands and raw results archived. No build/native failures. One local offline audit import failed using systemPython withoutnumpy,then completed under projectvenv;no device rerun. Models unchanged L32-0009 singlelayer and L32-0010 stack3/frontend;no weights generated. No PPL or quality/default promotion;SP2 remains repetitive in this case. DefaultoriginalU8 and frozenQwen/rotationbranch unchanged. Stop after bothformal gates pass.


Source docs/LLAMA32_SP2_PIPELINE.md; new runner
 tools/run_llama32_sp2_pipeline_e2e.py. Mode5 opt-in,baselineoriginalU8default.
Mode5 runtime requires4attentionHVXcontexts,32tileGate/Upbatches;mode3/4 controls
retained. No model regeneration. Use immutable L32-0009 packages-a01 single
layers and L32-0010 stack3/frontend-a01 with independent frozenSP2oracle.
Implementation only changes prefillproducer scheduling/low-plane base;Down
anddecode arithmetic/schedule unchanged. Existing worker timing subcounters
remain decode-only;prefill publication/consumption/join counters are measured.
Each fullmodelprefill published/consumed128groups. Failed Up aborts andjoins.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0011:PROFILE.md,profiling_summary.json,run_inventory.json,
contract-audit.json,producer-layout-audit.json,assembly-a01-audit.json,
binaries-a01-layer1/3 ande2e-a01/binaries contain alltested binaries/seals.
Ledger docs/experiments/L32-0011-evidence-sha256.json,150files,SHA25673ef239a76c6a07b7ebb5ed78504961fd025758cfffb26919635160c19114c8b.
All rawruns retained;oneformal attempt. Next userdiscussion;no additional
unchanged tests needed. Scope M64+15 only;PPLnotrun,SP2textstillrepetitive,
noquality/defaultpromotion. Historical W16/W4PPL andU8speed remain valid.
