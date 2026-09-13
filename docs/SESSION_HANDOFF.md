# L32-0012 completed: all three SP2 pipeline directions resolved

No active experiment or jobs. Source e03a0f8028b3f123dbc0263394c772483953dc22, clean/pushed.
Tested source/build16 seal ddbc07f422693534f7d29e554f1c92c1b87cea18; closure adds report only.
Read docs/SESSION_HANDOFF.md and docs/experiments/L32-0012.md.
Best opt-in mode8: same-build prefill2004.83 vsU82042.63token/s,
latency+1.8851%,95%CIupper3.4451%; decode46.08670 vs46.07593,
latency-0.0234%,CIupper0.3852%. Both10%gatespass.
Gather alone neutral; earlyGate/Up andDownHVX overlap improveprefill.
NoPPL/quality/defaultpromotion. FrozenQwen androtation unchanged.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0012; ledger 5277594741ac28c3f013ca33191456e9a1952160b30c2e6d616e2ad6f83e8552 (428files).

# L32-0012: SP2 prefill pipeline directions completed

The best tested candidate is opt-in mode8. Full-model prefill Host wall is
31.9228333ms versus originalU8 31.3321980ms (+1.8851%, paired95%CI
[+0.1907%,+3.4451%]); decode46.08670 versus46.07593token/s
(-0.0234%latency,CI[-0.4303%,+0.3852%]). Both10%gates pass.
Compared with same-build previousmode5, prefill saves2.1797345ms
(-6.3917%latency,CI[-7.9617%,-4.6879%]). No model-quality/default promotion.

## Three directions and measured attribution

- Mode6 issues four predicated half-table gathers into two private scratch
  vectors before either read. All16 immutableLUTs exhaust65536Gate/Up pairs
  each on device,old/newlow/high match independent scalar table values exactly.
  Gather alone versusmode5 gives+0.6551%prefill latency,CI[-0.5404%,+2.3586%]:
  no stable E2E benefit. CombinedGate/Up+SwiGLU falls only19.97us;
  this does not establish a useful isolated gather speedup or a bank-conflict cause.
- Mode7 reuses the pairedGate/Up DMA scheduler forM64. It alternates
  Gate0,Up0,Gate1,Up1 rather than completing allGate before streamingUp.
  Each completedUp32-tile group is immediately available to three existingHVX
  SP2 workers. Disjoint low/high output tiles and dedicatedmiddle allocation
  preserve input lifetimes. Compared withmode6, Gate/Up+SwiGLU drops
  9761.52→7930.08us and completeprefill saves5.0315%
  (CI[-5.6396%,-4.3093%]). Publication/consumption128groups per fullprefill.
- Mode8 adds two16KiBraw VTCM slots forDown. The soleHMX owner produces
  six signed24 retain stores perNtile (low/high,3digits each),publishesready,
  and starts the next tile while one persistentHVX worker reconstructsQ31
  output. Done is published after the final outputstore; slotreuse waitsdone.
  Each8-tile HMX command drains its epilogues before its DMA metadata slot is
  reused. Main keeps itsHVX slot inprefill; HMXproducer haszeroHVXinstructions.
  Compared withmode7,Down drops3572.30→3048.03us and completeprefill saves
  2.0738%(CI[-3.3390%,-1.1475%]). Decode retains prior packedrows/onepass path.

The measured mode8 combination includesmode6; its best result does not imply
that gather rescheduling was necessary. No extra unregistered ablation or
unchanged formal repeats. Relative toU8, remainingmode8 prefDown excess is
541.74us and Gate/Up+SwiGLU excess74.46us. HMXtilepairs remain1473024
forSP2prefill versus1210880U8; decodeboth1212928. Commands unchanged:
pref3381/decode1589. WeightDDR621918208B perfulltokenboundary for everyarm.
These counters support critical-path overlap, not elimination of extra SP2MACs.
Prefilljoin subcounter falls4489.32→2672.98us frommode6→7; it is included
inside the modulewall and must never be added again. No hardware stall counters
were collected,so avoid claiming a proven gather-bank or specificinstructionstall cause.

## Scope, correctness and evidence

Fixed10 rotated five-arm same-build cycles,M64+15,capacity80;everyarm occupies
eachposition twice. All50formalruns retained,no optionalstopping. Previousmode5
control hasprefillCIupper10.9408% inthiscohort (mean8.8419%),so doesnot pass
thiscohort's strictCIgate. Its historicalL32-0011 pass remains valid separately.
Mode6 also failscurrentprefCIgate; mode7 andmode8 passbothmodes.

Exactsinglelayers0/7/15 (399360outputcodes percandidate),consecutive3layers,
thenfull16-stepgreedy/logitcodes versus independentfrozenintegeroracle allpass.
83successfulDSPprocesses:16LUTprobeprocesses(32probeRPCs),12replayprocesses,
5fullmodelgates,50formalprocesses. All904modelboundaryledgers reconcile,
including800formalboundaries. MaxVTCM8229344B within8388608;no timed
intermediatetensorDDR/spill orW4→S8expansion. Newproducer/epilogue kernels
havezero vectorstackspills. Existingdecode consumer retains only256Bconstant
zero/255vectors onstack;no tensor-derivedspill. All1/3/16binaries andseals archived.

Weights,qparams,alpha,241-levelSP2LUT,Q31,attention unchanged. NoPPLrun;
SP2text stillrepetitive. Timingsincludeembedding,16layers,finalnorm,LMhead,
greedy andFastRPC;excludeexternal tokenizer,loading andsessionpreparation.
Results apply tothisfrozenprompt/context,not allshapes. OriginalU8 staysdefault.
Qwen3 andLlamarotationbranch remainfrozen.

Implementation/tested source: ddbc07f422693534f7d29e554f1c92c1b87cea18.
Runner: tools/run_llama32_sp2_pipeline_directions.py, modes0/5/6/7/8.
Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0012
(PROFILE.md,profiling_summary.json,contract-audit.json,run_inventory.json,
provenance.json,gather-a01,build-layer1/3/16,e2e-a01).
The closure commit adds only this report. Futurebuild/deployment must pass
newexperimentpreflight and rebuild/reseal;never relabel old binaries.


All tested1/3/16binaries/seals archived under build-layer1/3/16.
Ledger docs/experiments/L32-0012-evidence-sha256.json covers 428 files.
Nextwork needs an approvednewexperiment andfreshpreflight/buildseal;
do not relabel these old binaries after the documentationclosureHEAD change.
Mode8 invocation QBH_LLAMA_SP2=8 with the archived protocol's otherflags,
using immutable L32-0010/frontend-a01. No newweights. DefaultoriginalU8.
Do not repeat completed exploration or unchangedformaltests.
