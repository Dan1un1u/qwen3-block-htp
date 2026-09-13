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


# L32-0012 SP2 pipeline directions

SM8750 / HTP V79; Llama3.2-1B-Instruct,16 layers,M64+15,capacity80. Ten fixed rotated same-build five-arm cycles. Arithmetic: frozen241-level SP2 / native packed signedW4 / Q31.

U8: original integer; m5: previous Up/SP2 overlap; m6: +issue/read-separated gather; m7: +early Gate/Up alternation; m8: +Down HMX/HVX double-buffer overlap.

## prefill

Microseconds; parentheses are complete Host wall shares. Decode values are per token.

| Module | Original U8 | SP2 m5 | +gather m6 | +early G/U m7 | +Down overlap m8 |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 138.6 (0.44%) | 136.4 (0.40%) | 136.6 (0.40%) | 138.0 (0.42%) | 137.3 (0.43%) |
| Input RMSNorm | 1320.1 (4.21%) | 1320.4 (3.87%) | 1318.9 (3.84%) | 1319.1 (4.05%) | 1322.1 (4.14%) |
| QKV＋RoPE | 3853.3 (12.30%) | 3766.7 (11.05%) | 3782.6 (11.02%) | 3776.0 (11.58%) | 3777.3 (11.83%) |
| QK–Softmax–AV | 8094.6 (25.83%) | 8005.4 (23.47%) | 8072.7 (23.52%) | 8089.1 (24.81%) | 8061.9 (25.25%) |
| O projection | 712.0 (2.27%) | 714.1 (2.09%) | 713.3 (2.08%) | 712.5 (2.19%) | 714.7 (2.24%) |
| Post-attention residual＋RMSNorm | 697.0 (2.22%) | 693.4 (2.03%) | 694.1 (2.02%) | 694.7 (2.13%) | 695.0 (2.18%) |
| Gate/Up＋SwiGLU | 7888.1 (25.18%) | 9781.5 (28.68%) | 9761.5 (28.44%) | 7930.1 (24.33%) | 7962.5 (24.94%) |
| Down | 2506.3 (8.00%) | 3563.3 (10.45%) | 3563.9 (10.38%) | 3572.3 (10.96%) | 3048.0 (9.55%) |
| Final residual | 348.0 (1.11%) | 346.7 (1.02%) | 346.5 (1.01%) | 347.3 (1.07%) | 349.8 (1.10%) |
| KV carrier conversion | 27.0 (0.09%) | 27.4 (0.08%) | 27.7 (0.08%) | 27.4 (0.08%) | 27.9 (0.09%) |
| KV append DMA | 103.8 (0.33%) | 103.5 (0.30%) | 103.2 (0.30%) | 104.5 (0.32%) | 103.8 (0.33%) |
| Block orchestration | 22.3 (0.07%) | 21.6 (0.06%) | 21.7 (0.06%) | 21.7 (0.07%) | 22.1 (0.07%) |
| Layer bookkeeping | 12.8 (0.04%) | 12.7 (0.04%) | 12.9 (0.04%) | 13.1 (0.04%) | 13.2 (0.04%) |
| Stage-boundary bookkeeping | 7.7 (0.02%) | 7.6 (0.02%) | 7.7 (0.02%) | 7.6 (0.02%) | 7.7 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 83.2 (0.27%) | 81.6 (0.24%) | 80.9 (0.24%) | 81.7 (0.25%) | 81.6 (0.26%) |
| Embedding | 43.3 (0.14%) | 42.9 (0.13%) | 43.1 (0.13%) | 43.5 (0.13%) | 43.3 (0.14%) |
| Final model RMSNorm | 3.8 (0.01%) | 3.7 (0.01%) | 3.9 (0.01%) | 4.0 (0.01%) | 3.9 (0.01%) |
| LM head＋greedy（不含 final norm） | 3270.9 (10.44%) | 3236.2 (9.49%) | 3234.1 (9.42%) | 3232.2 (9.91%) | 3227.2 (10.11%) |
| Host–DSP 边界 | 2199.4 (7.02%) | 2237.6 (6.56%) | 2400.6 (6.99%) | 2484.2 (7.62%) | 2323.5 (7.28%) |
| 完整 Host wall | 31332.2 (100.00%) | 34102.6 (100.00%) | 34326.0 (100.00%) | 32598.9 (100.00%) | 31922.8 (100.00%) |

## decode

Microseconds; parentheses are complete Host wall shares. Decode values are per token.

| Module | Original U8 | SP2 m5 | +gather m6 | +early G/U m7 | +Down overlap m8 |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 112.3 (0.52%) | 111.5 (0.51%) | 111.6 (0.51%) | 112.5 (0.52%) | 112.6 (0.52%) |
| Input RMSNorm | 85.0 (0.39%) | 85.0 (0.39%) | 84.9 (0.39%) | 85.0 (0.39%) | 84.9 (0.39%) |
| QKV＋RoPE | 1649.3 (7.60%) | 1632.6 (7.51%) | 1636.2 (7.50%) | 1635.9 (7.53%) | 1635.5 (7.54%) |
| QK–Softmax–AV | 5929.8 (27.32%) | 5931.8 (27.27%) | 5935.9 (27.20%) | 5935.6 (27.31%) | 5935.2 (27.35%) |
| O projection | 712.8 (3.28%) | 711.0 (3.27%) | 711.4 (3.26%) | 712.7 (3.28%) | 711.9 (3.28%) |
| Post-attention residual＋RMSNorm | 168.7 (0.78%) | 168.4 (0.77%) | 168.6 (0.77%) | 168.6 (0.78%) | 168.8 (0.78%) |
| Gate/Up＋SwiGLU | 4891.2 (22.54%) | 4896.7 (22.51%) | 4909.5 (22.49%) | 4907.3 (22.58%) | 4908.7 (22.62%) |
| Down | 2489.1 (11.47%) | 2572.1 (11.83%) | 2574.7 (11.80%) | 2576.8 (11.86%) | 2577.6 (11.88%) |
| Final residual | 81.4 (0.38%) | 81.1 (0.37%) | 81.1 (0.37%) | 81.0 (0.37%) | 81.1 (0.37%) |
| KV carrier conversion | 49.9 (0.23%) | 50.1 (0.23%) | 50.0 (0.23%) | 50.0 (0.23%) | 50.0 (0.23%) |
| KV append DMA | 83.7 (0.39%) | 83.7 (0.39%) | 83.7 (0.38%) | 83.7 (0.39%) | 83.7 (0.39%) |
| Block orchestration | 16.6 (0.08%) | 16.6 (0.08%) | 16.6 (0.08%) | 16.6 (0.08%) | 16.6 (0.08%) |
| Layer bookkeeping | 9.0 (0.04%) | 9.0 (0.04%) | 9.0 (0.04%) | 9.0 (0.04%) | 9.0 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) | 1.2 (0.01%) | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.1 (0.24%) | 51.2 (0.24%) | 51.2 (0.23%) | 51.2 (0.24%) | 51.2 (0.24%) |
| Embedding | 1.3 (0.01%) | 1.0 (0.00%) | 0.9 (0.00%) | 1.0 (0.00%) | 1.0 (0.00%) |
| Final model RMSNorm | 2.8 (0.01%) | 2.7 (0.01%) | 2.7 (0.01%) | 2.7 (0.01%) | 2.7 (0.01%) |
| LM head＋greedy（不含 final norm） | 3254.0 (14.99%) | 3227.0 (14.84%) | 3227.0 (14.78%) | 3227.7 (14.85%) | 3227.8 (14.88%) |
| Host–DSP 边界 | 2114.1 (9.74%) | 2118.3 (9.74%) | 2170.3 (9.94%) | 2076.6 (9.55%) | 2038.7 (9.40%) |
| 完整 Host wall | 21703.3 (100.00%) | 21751.0 (100.00%) | 21826.7 (100.00%) | 21734.9 (100.00%) | 21698.2 (100.00%) |

## E2E

| Mode | Original U8 | SP2 m5 | m6 | m7 | m8 |
|---|---:|---:|---:|---:|---:|
| prefill token/s | 2042.63 | 1876.69 | 1864.48 | 1963.26 | 2004.83 |
| decode token/s | 46.08 | 45.97 | 45.82 | 46.01 | 46.09 |

## Matched latency gates

| Mode | Comparison | Latency change | Paired bootstrap95 CI | Upper <=10% |
|---|---|---:|---|---|
| prefill | m5_vs_u8 | +8.842% | [+6.758%, +10.941%] | False |
| prefill | m6_vs_u8 | +9.555% | [+8.331%, +11.292%] | False |
| prefill | m7_vs_u8 | +4.043% | [+3.089%, +5.418%] | True |
| prefill | m8_vs_u8 | +1.885% | [+0.191%, +3.445%] | True |
| prefill | m6_vs_m5 | +0.655% | [-0.540%, +2.359%] | True |
| prefill | m7_vs_m6 | -5.032% | [-5.640%, -4.309%] | True |
| prefill | m8_vs_m7 | -2.074% | [-3.339%, -1.148%] | True |
| prefill | m7_vs_m5 | -4.409% | [-5.296%, -3.108%] | True |
| prefill | m8_vs_m5 | -6.392% | [-7.962%, -4.688%] | True |
| decode | m5_vs_u8 | +0.220% | [-0.662%, +1.004%] | True |
| decode | m6_vs_u8 | +0.569% | [-0.110%, +1.364%] | True |
| decode | m7_vs_u8 | +0.145% | [-0.762%, +1.025%] | True |
| decode | m8_vs_u8 | -0.023% | [-0.430%, +0.385%] | True |
| decode | m6_vs_m5 | +0.348% | [-0.309%, +1.169%] | True |
| decode | m7_vs_m6 | -0.421% | [-1.086%, +0.242%] | True |
| decode | m8_vs_m7 | -0.169% | [-0.808%, +0.468%] | True |
| decode | m7_vs_m5 | -0.074% | [-0.750%, +0.604%] | True |
| decode | m8_vs_m5 | -0.242% | [-0.990%, +0.565%] | True |

Host wall includes embedding,all16 layers,final norm,LM head,greedy and FastRPC. Excludes external tokenizer,model loading and session preparation. No PPL/quality promotion. Repeat1 auxiliary only. All800 formal plus104 functional model boundaries have exact additive ledgers. Sixteen probe processes additionally exhaust all65536 pairs on each actual layer LUT; both old/new outputs match independent scalar LUT reconstruction.
