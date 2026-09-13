# EXP0268 handoff — completed 2026-09-14

Authority /home/daniuniu/work/qwen3-block-htp-project-memory, branch codex/qwen3-block-project-memory. Bootstrap first, readfourauthorityfiles inprintedorder, then thishandoff. Newstatefulwork requires newlyregistered approvedexperiment andpreflight. No device/backgroundjob remains. Do not restart closed268 or modify sealedresults/models.

## Completed EXP0268 — shared U8 scheduling and fair SP2 cost

Source codex/exp-0268-u8-prefill-pipeline closure6378066a15d867cbf184c1cc92288018d42c2925; testednative dc32985fec67af1254b2d645594843c40db3a390,ABI131. Active none,next269. Source/memory normalcommits/pushes; no native change aftermeasurement. Read docs/SESSION_HANDOFF_EXP0268.md, source docs/U8_PREFILL_PIPELINE.md and EXP-0268-RESULTS.md/PROFILE.md.

All four U8 modes preserve LUT/quantization/one-passDown/decode. m1 threeHVX afterGateUp;m2 consumesUpready;m3 additionallyearlyGateUp withdedicatedmiddle protectingliveUpinput. SP2m8 unchanged. OriginalEXP0267 packages/evidence andhistoricalQwen/Llama branches reverifiedunchanged. No newmodels/PPL/quality/defaultpromotion.

Singlelayers0/14/27 andconsecutive3 U8 livecarriers/output/KV exact,repeat10 exact. FullallU8 feedbacktokens/selectedlogitcodes exact;SP2matchessealedSP2. Unuseddecode middle/Downrows4..63 excludedfromsemanticcomparison; allfourliverows andallotherbytes exact. Validator-only repairs preserved; allfourbinaries equal acrosssinglelayerPythonfixes. No arithmeticfailure/thresholdrelaxation/resampling.

5short+10formal fivearmsrepeat1/10; repeat10primary.13200fullmodelRPCs,369600exclusive layerledgers and2970singlelayerRPCs independentlyrecomputed. Allfivepeak8365824B,zero timedintermediateDDR/spill,oneRPC/token; allU8 HMXwork/commandcounts/weightbytesidentical.

FullprefillU8old/m1/m2/m3/SP2m8 Hostus37553.753/32952.016/31223.683/30691.671/31523.392. U8m3/oldwall−18.2727% CI[−18.5249,−18.0423]%;decode−0.0046% CI[−0.2798,+0.2410]%. SP2m8/U8m3 prefill+2.7099% CI[+2.3299,+3.0716]%;decode+0.8185% CI[+0.6323,+1.0089]%,bothupperbelow10%.

ActualhotE2E prefill/decode tok/s: U8old1704.224/48.5654; U8m32085.256/48.5676;SP2m82030.238/48.1734. Completeembedding/28layers/finalnorm/head/greedy/FastRPC; excludes tokenizer/ADB/coldload.16outputgenerationloop44.0864/44.8957/44.6088tok/s respectively. SP2Down extra1.0235ms remains; earlieradvantageagainstoldU8 waslargelysharedscheduling,notintrinsicallycheaperSP2arithmetic. U8m3 isexplicit QBH_SP2=0 QBH_U8_PREFILL_OPT=3; defaultnotpromoted.

Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0268;1511files,1308439140B,ledger61f9fa2b7525d01a334aa4596a40c134814bee292204685a54eef622590100af. This closure supersedes earlieractive/next pointers. No furtherexperiment or tuningstarted; awaituserdirection.

Frozenbranchrefs: QwenEXP0265 48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5;EXP0267 a088d087d847e7c19c0ef1e0874f141d285828d0;Llamano-rotation e03a0f8028b3f123dbc0263394c772483953dc22;Llamarotation252aee5ecfbefd4a2df6cd5a5e4b9193d935f86b. Llamaauthority remainsseparate.

Implementation retainsoriginalU8saturatinggather;SP2retainsitspriorpipelinedgather. Thiscomparison isbetweenconcreteoptimizedimplementations,notglobaloptimality. Sourceclosurechanges onlydocs/config/reportscript; alltimedfullmodelrunsuseone dc32985 build in exp0268-l28-a1. Timing continues fixed16outputs evenafterEOS. No fullvocabularylogit/PPL/semanticqualityclaim. Allownedfailedvalidator/extractionattempts retained in attempts.json; finalindependentchecks andledgersealed.
