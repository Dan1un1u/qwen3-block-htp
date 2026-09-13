# EXP0268: shared U8 prefill scheduling

Qwen3 W4U8 M64 uses unchanged native packedW4 and per-output-channel scales. Modes selected explicitly by QBH_U8_PREFILL_OPT; QBH_SP2=0. Quantized activations, original U8 LUT gather, saturating pack, native one-pass Down requantization and residual arithmetic are unchanged.

- 0: original Gate projection, Up projection, then one HVX context produces all SwiGLU tiles.
- 1: Gate and Up remain serial stages; two persistent pool workers plus the main context produce disjoint SwiGLU tiles, with three independent256B gather scratch regions.
- 2: after Gate, three persistent workers consume completed Up groups while subsequent Up groups execute on HMX and DMA stages the next weights. Group readiness uses release publication and consumer barriers. Workers join before Down.
- 3: alternate Gate and Up batches for each1024-channel group. The first SwiGLU group can start before the remaining Gate groups complete. Three workers consume each ready group; HMX ownership and weight double buffering stay unchanged.

Modes2/3 store the producer output in the existing dedicated middle buffer. The legacy q arena remains live as Up input until all projections complete; writing middle there early would violate the input lifetime. No new allocation is needed. Readiness/abort state resets each layer. Per-worker scratch and output tiles are disjoint; all workers join on success/error before buffer reuse. This is a storage-lifetime and scheduling change, not a quantization change.

Decode ignores the U8 prefill selector and retains the original row4 producer and native Down. Only four physical decode rows are defined; relocation changes stale padding in middle and corresponding unused Down rows. Audits require all four live rows, all other boundary bytes, full prefill carriers, output and persistent KV to match. Full generation compares feedback token IDs and selected logit codes; it does not establish full-vocabulary logit equality or PPL.

Unchanged optimized SP2 mode8 remains a separate numerical recipe: two radix256 U8 planes, two native-W4 prefill products and exact Q31 recombination, with its existing two-slot Down epilogue overlap. Decode places low/high in otherwise unused physical rows and retains one HMX tile traversal. SP2 uses its previously validated pipelined gather; U8 retains its original saturating gather. The comparison measures two concrete optimized implementations, not a global optimization bound.

Interpret work and wall time separately. All U8 modes retain identical HMX tile pairs, command counts and weight bytes. Scheduling moves producer work off the exposed critical path. SP2 still has extra prefill tile products; any advantage against the old U8 baseline must be separated from shared scheduling improvements before attributing a benefit to SP2 arithmetic.

## Measured conclusion, 2026-09-14

EXP0268 completed on Qwen3-1.7B,28layers,M64+15feedback tokens,capacity128,5short+10formal rotated fivearm repeat1/10. Repeat10 primary; repeat1 auxiliary. Source tested dc32985fec67af1254b2d645594843c40db3a390. No PPL or default promotion.

OriginalU8 -> threeHVX -> Upready -> earlyGateUp Host prefill:37553.753 ->32952.016 ->31223.683 ->30691.671us. Incremental reductions4601.736/1728.333/532.012us; final−18.2727% CI[−18.5249,−18.0423]%,throughput+22.36%. Decode20590.792 ->20589.842us,−0.0046% CI[−0.2798,+0.2410]%: unchanged within uncertainty.

SP2m8 Host31523.392us/20758.365us. Against optimizedU8, prefill+2.7099% CI[+2.3299,+3.0716]%;decode+0.8185% CI[+0.6323,+1.0089]%. Bothupper<10percent. This is the cost of the current concrete optimizedSP2 implementation relative to optimizedU8, not a global optimization bound.

GateUp+SwiGLU13989.263 ->7158.924us forU8, whileDown3388.288 ->3374.706us. OptimizedSP2 GateUp7038.054us,Down4398.214us: SP2's extraDown cost1023.508us remains, partly offset by its producer and measured boundary-time differences. The earlier SP2 advantage against oldU8 largely reflected the oldU8 producer schedule. No claim SP2 arithmetic is intrinsically faster; shared scheduling should be backported before making quantization-method performance comparisons.

All fourU8 modes have identical HMX work/weightDDR/commandcounts; allfive sharepeak8365824B,zero timed intermediateDDR/spill andoneRPC/token.Independently counted13200fullmodelRPCs and369600exclusive layerledgers, plus2970singlelayer timedRPCs. All selectedlayers0/14/27 andconsecutive3 U8 livecarriers/output/KV exact againstsealed original; fullU8 feedbacktokens/selectedlogitcodes exact,SP2 matchessealedSP2. No fullvocabulary/PPL claim. Failedvalidator attempts retained; no arithmetic changes, newweights or resampling.

Actual hot E2E prefill/decode token/s: oldU8 1704.224/48.5654; optimizedU8 2085.256/48.5676; SP2m8 2030.238/48.1734. Tokenizer,ADB,coldload excluded; model includesembedding/all28layers/finalnorm/head/greedy/FastRPC.16output generationloop rates44.0864/44.8957/44.6088token/s respectively, separately includingdeviceHostloop/serialization.
