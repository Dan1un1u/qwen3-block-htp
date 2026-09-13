# Native packed-W4 SP2 activation execution

The user confirmed Llama L32-0012's implementation/performance conclusion and requested Qwen migration in EXP-0267. This fixes an arithmetic and scheduling method; it is not PPL or usable-text acceptance. Historical Llama results remain sealed and neither Llama branch changes.

## Quantization and exact multiplier mapping

The frozen codebook has 241 levels: zero, signed single powers, and shared-sign sums of two distinct powers, with exponents 0..14 and maximum magnitude24576. For each layer alpha=f32(maxabs(frozen A8 middle range)/24576). A65536-entry LUT maps dequantized Gate/Up codes through SiLU(Gate)*Up to the nearest alpha-scaled level, tie toward the lower numeric level. It emits v+32768 in two native U8 HMX planes; no intermediate FP16 tensor or explicit SP2 decoding pass is introduced.

For each SP2 reconstructed integer v, let l=(v+32768)&255 and h=(v+32768)>>8. Then

\[
v=l+256h-32768,\qquad
\sum_k v_k w_{kn}=\sum_k l_k w_{kn}+256\sum_k h_k w_{kn}-32768\sum_k w_{kn}.
\]

Both dot products use the existing native packed signed-W4 instruction, original W4 weights and one FP32 weight scale per output channel. Weight sums are precomputed. Three retained conversion byte planes recover each signed24 dot exactly; exported worst-case positive/negative bounds include arbitrary U8 inputs. The combined accumulator fits signed32 for the frozen codebook/weights. Q31 multipliers m=round(alpha*s_w/s_down*2^31), then clamp(((acc*m+2^30)>>31)+zp_down), reproduce the frozen output quantization contract and original Q14 final residual. This is exact execution of these quantized values, not equality to floating-point SiLU or the teacher.

## Why decode avoids a second matrix traversal

The original decode path consumes only four physical rows of a64-row tile (one logical token). Put the low plane in rows0..3 and high plane in rows4..7. Both see the same W4 matrix in one HMX command/weight traversal. HVX combines the paired outputs. It uses spatial rows that the prior implementation already paid for. Thus unchanged tile/command counts and weight reads are possible despite carrying a16-bit reconstructed activation. This does not imply zero scalar/HVX work or universal zero latency overhead.

Prefill already occupies64 rows, so it needs two HMX passes; additional arithmetic cannot be removed by the decode trick. The measured solution hides much of the added work through production and consumption scheduling.

## Prefill pipeline

1. Gate/Up publish completed32-tile groups. Three existing HVX workers directly produce low/high native activation planes. Mode5 overlaps producer work after Up groups complete; mode8 alternates Gate and Up groups early so the first usable pair is available sooner.
2. Mode8 Down uses one HMX producer and one existing HVX worker. Two16KiB VTCM raw slots carry retained dot bytes; an additional2KiB bias area yields34816B scratch. Ready/done publication protects slot and metadata lifetimes. Each command drains outstanding epilogues before DMA bias-slot reuse.
3. The HMX producer has no HVX instructions; the HVX epilogue and LUT producer have no vector stack spills. The serial reference's256B vector stack contains only zero/255 constants, not tensor payloads. DMA maintains the original packed-weight double buffering.

The separate four-gather scheduling change had no stable Llama benefit; it is retained inside the proven mode8 implementation without a renewed parameter search. No butterfly rotations, grouping, additional HMX owners, or timed intermediate DDR are introduced.

## Qwen lifetime adaptation

Qwen3 uses28layers and K6144, while Llama uses16layers and K8192. Qwen's existing capacity128 persistent KV atlas nearly fills8MiB. A separate393216B high plane plus34816B scratch would not fit. During direct-n MLP, Gate/Up input occupies the q-based arena; earlier attention/O consumers have joined. The hmx_activation allocation is dead until next-layer input normalization or the final LM head. Qwen overlays the SP2 high plane and Down scratch there, preserving live KV, residual and Gate/Up buffers. R3/R4 combinations are rejected. This is a lifetime reuse, with no extra copy or reduced KV capacity. U8 and allSP2 arms reach the same8365824B planned peak.

## Evidence and scope

Llama L32-0012: optimized SP2 prefill2004.835tok/s vsU82042.627tok/s, latency+1.885%, decode46.0867 vs46.0759tok/s; optimized prefill latency−6.392% vs prior mode5. Both10% gates pass. These are sealed historical measurements at capacity80, not paired absolute comparisons to Qwen capacity128.

Qwen EXP-0267 final full28layer repeat10: U8 prefill1705.305tok/s, decode48.4689tok/s; SP2mode8 prefill2020.269tok/s, decode47.8808tok/s. SP2mode8/U8 latency−15.590% prefill (95%CI[−15.850%,−15.326%]), +1.228% decode (95%CI[+0.979%,+1.470%]); both10%gates pass. Mode8 prefill latency−5.235% vs prior mode5 and−9.819% vs serialstage mode4. The within-SP2 optimization is comparable to Llama mode8/mode5−6.392%. Complete M64+15 modelHost346.406ms(mode5) to344.957ms(mode8), illustrating that decode-dominated total generation gains are smaller than prefill gains. See REPORT.md, summary.json and module tables for full telemetry. Before fullmodel, all28 device LUTs exhaust65536 pairs; independent Down/Q31/residual references pass on real teacher-supplied isolatedlayer0/14/27 inputs and a consecutive3layer slice. Optimized and serial SP2 feedback outputs agree. Decode unused middle rows8..63 are not semantic comparison targets; rows0..7 and other captures/KV must agree. Full generation comparisons cover selected tokens/logit codes, not all vocabulary logits. Fixed5short/10formal, repeat10 primary and repeat1 auxiliary; no changed thresholds, outlier filtering, optional resampling or default promotion.

An observed speedup over original Qwen U8 must be attributed to the combined producer/consumer pipeline: it does not establish that SP2 arithmetic itself is cheaper, or that an equally optimized U8 producer could not improve too. The transferable conclusion is that native W4 instructions can execute a wider nonuniform activation contract via exact integer decomposition, and unused spatial capacity plus scheduling/lifetime alignment determine its exposed cost.

## Fair U8 comparison after EXP0268

EXP0268 completed on Qwen3-1.7B,28layers,M64+15feedback tokens,capacity128,5short+10formal rotated fivearm repeat1/10. Repeat10 primary; repeat1 auxiliary. Source tested dc32985fec67af1254b2d645594843c40db3a390. No PPL or default promotion.

OriginalU8 -> threeHVX -> Upready -> earlyGateUp Host prefill:37553.753 ->32952.016 ->31223.683 ->30691.671us. Incremental reductions4601.736/1728.333/532.012us; final−18.2727% CI[−18.5249,−18.0423]%,throughput+22.36%. Decode20590.792 ->20589.842us,−0.0046% CI[−0.2798,+0.2410]%: unchanged within uncertainty.

SP2m8 Host31523.392us/20758.365us. Against optimizedU8, prefill+2.7099% CI[+2.3299,+3.0716]%;decode+0.8185% CI[+0.6323,+1.0089]%. Bothupper<10percent. This is the cost of the current concrete optimizedSP2 implementation relative to optimizedU8, not a global optimization bound.

GateUp+SwiGLU13989.263 ->7158.924us forU8, whileDown3388.288 ->3374.706us. OptimizedSP2 GateUp7038.054us,Down4398.214us: SP2's extraDown cost1023.508us remains, partly offset by its producer and measured boundary-time differences. The earlier SP2 advantage against oldU8 largely reflected the oldU8 producer schedule. No claim SP2 arithmetic is intrinsically faster; shared scheduling should be backported before making quantization-method performance comparisons.

All fourU8 modes have identical HMX work/weightDDR/commandcounts; allfive sharepeak8365824B,zero timed intermediateDDR/spill andoneRPC/token.Independently counted13200fullmodelRPCs and369600exclusive layerledgers, plus2970singlelayer timedRPCs. All selectedlayers0/14/27 andconsecutive3 U8 livecarriers/output/KV exact againstsealed original; fullU8 feedbacktokens/selectedlogitcodes exact,SP2 matchessealedSP2. No fullvocabulary/PPL claim. Failedvalidator attempts retained; no arithmetic changes, newweights or resampling.

Actual hot E2E prefill/decode token/s: oldU8 1704.224/48.5654; optimizedU8 2085.256/48.5676; SP2m8 2030.238/48.1734. Tokenizer,ADB,coldload excluded; model includesembedding/all28layers/finalnorm/head/greedy/FastRPC.16output generationloop rates44.0864/44.8957/44.6088token/s respectively, separately includingdeviceHostloop/serialization.

See docs/U8_PREFILL_PIPELINE.md and EXP-0268-RESULTS.md for the fixed fivearm ablation. EXP0267/Llama results remain valid under their original control implementation; no historical evidence is rewritten.
