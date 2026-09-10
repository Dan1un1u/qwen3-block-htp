# L32-0005 closed: A8 relative speed restored

Both Llama source branches are clean and synchronized; common trees differ only
in config/branch.json. Qwen3 freeze unchanged. No active experiment or job. Next
experiment L32-0006 requires user direction. Source heads and ledger are in status.

Native W4 was already faster but repeated V LUT generation, scalar softmax,
unused64-row common/SwiGLU/residual work and RoPE row copies hid the advantage.
Reuse exact per-config LUT, HVX softmax, row4 common/SwiGLU, pooled residuals,
integer RMS tree and aligned head64 RoPE gather with sparse scalar edge repair.
Weights/qparams and all quality status remain unchanged.

Independent A8 single/3/16 exact output and KV gates pass. Padding poisoning
and8 softmax scalar comparisons pass. W4 full16 byte-identical to prior evidence;
OPT2 audit passes. Full generated IDs and selected codes equal baseline, including
A8 full15-decode. Exactly8MiB VTCM,peak7668960 A8/8330752 W4,zero timed
intermediate DDR/spill,single HMX owner,native W4 weights without expansion.

Fixed10 rotated ABC/BCA/CAB cycles, shared original M64 prompt plus7 decode each:
W4A16 prefill1260.21580/decode23.59130 tok/s;
old A8 prefill1081.55838/decode16.70891;
new A8 prefill1431.83506/decode39.55584.
New A8 vs W4: throughput+13.62%/+67.67%, Host ratio.880140/.596405;
95% CI[.871268,.887285]/[.593203,.598958].
New A8 vs old: throughput+32.39%/+136.74%, Host ratio.755365/.422413;
CI[.751633,.760720]/[.421104,.423666]. Both retained1.10 gates pass.
All240 timed additive ledgers reconcile. Excludes load/session preparation and
external tokenizer; includes embedding,16layers,finalnorm,head,greedy,FastRPC.
No repeat1 gate, optional stopping, cross-length comparison or extrapolation.

Primary remaining prefill costs: GateUp/SwiGLU32.0%,attention25.9%,KV conversion
6.1%. Decode: GateUp/SwiGLU30.8%,attention22.8%,LMhead12.6%,Down11.7%,Host9.0%.
Current A8 quality is still unusable (prior PPL1206603.740108); no quality gate.
W4A16 prior PPL31.039101 vs BF16teacher26.697999 still fails quality criteria.
No new PPL or quality-baseline promotion. Testedcapacity80,M64+7/15 only;
Llama rotations and arbitrary-length serving remain unsupported.

Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0005. Source report
/docs/LLAMA32_A8_RELATIVE_SPEED.md; profiler/report tools are experiment-specific
and require new immutable destinations for future use. Native last-changebe389d2,
profiledsourcea6f5e61; final source changes after profiling only reporting.
All failed predicate,CLI,compile attempts and SDK probes retained, see recovery_notes.
44 DSP processes incl1 rejected candidate,43 successful;2 additional CLI rejects;
306 executed token boundaries including2 in the failed candidate,304 successful.
No model/Qwen artifact removed or changed. Original BF16 root remains read-only.
