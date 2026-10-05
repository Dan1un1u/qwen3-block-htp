# EXP-0319 — Qwen3-1.7B long-context decode pipeline

## Outcome
Same2048prefill+64continuousdecode, frozen W4A8 uniformINT16Down/FP32residual/no rotation: five alternating paired repeat1 short rounds and ten alternating paired repeat10 formal rounds. Same binary, same tokens/weights/quantization/capacity2112, option196511 control versus458655 candidate. No baseline promotion or new quality acceptance.

|Formal mean per sequence|Control|Candidate|Change|
|---|---:|---:|---:|
|Prefill Host wall ms|1313.190212|1267.189295|3.503% lower|
|Prefill token/s|1559.560818|1616.175269|3.630% higher|
|64decode Host wall ms|1888.106393|1745.084839|7.575% lower|
|Decode token/s|33.896395|36.674435|8.196% higher|

Paired bootstrap95% CI candidate/control wall: prefill[0.9641370698197914, 0.9657318044999739],decode[0.9233199790552935, 0.9251844626412858]. Complete Host wall includes per-call host staging and FastRPC, excludes cold load/tokenization. Each repeat resets KV then executes32prefill chunks+64decode calls; no repeated snapshot or throughput extrapolation. No Llama/other recipe changes or workbook edits.

## Matched length diagnosis
Three alternating-order scans at64/768/1024/2048prefill, all+64decode, common capacity2112 and the same frozen prefix/fixed continuation. See scan-summary.json. Control median TPS:64=1769.91/46.45;768=1856.72/43.93;1024=1804.90/42.83;2048=1571.52/34.30(prefill/decode). Timing scope differs from prior historical3decode observations.
First-scan per-token complete28layer attention wall grows2.405ms(short) to10.320ms(2048). Kpack work0.308->3.217ms,Vpack0.588->6.210ms,softmax0.671->4.398ms. These work counters overlap and MUST NOT be summed into wall fractions. QKV/head DMA waits remain approximately stable. Increasing KV work plus concurrency reduction explains the observed attention bottleneck; not all long-context costs are removable.

## Implemented candidate
Reserve393216bytes(384KiB) from unused space inside the existing8MiB grant, before projection weight buffers. Attention overlay increases2359296->2752512bytes; total planned peak6620384->7013600bytes. No extra VTCM request, intermediateDDR, KV duplication or modified arithmetic. Physical buffers remain disjoint; weights follow the reserve. Existing adaptive client scheduler now keeps four clients through1664validtokens and three through2112, versus old4through1408,3through1920,then2. Each group still computes one causal global softmax over its entire context.
The feature is opt-in long_optimization bit262144. Short/non-long default is unchanged. Activation/weight formats, norms, native head and INT16 decomposition remain identical.
Initial short attribution: attention wall10.331->8.059ms/token while K/V packing and softmax work remain essentially unchanged. Thus the main gain is coverage/concurrency, not less arithmetic. Repeated historical KV packing is still present. This is a partial recovery, not a claim of complete long-context optimization or short-context throughput parity.

## Correctness
Candidate2048+64 matches all2688nonzero per-layer hashes and65head outputs from auditedEXP0318; append guards check11505500160unchanged cache bytes. First35calls also match980independent CPU/ISA reference hashes.2047padding-poison and64token short-context regression are retained separately. All timing20160call records and13650head outputs match audited token/code/position references; exact8MiBVTCM and zero intermediateDDR/spill. Production zero hash fields are never treated as correctness evidence. PPL not rerun because this candidate preserves arithmetic exactly; previous PPL28.4670 remains an EXP0318 measurement, not newly measured here.

## Failed candidate retained
An earlier1179648byte reserve retained four clients at2048 but hit a precise HMX exception during prefillstep31 at qbh_hmx_accumulate_u8s8_projection+0x20,BadVA0xFF3FF000. It is rejected and excluded from accepted timing. Root cause is unresolved; no claim that four clients are inherently unsupported. Evidence audit-candidate and failed-audit-logcat.txt, source8c110dd. The bounded384KiB candidate passed the same boundary. Numerical/physical gates were not relaxed.

## Evidence and follow-up
Final source486c3cc5244d6d047988cf00f521485a09f6e968,branchcodex/exp-0319-long-decode-pipeline. Binary-a2 sealed against this source. Full additive stage ledger, overlapping projection/attention/MLP/engine counters and physical fields: FULL_PROFILING_REPORT.md and complete-profile-counters.csv/json. Historical A16 at2048+64 is N/A, not extrapolated. Full short/formal sample data and bootstrap ratios retained.
Further work should target persistent native KV with incremental append or reduced live attention operands; neither is claimed implemented in this experiment. The four-client fault requires a separate address/layout diagnosis before reuse. Existing baselines stay unchanged pending user selection.

Supplementary candidate short-context scan (not paired with the earlier scan; diagnostic only):
{
  "64": {
    "prefill_ms": 36.370052,
    "decode64_ms": 1383.258183,
    "prefill_tps": 1759.689537974815,
    "decode_tps": 46.267573751992764
  },
  "768": {
    "prefill_ms": 413.945575,
    "decode64_ms": 1459.323227,
    "prefill_tps": 1855.3163661672188,
    "decode_tps": 43.85594556153802
  },
  "1024": {
    "prefill_ms": 564.91599,
    "decode64_ms": 1495.295367,
    "prefill_tps": 1812.6589052648342,
    "decode_tps": 42.800908377321285
  }
}

LedgerSHA256 f4e72f9c08d487a1a0762769edb2290842fca9598dbc8733ec307a30af114c8c
