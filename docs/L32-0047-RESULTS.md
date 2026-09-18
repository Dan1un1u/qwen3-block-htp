# L32-0047 generic Llama3B W4A16 completion

Reuses sealed0041 per-channel W4 codes/scales. Existing FP16 embedding, residual, nonlinear/attention/KV and final norm; existing W4 coarse LM head. No SP2 or rotation. No shape-specific performance search.

Generic head128 HVX RoPE; separate capacity-correct compressed/expanded DMA slots; existing Gate4 pipeline with bounded16-slot producer/consumer ring and dead-phase reuse. Largest expanded slots cover the unchanged8-tile coarse head. Peak VTCM7964160/8388608 bytes, one HMX owner, one RPC/token, zero timed intermediate DDR/spill.

A genuine inherited native-FP16 KV bug was found: only one32-token delta tile was packed. At decode33 (KV97), additional tokens were omitted. The corrected path packs all padded tail tiles. Failed attempt retained in full-audit-b; corrected full-audit-c agrees bitwise with independent row-major cache reconstruction for264192 hidden/norm elements and all43 selected IDs/logit codes.

Selected0/13/27 and continuous3-layer M64+M1 pass unchanged composition_v2. Head128 RoPE196608 elements and W4 decode for K3072/K8192 (360448 elements) exact against independent ISA-simulator references. 1B floating bounded regression and3B A8/SP2 full audits retained.

Full independent FP16 mathematical alignment remains FAILED: maximum hidden/norm relative L2 0.006557147, existing limit0.003; minimum prefill cache cosine 0.999985184, existing limit0.99999. This is not relabeled a pass and thresholds were not relaxed. Numerical implementation crosschecks are separate from this mathematical alignment and from model-quality/PPL acceptance. No PPL or quality claim.

| Configuration | Prefill TPS | Decode TPS |
|---|---:|---:|
| Llama3B W4A16 | 493.223307 | 7.929781 |

Five short and ten formal repeat10 rounds. Repeat1 auxiliary only. Complete Host wall includes embedding/all28layers/final norm/LM head/greedy/FastRPC; excludes cold loading, tokenizer and audit collection. Project-owned prompt/trajectory, not named-dataset results. No automatic baseline promotion. Llama3B W16A16 remains user-deferred (5.25GiB FP16 backbone exceeds uint32 shared-offset ABI).
