# L32-0025 fast HMX normalization placement

User prioritizes speed and correct implementation. Continue incremental repairs
on the original fast HMX pipeline; exact mode3 is diagnostic only. Model-quality
PPL/text work is deferred. Numerical, physical and10% speed gates remain intact.

One candidate at6cdd7b4a7e5d02f159d14ae79f4019ed93d0ac1f moved H16 normalization
from FP16 output conversion into its matrix entries, exact +/-0.25, with unity
conversion. Padded values are masked after XOR. Prefill uses H16 as left operand,
decode as right. No extra HMX call, scalar recomputation, new layout or allocation.
H512 and R3, SP2 LUT and weights/goldens were unchanged.

Three audited native invocations, layer0/7/15 M64+decode1. All component
1ULP+minnormal checks and independent ideal singlelayer cosine gates pass;
actual conditional tail is exact. All six final output files are byte-identical
to sealed0022 fast outputs. All valid rawR4, stage1 and finalR4 halfwords match.
H16 on the actual stage1 input is exact both before and after this change in
all six fixtures. Therefore this candidate does not improve the observed error.

| Layer | Phase | H512 differing halfwords | Final R4 differing halfwords | Differing SP2 reconstructions |
|---|---|---:|---:|---:|
|0|prefill64|31|48|1|
|0|decode1|3|1|0|
|7|prefill64|27|60|1|
|7|decode1|1|0|0|
|15|prefill64|6|8|2|
|15|decode1|0|0|0|

Differences are against the independent full ideal R4 on the same captured raw
input, not against a newly accepted native golden. R4 H16 introduces no additional
conditional rounding difference here; pre-existing first-factor errors propagate.
These data do not prove that HMX fast execution has unusable PPL/text quality.
Historical0023 full16 numerical failure remains unresolved and unpromoted.

Reject the no-benefit candidate. Forward commitbcce0f4c3d6a0b66f19cafe7baecaf774ce79c84
restores the entire native R4 file to pre-experiment content, retaining0022 speed
pipeline and mode3 diagnostic code. A fourth fresh native layer0 audit confirms
restored outputs byte-identical to the sealed fast control and all local gates.
Source tooling adds only the L32-0025 result directory whitelist.

FourCLI/eightRPC, all original ideal-exact CLI exit1 retained; no crash/build
failure. Runtime counters retain R4 HMX calls9prefill/2decode, VTCM allocation8MiB,
peak8229344, no ordinary intermediate DDR/spill, exact local KV checks. The audits
write explicit untimed capture buffers and are not speed/physical promotion.
No new code-generation/stack-spill acceptance is claimed for the rejected candidate.
No new models, changed goldens, formal timing, chain3/full16 rerun, frontend,
E2E or PPL. No new token/s result. Identical component/output evidence provides
no reason to repeat expensive full16 or formal timing for this rejected change.
L32-0022 historical singlelayer speed remains at its original scope; it is not
an E2E result or acceptance of0023 full16 alignment.

Evidence: /mnt/d/llm_exp/results/llama32-htp/l32-0025.
compare_fast_a01.py verifies18 sealed0022 files and three model manifests with
309 payload entries, and writes comparison-a01.json. All attempts and binaries
are retained in the new ledger. Next direction: first-factor H512 fast arithmetic
and rounding, avoiding a scalar exact recomputation production path. Do not treat
small component errors or local exact tails as proof of full16 gate success.
