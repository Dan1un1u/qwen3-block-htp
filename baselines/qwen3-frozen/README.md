# Frozen Qwen3 migration baselines

Source freeze: EXP-0265 `48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5`.
`manifest.json` pins historical experiment and measured binary identities,
package manifests, report hashes and actual launch evidence. Each schedule is
copied into `recipes/`; branch policy is in `config/branch.json`.

| Recipe | Evidence | Prefill 64 tokens (us / tok/s) | Decode 15 steps (us / tok/s) |
|---|---|---:|---:|
| W16A16 | EXP-0218 | 80692.1875 / 793.1375 | 1644005.3655 / 9.1241 |
| W4A16 C64 OPT2 | EXP-0260 | 61524.25515 / 1040.2401 | 980629.13765 / 15.2963 |
| W4A8 OFF | EXP-0259 a0 | 37529.6641 / 1705.3177 | 310191.34605 / 48.3572 |
| W4A8 R3 OPT2 | EXP-0259 a1 | 37572.823 / 1703.3588 | 313354.7289 / 47.8691 |
| W4A8 R3 + R4 OPT6 | EXP-0265 a1 | 40478.9505 / 1581.0687 | 325648.7318 / 46.0619 |

Warm complete 28-layer token boundary includes embedding, final norm, head,
greedy and FastRPC; excludes model loading, ADB and external tokenizer. F16 is
ten independent sessions with RPC repeat1, the others formal repeat10. Only
EXP-0259 OFF/R3 entries are same-campaign pairs here. EXP-0265 R3-only control
remeasured 1709.6976 prefill and 48.6855 decode tok/s; its paired R4 cost changes
are +8.0156% / +4.4334%. Do not rank unrelated campaigns by tiny differences.

W4A16 selects the exact decode optimization and per-channel C64 method, not
retired grouped artifacts. R3 retains the known ideal-reference whole-layer
failure. R3+R4 has no PPL acceptance; its single France-capital sample did not
answer the question. Neither is an accuracy-certified baseline. All historical
Selected records remain unchanged. There is no Llama speed or quality claim.

`verify --artifacts` hashes compact provenance and package manifests. It does
not rehash all weight payloads or rerun the historical experiments.
