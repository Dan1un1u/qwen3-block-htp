# L32-0022: repair dense R4 numerical error without losing pipeline speed

The previous L32-0020 implementation also fails the newly introduced three-layer
rotation check. Its native R4 file was restored from d3ba5e6 in ordinary commit
24595931ff874e80c095ac829902bf25e7092f1f and rebuilt for three layers. On the exact
sealed L32-0021 chain3 fixture, both output files are byte-identical to L32-0021.
Previous experiments never tested this chain. It was not a regression introduced
by L32-0021 layout/SP2 scheduling on this fixture; rollback alone does not fix it.

The repaired candidate preserves the fast L32-0021 pipeline and changes only
where H512 normalization is applied: normalized FP16 dense weights and unity
HMX converter scale replace +/-1 weights and non-unity converter scale. The
reference remains sum(x * sign) * FP16(1/sqrt512), rounded to FP16. The scale is
an exact dyadic constant; moving it into the products is equivalent in exact
arithmetic, but the two HMX paths have different finite-precision behavior.
The experiment identifies non-unity output scaling as the dominant observed
error source; it does not establish undocumented converter internal bit widths.

Candidate1 be978d098f916bd6ee88b623d12e8126bb07c7ad validates this move.
Final candidate 3fb9a4f694e36552c1d7f75fc40de59007cfc8cf generates normalized
constants with one XOR (0x3c00 xor0x15a8 =0x29a8), at the original sign-table
construction instruction cost. It is byte-equivalent to candidate1 on all
three isolated layers and the continuous3 fixture. No changed weight bins,
SP2 LUT, calibration, dense GEMM dimensions/counts, residual arithmetic or gates.

| Continuous3 check | Old0020 / optimized0021 | Repaired0022 |
|---|---:|---:|
| Prefill minimum row cosine | 0.79150509299 | 0.999999998751 |
| Prefill relative L2 error | 0.02600079775 | 0.000000464915 |
| Decode minimum row cosine | 0.91193451478 | 1.0 |
| Decode relative L2 error | 0.4185249805 | 0 |

All isolated layer0/7/15 component and exact conditional-tail gates pass. The
worst final isolated minimum row cosine is0.999999999709; every isolated decode
and the continuous3 decode is bit-exact against the independent ideal reference.
Continuous3 prefill retains11456 differing FP32 elements, with errors shown
above; it passes the unchanged .999 minimum-row gate. Original CLI exact-match
exit1 is retained, never relabeled an exact pass. OFF continuous3 remains exact.
Layer0 H512 prefill relative error drops from1.348232e-4 to6.262318e-7; the full
single-layer output error drops from1.084299e-3 to7.678631e-7.

VTCM peak8,229,344B <=8,388,608B. No timed intermediate activation DDR/spill,
weight expansion, cache prefix/structure failures or extra HMX calls. Exact
function disassembly shows finish/layout scalar frames88/24B with no activation
stack vector staging. The run_dense_r4 saved vector is traced to a zero constant,
not activation. Initial disassembly extraction accidentally included the entire
binary; retained a01 text is superseded by exact-function a02 extraction.

Formal speed: ten fixed cyclic three-arm cycles,30 CLI processes, each one warmup
and ten measured M64+decode1 pairs (past64,capacity80),660RPC/600timed.
Paired process-mean bootstrap20,000 draws seed22022. Each current arm is compared
to current OFF; both95% upper latency ratios must be <=1.10. No optional stopping.
Old0021 is an immutable timing control only; its chain3 failure stays unaccepted.

| Phase | OFF us | Repaired R3+R4 us | Delta OFF | 95% latency-ratio CI | Gate |
|---|---:|---:|---:|---|---|
| prefill | 1945.139 | 2065.881 | +6.207% | 1.038908–1.087676 | PASS |
| decode | 1464.737 | 1451.117 | -0.930% | 0.961116–1.021637 | PASS |

Matched repaired/old0021 Host-wall ratios are0.975361 (95%CI0.959475–0.991106)
for prefill and0.971143 (0.946454–0.996836) for decode. Most of that observed
reduction is in Host–DSP boundary time. Gate/Up+SwiGLU actually changes
525.80->527.84us prefill and340.77->341.59us decode. Therefore this is evidence
that the numerical repair preserves pipeline speed, not evidence that changing
normalization accelerates HMX computation. Do not attribute Host variation to
a faster matrix kernel. OFF/rotation weights differ by the prescribed R4 fold;
old/new rotated comparisons reuse the same packages byte-for-byte.

Complete module ledgers follow; units us (own Host-wall share). R3 is included in
QKV+RoPE and R4/SP2 in Gate/Up+SwiGLU. Frontend modules are outside this scope.

prefill

| Module | OFF | Previous0021 R3+R4 | Repaired0022 R3+R4 |
|---|---:|---:|---:|
| I/O、metadata | 40.56 (2.09%) | 44.50 (2.10%) | 37.55 (1.82%) |
| Input RMSNorm | 76.85 (3.95%) | 76.84 (3.63%) | 76.81 (3.72%) |
| QKV＋RoPE | 236.67 (12.17%) | 181.77 (8.58%) | 181.38 (8.78%) |
| QK–Softmax–AV | 523.59 (26.92%) | 522.17 (24.65%) | 522.05 (25.27%) |
| O projection | 99.24 (5.10%) | 98.01 (4.63%) | 98.35 (4.76%) |
| Post-attention residual＋RMSNorm | 78.80 (4.05%) | 86.65 (4.09%) | 86.75 (4.20%) |
| Gate/Up＋SwiGLU | 325.15 (16.72%) | 525.80 (24.82%) | 527.84 (25.55%) |
| Down | 198.46 (10.20%) | 197.58 (9.33%) | 198.97 (9.63%) |
| Final residual | 0.08 (0.00%) | 0.08 (0.00%) | 0.08 (0.00%) |
| KV carrier conversion | 1.62 (0.08%) | 1.48 (0.07%) | 1.46 (0.07%) |
| KV append DMA | 8.68 (0.45%) | 8.87 (0.42%) | 8.29 (0.40%) |
| Block orchestration | 1.10 (0.06%) | 1.10 (0.05%) | 1.08 (0.05%) |
| Layer bookkeeping | 0.68 (0.04%) | 0.68 (0.03%) | 0.69 (0.03%) |
| Stage-boundary bookkeeping | 0.41 (0.02%) | 0.41 (0.02%) | 0.40 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.97 (2.52%) | 48.74 (2.30%) | 48.93 (2.37%) |
| Embedding | N/A: outside single-layer scope | N/A: outside single-layer scope | N/A: outside single-layer scope |
| Final model RMSNorm | N/A: outside single-layer scope | N/A: outside single-layer scope | N/A: outside single-layer scope |
| LM head＋greedy（不含 final norm） | N/A: outside single-layer scope | N/A: outside single-layer scope | N/A: outside single-layer scope |
| Host–DSP 边界 | 304.28 (15.64%) | 323.40 (15.27%) | 275.26 (13.32%) |
| 完整 Host wall | 1945.14 (100.00%) | 2118.07 (100.00%) | 2065.88 (100.00%) |

decode

| Module | OFF | Previous0021 R3+R4 | Repaired0022 R3+R4 |
|---|---:|---:|---:|
| I/O、metadata | 40.44 (2.76%) | 44.43 (2.97%) | 38.18 (2.63%) |
| Input RMSNorm | 53.01 (3.62%) | 53.06 (3.55%) | 53.05 (3.66%) |
| QKV＋RoPE | 100.90 (6.89%) | 69.98 (4.68%) | 70.09 (4.83%) |
| QK–Softmax–AV | 358.24 (24.46%) | 359.15 (24.04%) | 359.50 (24.77%) |
| O projection | 58.57 (4.00%) | 55.98 (3.75%) | 56.08 (3.86%) |
| Post-attention residual＋RMSNorm | 53.74 (3.67%) | 53.67 (3.59%) | 53.68 (3.70%) |
| Gate/Up＋SwiGLU | 289.83 (19.79%) | 340.77 (22.81%) | 341.59 (23.54%) |
| Down | 157.77 (10.77%) | 157.65 (10.55%) | 158.73 (10.94%) |
| Final residual | 0.07 (0.00%) | 0.07 (0.00%) | 0.07 (0.01%) |
| KV carrier conversion | 3.15 (0.21%) | 3.17 (0.21%) | 3.18 (0.22%) |
| KV append DMA | 6.00 (0.41%) | 6.36 (0.43%) | 6.02 (0.42%) |
| Block orchestration | 1.07 (0.07%) | 1.05 (0.07%) | 1.07 (0.07%) |
| Layer bookkeeping | 0.66 (0.04%) | 0.67 (0.04%) | 0.66 (0.05%) |
| Stage-boundary bookkeeping | 0.40 (0.03%) | 0.40 (0.03%) | 0.41 (0.03%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.69 (3.32%) | 48.88 (3.27%) | 48.87 (3.37%) |
| Embedding | N/A: outside single-layer scope | N/A: outside single-layer scope | N/A: outside single-layer scope |
| Final model RMSNorm | N/A: outside single-layer scope | N/A: outside single-layer scope | N/A: outside single-layer scope |
| LM head＋greedy（不含 final norm） | N/A: outside single-layer scope | N/A: outside single-layer scope | N/A: outside single-layer scope |
| Host–DSP 边界 | 292.19 (19.95%) | 298.94 (20.01%) | 259.92 (17.91%) |
| 完整 Host wall | 1464.74 (100.00%) | 1494.24 (100.00%) | 1451.12 (100.00%) |

E2E token/s: N/A in L32-0022. No complete16-layer frontend, continuous-generation
speed or PPL measured for repaired rotations. Do not extrapolate single-layer
throughput. Scope completed here: prior-version reproduction, arithmetic repair,
singlelayer0/7/15 and continuous3 correctness, fixed10 single-layer speed gates.
Next extension is full16 numerical validation, then actual frontend/E2E if it
passes; keep the same numerical/physical/speed gates.

Evidence includes43 CLI attempts:13exit0,30retained exit1;686token-boundary
executions including660formal/600measured. One old-native continuous3 numerical
failure was deliberately reproduced. No native crashes or failed builds, no new
model files, no default or quality promotion. Prior0020/0021 ledgers and their
files are independently reverified unchanged. Other Llama branch and Qwen frozen.
