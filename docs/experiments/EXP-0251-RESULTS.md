# EXP-0251 hardware speed of wide-score repair

Only the no-R3 wide-score repair is numerically eligible and passes the10% speed gate. Plain dense HMX R3 + wide repair still fails the independent whole-layer gate and has no formal timing result.

Device PJZ110 / SM8750 / HTP V79. Frozen EXP0247 C64 per-output-channel W4 packages (C64 is prefix length, not weight group size), native HMX weight.n. Five short + ten rotated paired formal rounds, each repeat1 and repeat10:2970 timed layer RPCs. M64 prefill then8 real M1 cache appends, one layer0. No discarded rounds.

|Scope|Old us|Wide repair us|Paired time change|95% change interval|
|---|---:|---:|---:|---|
|repeat1_prefill|1782.344|1556.693|-9.33%|[-14.96%, +0.88%]|
|repeat1_decode|962.803|957.438|+0.44%|[-7.03%, +2.98%]|
|repeat10_prefill|1542.820|1525.797|-0.41%|[-3.44%, +2.47%]|
|repeat10_decode|952.656|960.633|+0.67%|[-0.38%, +2.02%]|

Host wall cells are separate medians of paired-round means; paired changes use the median of within-round ratios, so dividing displayed median times gives a different percentage. Primary repeat10 confidence intervals are entirely below+10% and contain zero; no demonstrated speed penalty or speedup. No extra rounds required.

Numerical audit: all45 arm/step raw-QK, probability and AV boundary sets match the independent NumPy integer oracle exactly, including cached decode. W0 all9 outputs exactly match untimed scalar-wide reference and repeated capture; C0 reproduces sealed EXP0248 control. K cache independently packed and V cache matched to the oracle. Formal output hashes match these audited references. Wide exponent range/mask/future-max/shift-invariance proof retained from sealed EXP0250.

R3 limit: dense matmul itself passes1FP16ULP+min-normal and Q/K differences<=1code, but prefill whole-layer max11LSB/cosine0.9927222 and several decode steps3-7LSB/cosine0.98547-0.99056 fail the unchanged2LSB/cosine0.999 gate. Exact downstream arithmetic from actual captured Q/K does not imply equality to Float64 rotation. No mode5 refinement or scalar fallback was timed.

Every timed invocation requests/acquires8MiB VTCM, peak6,682,752bytes; zero intermediate DDR/spill and audit exports;7 native W4 projections and1RPC. Additive ledger exactly closes. Complete repeat1/repeat10 additive module, overlapping engine/DMA/wait and physical counters in FULL_PROFILE.md; stable recipe tables in USER_PROFILE.md.

Recovery: first build hit a Host telemetry printf type mismatch for the new uint32 mode; corrected U32 emission without arithmetic changes and rebuilt successfully. Initial collector used system Python missing NumPy; no device call occurred in that attempt, reran using existing analysis environment. Both failed logs retained. No data deletion, threshold relaxation, quantizer or weight change.

Deployment scope: compiled layer0 ABI115; wide mode capacity<=72, testedM64+eightM1 only. Do not treat this experimental cache as a full-model runtime. Other recipes frozen, no baseline promotion. Device PPL and full-model E2E token/s:N/A. Software EXP0250 PPL is not a device measurement.

Source 27972a15d770ada194f712805b1be01e1a44dd68; runtime 910f6f95c8d144194b0711c974b60c7159d3931b.
