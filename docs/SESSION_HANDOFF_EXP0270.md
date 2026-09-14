# EXP0270 closure — exact FP32 path; prefill gate still fails

Source codex/exp-0270-fp32-norm-pipeline at 8268f8f7f5234902f7dd1dffc7c1a6194bae4b7a, tested native 645b0cdc4d6787dcde7bade9af59c3de3d6d8451. Source clean/pushed; final commit only report/script/docs, native unchanged after device. Active none,next271. User's continue optimization was implemented as three bounded candidates under EXP0270, preserving PC083 numeric/physical/10percent gates. Llama frozen and original two paper rows unchanged.

Retain C1 eight-row register Norm reduction (four columns/vector, ascending sums, output sums VTCM) and C3 invariant raw-output bias tables (three256B tables in existing2KiB VTCM bias reserve, initialized once per HMX command). No activations/partial sum arrays on stack in tested affected assembly. Uniform constants and row inverse-RMS coefficient splats only in generic Norm stack. C2 four-row output attempts compiler-spilled masks/codes and were stopped before device; repaired two-row exact but slower101us Norm vs75us C1, reverted. C3 compiler const-API error fixed and retained evidence. All native hardware attempts numerically passed.

Final independent selectedlayers0/14/27 M64+decode1 exact399360 FP32 values and actual Q/K,AV,postnorm,Gate/Up,SP2 planes,prefill physicalKV. repeat10 deterministic. Raw signed24/mergedsigned32 bounds pass.8MiB requested/acquired,peak6682752,zero timed tensorDDR/spill/unattributed.33 CLI/372 profiles. All546 oldEXP0269 evidencefiles and three model manifests reverified. Reuse exp0269 models only, no new quantization.

Five fixed short AB/BA,repeat10 primary,20000 bootstrap seed270:
prefill1357.2636->1552.975us,ratio1.144195571,95CI[1.101443554,1.186659171],fail;
decode1017.1792->1059.21244us,ratio1.041323338,CI[1.015607006,1.082241126],pass.
No formal10/chain3/full28/E2E/PPL, no resampling or gate changes.
Nonpaired diagnostic C1->C3 Down prefill162.81->142.12us,O decode51.82->46.39us. O prefill did not improve. These are not formal gains vs EXP0269.

Main remaining prefill additions vs samebinary integer control: inputNorm54.99us,postnorm55.50us,O29.39us,I/O27.34us,Host-DSPboundary53.47us,offset byDown19.08us faster and no final U8residual6.52us. Need distinguish one-layer boundary costs from fullmodel; cannot extrapolateE2E. Suggested next discussion: common FP32 residual physical layout for O/Down production and Norm consumption to eliminate transpose, maintaining exact ordered arithmetic. No new experiment started or performance eligibility assumed.

Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0270; 271files,101661814bytes; ledger 1ec0df4c36fc44ad0cfc79a7e3503644fe8acd3135ab75f968074c0732960717. Read SUMMARY.json,REPORT.md,layer-short.json,single_gate.json,candidate_selection.json,c3 assembly and source_closure.json. Source scripts common/device/audit/report_exp0270.py; report fully reconstructs raw counters/medians,CI and ledgers without hardware. Existing runtime-l1 seal is testedhead, not finaldochead; futuredevice requires new experiment/preflight/build.

Paper Qwen EXP0268 SP2 integer2030.238/48.173tok/s; Llama L32-0018 SP2FP32residual2069.703/42.510tok/s, historical M64+15 fullmodel. Both quality unaccepted. New Qwen C1/C3 not measured on Llama. No baseline replacement.
