# L32-0022 complete: existing R4 numerical error repaired
No active experiment. Source closure 9209ac60265e8d6daf966e02a18fe82e7b59f024; native/profiled 3fb9a4f694e36552c1d7f75fc40de59007cfc8cf.
Read docs/LLAMA32_ROTATION_NUMERICAL_REPAIR.md and docs/experiments/L32-0022.md.
User requires numerical correctness, rollback if latest optimization introduced error.
Restored exact0020 native R4 at2459593, rebuiltN3, reused sealed0021 chain3
weights/inputs/reference unchanged. Outputs byte-identical to0021 in both phases:
minrowcos .791505/.911935 fail.0020 had never tested chain3. Thus rollback alone
cannot fix this fixture;0021 scheduling change did not introduce its failure.
Retained fast0021 pipeline and moved FP16 H512 normalization into dense weight
entries with unity output converter scale. Same mathematical reference, unchanged
W4/SP2/calibration/FP32 residual and dense GEMM calls/layout. Generate normalized
bits with singleXOR at original constant generation cost. No butterfly.
Final singlelayer0/7/15 actual conditional-tail exact, component<=1ULP+minnormal,
idealcos>.9999999997. Continuous3 prefill mincos .999999998751,relativeL2
4.64915e-7; decode bitexact, all KV values/prefix/structure exact. Original exact
CLI failures retained for tiny prefill FP32 differences. No numerical gate relaxed.
8,229,344B VTCM, no timed activation DDR/spill. finish/layout frames88/24B scalar;
run_dense_r4 vector stack save is a traced zero constant, not activation.
Fixed10 cyclic three-arm OFF/repairedboth/sealed0021both,1warmup+10measured
pairs/process. OFF1945.139/1464.737us; repaired2065.881/1451.117us; old2118.069/
1494.236us. Both vsOFF +6.207%/-.930%,95%CIupper+8.768%/+2.164% PASS.
Matched old Host wall improves2.464%/2.886%; mostly Host-DSP boundary, not evidence
of faster R4 matrix compute (GateUp/SwiGLU ~unchanged). Preserve full ledger.
43CLI,686RPC,660formal/600measured;13exit0,30retainedexit1. No crashes/buildfails.
No full16 rotation/frontend/E2E/PPL; no default/quality promotion. Next full16
independent numerical validation then actualfrontend/E2E if it passes.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0022/evidence-ledger-checkpoint-a01.json
SHA256 1ce076bf6ff75fbc54f4a9c6de1516b414311cf6895d20640d26011681dbbcf3;249files,6reusedmodelmanifests.
All prior0020(371files)/0021(822files) ledger entries and model manifests reverified.
Qwen and other Llama branch remain frozen. BuildN3 seal is native/profiled HEAD;
closure adds docs only: rebuild from current sourceHEAD before next device use.
