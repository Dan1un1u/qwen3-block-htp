# L32-0007 closed: matched W16A16 and W4A16 device PPL

No active experiment or background job. Both source branches remain unchanged,
clean and synchronized. Next experiment L32-0008 awaits user direction.
Current sealed native cfd9fee, source b9932bd, W4 OPT2 and original W16 carriers.
No native code, weights, qparams or rotation changes.

Fresh complete16-layer hardware evaluation, one run per recipe on the SAME frozen
128 Wikipedia documents (64EN/64ZH),M64 context plus16 targets,2048 targets each.
Original BF16 teacher26.6979986; FP16 software26.6988837.
W16 device26.6917492 (-0.0234% vs BF16); EN17.7645866,ZH40.1050412.
W4 device31.0391011 (+16.2600%); EN20.6041358 (+15.8353%),ZH46.7588551 (+16.6863%).
W16 overall/language gates pass including paired-document bootstrap95% upper;
W4 fails. Overall ratio CI W16[.998013,1.001480],W4[1.123305,1.212812].
W4 all2048 target codes and NLL exactly equal sealed prior hardware evidence.
No additional quality degradation from L32-0004/0005/0006 speed work.
W16 old25.104770 figure used a DIFFERENT eight-document diagnostic, do not compare.

4096 unique scored pairs verify frozen sample/step/target,finite NLL,vocab128256,
exact8MiB and zero timed intermediate DDR/spill. Two8-step generation smokes pass.
Four successful DSP processes,4112 total boundaries. No native failures/repeats.
Tooling syntax/legacy ledger shape and report token_id-vs-target_token confusion
were repaired with original attempts retained; final postprocessing reused valid
raw hardware results, no weakened gate. See recovery_notes.txt.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0007, SUMMARY.md,summary.json,
26-file ledger docs/experiments/L32-0007-evidence-sha256.json.
PPL is lightweight short-context EN/ZH Wikipedia, not broad/long-context acceptance.
W4 baseline quality remains unaccepted; A8 remains unusable/unmodified.
Latest formal SPEED evidence remains L32-0006, current values in status.
Qwen frozen, Llama rotations unsupported; original BF16 inputs read-only.
