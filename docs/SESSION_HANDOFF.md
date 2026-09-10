# L32-0003 W4A8 PPL running

Bootstrap /home/daniuniu/work/llama32-htp and read four authority files.
Active L32-0003 no-rotation owns current source (status has HEAD); rotation
still e91ff0f with W4A16 common code only. Do not run a second device workload.

W4A16 L32-0002 sealed functional pass, PPL+16.26% quality failed. Never rerun
completed work to recover context. A8 now exact-pass1/3/16-layer prefill+decode,
cache gates and16-step independent token/code oracle. Current text repeats Sleep.
Generation and ongoing PPL runner:
results /mnt/d/llm_exp/results/llama32-htp/l32-0003/device-frontend-a02
package /mnt/d/llm_exp/models/llama32-htp/l32-0003/frontend-a01
reference results/l32-0003/frontend-reference-a01.
The existing Python runner is evaluating2048 targets (128 EN/ZH docs). It writes
evaluation.stdout.txt and final result.json on completion; do not duplicate.
Current Codex exec session22724 runs it. Its generation result is already saved.

After completion audit all2048 unique target records/physical counters and
compare first NLLs against nll-arithmetic-oracle.json. If complete, seal results,
update source validation summary/model JSON and authority, transfer ownership
before propagating common commits990bf61..current to rotation by cherry-pick.
No baseline promotion. W4A16 quality remains failed; A8 has no quality threshold.
All failed attempts retained. Native build/source-core55f52e8; newer docs/tools
commits do not change binaries. See CONTEXT and docs/LLAMA32_W4A8.md.
