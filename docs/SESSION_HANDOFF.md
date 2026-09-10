# L32-0001 completed; next W4A16

Run Llama bootstrap for /home/daniuniu/work/llama32-htp and read four authority
files. No active experiment. Next L32-0002, per-channel W4A16 adaptation, already
authorized in the user's W16A16 -> W4A16 -> W4A8 order. Register before stateful work.

No-rotation a5d2e8ca3f2f98787b36652db3d8f6f7e74d11f5
Rotation 3e0076d6d06178b67adaf514d3bf3b7d3fbe604b
Both clean/synced and common code identical, only branch.json differs. Qwen3
frozen unchanged. Do not repeat organization, checkpoint downloads or W16 port.

Original /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin. Generated packages
/mnt/d/llm_exp/models/llama32-htp/l32-0001, final frontend-a02. Results
/mnt/d/llm_exp/results/llama32-htp/l32-0001, validation_summary.json authoritative.
Evidence ledger SHA256 6791ea14bf6453e5fd02da9827cdcab71976c00d6f8d39f63d90303fde7b3768.

Independent math and device single3/whole16 stack pass. Text matches16 tokens.
Tiny EN/ZH PPL device25.1047703 / direct-original BF16 25.0220618 (+0.33054%).
Initial frontend-a03 result's BF16 field is superseded by direct BF16 reload in
validation_summary.json; FP16 and device fields unchanged. See experiment record.
Functional speed395.4202/5.7742 tok/s, not formal profiling; DSP RoPE is scalar.
W4A16/A8 launches unvalidated/rejected. Implement fresh W4A16 export/calibration
and extend fixed independent PPL acceptance beyond this128-token integration set.
