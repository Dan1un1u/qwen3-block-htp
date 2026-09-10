# Active Llama W16A16 port

Bootstrap Llama authority; active L32-0001 on /home/daniuniu/work/llama32-htp,
branch codex/llama32-no-rotation. Read docs/experiments/L32-0001.md.
Original model at /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin verified.
Independent host math and single layer0/7/15 prefill/decode pass; stack3 pass.
Current source 2eb88bc001731d70485820adf50a8bd94f594970 (clean/synced at checkpoint).
Building16 and exporting stack16-a01. Resume full16 replay via
tools/run_llama32_stack.py, then implement tied-head generation and heldout PPL.
No Qwen research or source reset. Rotation branch remains initial, unmodified.
Results /mnt/d/llm_exp/results/llama32-htp/l32-0001; packages corresponding model root.
