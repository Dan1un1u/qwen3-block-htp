# EXP0237 artifact-retirement results

Deleted 6436 explicitly inventoried regular payload files. Actual D-drive free-space increase 174155087872 bytes (162.195 GiB); available space 49.015 -> 211.210 GiB. Main models cleanup 152.591 GiB; legacy mllm-v2 quantized weights 9.604 GiB. Logical byte sums count hardlinks; free-space change is measured separately.

Retained original checkpoint shards plus 11 model packages verified, 24012 manifest/shard entries with zero mismatches. F16 canonical, C64, AR-P, W4A16 selected device baseline and W4A8 selected device dependencies remain intact. No source/toolchain/data/report deletion. Remaining legacy files above100MiB are compiler logs, retained outside weight cleanup scope.

Evidence: /mnt/d/llm_exp/results/qwen3-block-htp/exp0237. Original experiment hashes/conclusions remain immutable; inventory.json and legacy_inventory.json enumerate the exact payload paths now physically retired and no longer locally replayable. Retiring grouped diagnostics is a storage decision, not a revision to their measured PPL. Do not rerun their old artifact validators as if the payloads were retained; use the availability ledger first.

Evidence ledger SHA256: 9ea8741d9295f8cd258aebc689148bd147fa78fe112c3dc1e0cbb501487c1ab2. Source: 2141e6f94fcdfb0679dc2f37c089e68814b82a07. No baseline promotion. Next authorized: EXP238 Qronos, then EXP239 OmniQuant under their frozen protocols.
