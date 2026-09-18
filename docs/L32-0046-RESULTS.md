# L32-0046 ordinary A8 completion

Same latest0045 W4 weights/scales, FP32 residual, embedding/head/KV, no rotations. Only Down representation differs. Five short and ten balanced formal paired repeat10 rounds. All independent selected0/13/27, chain3, full43 fixed/free-greedy hidden/IDs/codes and prefill KV gates pass. 8MiB VTCM, one HMX owner, one RPC/token, no timed intermediate DDR/spill. No model-quality acceptance or automatic baseline promotion.

| Recipe | Prefill TPS | Decode TPS |
|---|---:|---:|
| SP2 | 1186.505056 | 23.717208 |
| A8 | 1206.676361 | 23.726358 |

Complete Host wall; loading/tokenizer/audit I/O excluded. Project-owned prompt, not named-dataset performance.

{
  "prefill": {
    "a8_over_sp2_wall_ratio": 0.9832835827715714,
    "ci95": [
      0.9816984175985882,
      0.9848798752203622
    ]
  },
  "decode": {
    "a8_over_sp2_wall_ratio": 0.9996143649802196,
    "ci95": [
      0.9982838334288546,
      1.0007995899350064
    ]
  }
}
