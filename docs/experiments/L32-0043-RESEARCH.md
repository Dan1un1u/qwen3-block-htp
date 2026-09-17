# L32-0043 primary-source scale comparison

Retrieved 2026-09-17. Pinned Qualcomm ai-hub-models v0.62.2. Same Snapdragon8Elite QRD, NPU, declared context4096; ratios within matched backend and precision labels. These are official end-to-end llm_metrics, not individual-layer kernels. Exact prompt/decode lengths and treatment of frontend differ from our M64+15/cache80; no absolute head-to-head claim. The w4 label is preserved as published and is NOT reinterpreted as our W4A8-SP2.

| Backend/precision | 1B prefill/decode | 3B prefill/decode | 3B/1B prefill/decode | Projected ours3B |
|---|---:|---:|---:|---:|
| geniex_qairt w4 | 1383.589333 / 31.243782 | 672.519690 / 12.888389 | 0.486069 / 0.412511 | 1135.085573 / 18.868384 |
| genie w4a16 | N/A / 58.379513 | N/A / 28.378454 | N/A / 0.486103 | N/A / 22.234529 |
| geniex_qairt w4a16 | 3410.332241 / 65.745355 | 1256.182775 / 27.651214 | 0.368346 / 0.420580 | 860.175261 / 19.237507 |

Sources:
- https://raw.githubusercontent.com/qualcomm/ai-hub-models/v0.62.2/src/qai_hub_models/models/llama_v3_2_1b_instruct/perf.yaml (SHA256 6a44506b034911c336666ae500fe264de2a0a2319b3d6f26a6be78629d819a71)
- https://raw.githubusercontent.com/qualcomm/ai-hub-models/v0.62.2/src/qai_hub_models/models/llama_v3_2_3b_instruct/perf.yaml (SHA256 7114c2cfa2ec4dd2f03a2d74300a4a09583aca1aeb816cf65f3dd09042359253)

Quant.npu https://arxiv.org/html/2605.20295v1 Table12 reports Llama3.2-3B but no Llama1B paired speed. Its3B SM8750 HellaSwag M64+42 is809.15/28.04tps; useful external context, unsuitable for deriving1B/3B ratio or direct timing comparisons with M64+15.

Inference: official matched ratios imply ours3B860–1135 prefill and18.87–22.23decode. Select near-upper-envelope engineering goal1100/22 before candidates, not claimed hardware limit. Other backend differences can readily alter these ratios. Standard exact numerical/physical gates and paired95% CI upper<=1.10 unchanged. Finalmeans define absolute goal; all10formalcycles mandatory.
