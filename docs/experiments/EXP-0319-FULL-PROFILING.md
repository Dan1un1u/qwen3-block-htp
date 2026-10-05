# Complete profiling comparison
Same binary, frozen uniform INT16 Down weights; long option control196511 versus candidate458655. Short5paired repeat1; formal10paired repeat10. All profiles are complete28layer calls. Prefill rows aggregate32chunks per2048prompt; decode rows are per-token averages over64continuous calls. Values below are medians across rounds. Engine-work/wait rows overlap and must not be summed. Full numeric counters including all physical/MLP/projection fields are retained in complete-profile-counters.csv/json. Historical W16A16/W4A16 at this2048+64scope are N/A (not measured), not extrapolated.

## short prefill

|Metric|Control (us)|Candidate (us)|Change|
|---|---:|---:|---:|
|host_wall_ns|1304388.178|1272302.397|-2.46%|
|scan_cache_stage_ticks|45530.365|45587.292|+0.13%|
|scan_cache_append_ticks|15388.229|15342.604|-0.30%|
|block_orchestration_ticks|1001.719|999.271|-0.24%|
|layer_bookkeeping_ticks|550.052|549.844|-0.04%|
|scan_dynamic_attention_ticks|433511.250|395661.562|-8.73%|
|total_ticks|1268050.625|1230212.240|-2.98%|
|invocation_ticks|1269294.583|1231457.917|-2.98%|
|runtime_setup_ticks|1241.823|1240.938|-0.07%|
|runtime_teardown_ticks|1165.104|1163.646|-0.13%|
|stage_boundary_ticks|63.490|62.031|-2.30%|
|ledger_unattributed_ticks|0.000|0.000|N/A zero denominator|
|input_stage_ticks|12.500|12.552|+0.42%|
|metadata_stage_ticks|8633.854|8625.208|-0.10%|
|input_norm_ticks|66852.344|66870.000|+0.03%|
|qkv_projection_ticks|247784.323|247490.833|-0.12%|
|qk_norm_rope_ticks|9.167|9.010|-1.70%|
|attention_ticks|437169.844|399322.708|-8.66%|
|o_projection_ticks|68068.802|67966.146|-0.15%|
|post_attention_residual_ticks|66565.312|66563.906|-0.00%|
|post_attention_norm_ticks|36.250|35.260|-2.73%|
|gate_up_ticks|222660.469|222701.094|+0.02%|
|activation_ticks|0.000|0.000|N/A zero denominator|
|down_ticks|122117.396|122476.719|+0.29%|
|final_residual_ticks|59.948|60.208|+0.43%|
|output_stage_ticks|0.000|0.000|N/A zero denominator|
|generation_embedding_ticks|1905.990|1901.927|-0.21%|
|generation_final_norm_ticks|15.000|14.635|-2.43%|
|generation_lm_head_ticks|5188.021|5178.958|-0.17%|
|u8_attention_k_pack_ticks|262192.552|264532.396|+0.89%|
|u8_attention_v_pack_ticks|108829.583|108819.062|-0.01%|
|u8_attention_qk_hmx_ticks|36567.604|37530.417|+2.63%|
|u8_attention_softmax_ticks|706490.208|710577.083|+0.58%|
|u8_attention_av_hmx_ticks|70938.073|71686.146|+1.05%|
|u8_attention_pipeline_wait_ticks|15023.646|17345.729|+15.46%|
|host_dsp_boundary_ns|35149.009|41355.053|+17.66%|

## short decode

|Metric|Control (us)|Candidate (us)|Change|
|---|---:|---:|---:|
|host_wall_ns|29172.200|26923.826|-7.71%|
|scan_cache_stage_ticks|2300.493|2306.301|+0.25%|
|scan_cache_append_ticks|225.640|225.209|-0.19%|
|block_orchestration_ticks|30.413|30.419|+0.02%|
|layer_bookkeeping_ticks|16.532|16.501|-0.19%|
|scan_dynamic_attention_ticks|10328.552|8060.291|-21.96%|
|total_ticks|28448.612|26203.662|-7.89%|
|invocation_ticks|28486.575|26241.745|-7.88%|
|runtime_setup_ticks|38.046|38.006|-0.10%|
|runtime_teardown_ticks|35.835|35.821|-0.04%|
|stage_boundary_ticks|1.851|1.860|+0.48%|
|ledger_unattributed_ticks|0.000|0.000|N/A zero denominator|
|input_stage_ticks|0.363|0.357|-1.57%|
|metadata_stage_ticks|255.824|250.020|-2.27%|
|input_norm_ticks|364.062|364.092|+0.01%|
|qkv_projection_ticks|2487.769|2491.510|+0.15%|
|qk_norm_rope_ticks|0.332|0.323|-2.70%|
|attention_ticks|10334.250|8066.141|-21.95%|
|o_projection_ticks|1363.783|1363.094|-0.05%|
|post_attention_residual_ticks|369.134|368.764|-0.10%|
|post_attention_norm_ticks|1.147|1.147|+0.00%|
|gate_up_ticks|6246.214|6243.010|-0.05%|
|activation_ticks|0.000|0.000|N/A zero denominator|
|down_ticks|3491.448|3498.878|+0.21%|
|final_residual_ticks|1.785|1.799|+0.73%|
|output_stage_ticks|0.000|0.000|N/A zero denominator|
|generation_embedding_ticks|5.946|5.865|-1.37%|
|generation_final_norm_ticks|14.688|14.753|+0.45%|
|generation_lm_head_ticks|3202.324|3212.028|+0.30%|
|u8_attention_k_pack_ticks|3217.412|3248.311|+0.96%|
|u8_attention_v_pack_ticks|6210.729|6218.193|+0.12%|
|u8_attention_qk_hmx_ticks|1379.235|1469.934|+6.58%|
|u8_attention_softmax_ticks|4397.352|4462.648|+1.48%|
|u8_attention_av_hmx_ticks|907.545|863.407|-4.86%|
|u8_attention_pipeline_wait_ticks|21.715|179.545|+726.81%|
|host_dsp_boundary_ns|686.475|703.898|+2.54%|

## formal prefill

|Metric|Control (us)|Candidate (us)|Change|
|---|---:|---:|---:|
|host_wall_ns|1312943.034|1266531.351|-3.53%|
|scan_cache_stage_ticks|48294.935|47137.867|-2.40%|
|scan_cache_append_ticks|15239.182|14920.904|-2.09%|
|block_orchestration_ticks|999.365|999.461|+0.01%|
|layer_bookkeeping_ticks|557.320|555.385|-0.35%|
|scan_dynamic_attention_ticks|434973.852|396366.677|-8.88%|
|total_ticks|1289570.042|1241913.378|-3.70%|
|invocation_ticks|1290823.294|1243166.357|-3.69%|
|runtime_setup_ticks|1255.831|1254.044|-0.14%|
|runtime_teardown_ticks|1165.792|1162.669|-0.27%|
|stage_boundary_ticks|63.242|62.932|-0.49%|
|ledger_unattributed_ticks|0.000|0.000|N/A zero denominator|
|input_stage_ticks|12.471|12.536|+0.52%|
|metadata_stage_ticks|9037.698|8997.896|-0.44%|
|input_norm_ticks|66868.982|66863.203|-0.01%|
|qkv_projection_ticks|251252.276|249839.893|-0.56%|
|qk_norm_rope_ticks|9.383|9.352|-0.33%|
|attention_ticks|438642.091|400032.576|-8.80%|
|o_projection_ticks|68163.909|67864.135|-0.44%|
|post_attention_residual_ticks|66590.510|66577.911|-0.02%|
|post_attention_norm_ticks|35.318|35.240|-0.22%|
|gate_up_ticks|233725.169|229782.807|-1.69%|
|activation_ticks|0.000|0.000|N/A zero denominator|
|down_ticks|125834.255|123958.701|-1.49%|
|final_residual_ticks|60.195|60.622|+0.71%|
|output_stage_ticks|0.000|0.000|N/A zero denominator|
|generation_embedding_ticks|1953.008|1931.312|-1.11%|
|generation_final_norm_ticks|15.380|15.203|-1.15%|
|generation_lm_head_ticks|6079.539|5805.133|-4.51%|
|u8_attention_k_pack_ticks|262221.544|264559.880|+0.89%|
|u8_attention_v_pack_ticks|108892.760|108827.648|-0.06%|
|u8_attention_qk_hmx_ticks|36674.865|37726.094|+2.87%|
|u8_attention_softmax_ticks|706414.940|710577.492|+0.59%|
|u8_attention_av_hmx_ticks|71281.263|71884.372|+0.85%|
|u8_attention_pipeline_wait_ticks|14398.297|16905.138|+17.41%|
|host_dsp_boundary_ns|22651.074|23201.026|+2.43%|

## formal decode

|Metric|Control (us)|Candidate (us)|Change|
|---|---:|---:|---:|
|host_wall_ns|29470.141|27243.065|-7.56%|
|scan_cache_stage_ticks|2346.607|2350.796|+0.18%|
|scan_cache_append_ticks|219.544|217.926|-0.74%|
|block_orchestration_ticks|30.453|30.480|+0.09%|
|layer_bookkeeping_ticks|16.564|16.585|+0.13%|
|scan_dynamic_attention_ticks|10344.901|8079.074|-21.90%|
|total_ticks|28815.390|26541.077|-7.89%|
|invocation_ticks|28853.481|26579.032|-7.88%|
|runtime_setup_ticks|38.009|38.013|+0.01%|
|runtime_teardown_ticks|35.856|35.903|+0.13%|
|stage_boundary_ticks|1.855|1.854|-0.05%|
|ledger_unattributed_ticks|0.000|0.000|N/A zero denominator|
|input_stage_ticks|0.363|0.362|-0.25%|
|metadata_stage_ticks|249.427|251.490|+0.83%|
|input_norm_ticks|364.118|364.133|+0.00%|
|qkv_projection_ticks|2535.631|2537.689|+0.08%|
|qk_norm_rope_ticks|0.325|0.328|+0.86%|
|attention_ticks|10350.563|8084.917|-21.89%|
|o_projection_ticks|1377.349|1379.432|+0.15%|
|post_attention_residual_ticks|369.057|369.071|+0.00%|
|post_attention_norm_ticks|1.157|1.156|-0.10%|
|gate_up_ticks|6398.911|6404.080|+0.08%|
|activation_ticks|0.000|0.000|N/A zero denominator|
|down_ticks|3558.399|3559.371|+0.03%|
|final_residual_ticks|1.790|1.795|+0.32%|
|output_stage_ticks|0.000|0.000|N/A zero denominator|
|generation_embedding_ticks|5.819|5.834|+0.27%|
|generation_final_norm_ticks|14.629|14.637|+0.05%|
|generation_lm_head_ticks|3264.966|3271.840|+0.21%|
|u8_attention_k_pack_ticks|3215.961|3251.269|+1.10%|
|u8_attention_v_pack_ticks|6212.191|6217.986|+0.09%|
|u8_attention_qk_hmx_ticks|1376.588|1469.946|+6.78%|
|u8_attention_softmax_ticks|4394.566|4460.262|+1.49%|
|u8_attention_av_hmx_ticks|905.667|868.402|-4.11%|
|u8_attention_pipeline_wait_ticks|20.300|165.925|+717.35%|
|host_dsp_boundary_ns|627.480|653.738|+4.18%|

## Physical and correctness

20160 collected profile calls: exact8MiBVTCM, zero intermediateDDR/spill throughout. Maximum recorded unattributed/total ratio=0. Cache and nonzero boundary hashes are validated in audit-three and supplementary audit results; ordinary timing hashes are not numerical evidence.

Failed four-client candidate is excluded: HMX precise exception at last prefill chunk, retained in audit-candidate and failed-audit-logcat.txt. Root cause not established; do not claim a hardware limit.
