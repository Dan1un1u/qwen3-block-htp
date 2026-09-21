# L32-0062 complete additive module table
Microseconds; decode per token. Fullmodel,16layers.
| Module | Control prefill | Folded prefill | Control decode | Folded decode |
|---|---:|---:|---:|---:|
|I/O、metadata|123.222|122.955|120.391|120.177|
|Input RMSNorm|1278.382|1278.641|868.973|869.153|
|QKV＋RoPE|3870.875|3872.905|1668.091|1669.466|
|QK–Softmax–AV|6296.307|6307.403|6344.319|5477.910|
|O projection|1592.603|1589.509|908.207|900.279|
|Post-attention residual＋RMSNorm|1338.145|1332.816|877.136|877.808|
|Gate/Up＋SwiGLU|5469.831|5457.082|4781.061|4779.269|
|Down|3146.678|3151.203|2608.655|2607.820|
|Final residual|2.451|2.394|0.977|0.975|
|KV carrier conversion|29.864|29.793|50.111|50.119|
|KV append DMA|114.688|114.564|90.366|90.388|
|Block orchestration|23.960|23.945|16.513|16.517|
|Layer bookkeeping|13.404|13.239|9.467|9.471|
|Stage-boundary bookkeeping|6.780|6.742|1.241|1.242|
|DSP unattributed|0.000|0.000|0.000|0.000|
|Runtime setup/teardown|92.533|91.427|51.192|51.202|
|Embedding|62.777|62.934|1.580|1.580|
|Final model RMSNorm|55.245|58.208|55.407|55.570|
|LM head＋greedy（不含 final norm）|3301.295|3305.302|3297.600|3304.012|
|Host-DSP boundary|772.429|789.983|640.807|648.477|
|Complete Host wall|27591.467|27611.045|22392.095|21531.434|
