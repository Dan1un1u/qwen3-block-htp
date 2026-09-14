# L32-0028 closure: no-rotation paper speed baseline selected

User selects L32-0018 W4A8 SP2mode8+FP32 residual,R3/R4 OFF, historical
2069.7025566101083/42.51010980249135tok/s as primary Llama paper speed baseline.
Supersedes0027 rotated primary selection; rotation acceptance/failures retained
as supplementary evidence. This does not promote fixed-scale text/PPL quality.

Source0ae07a92e4d99fd3fd23c89adc3ce4a2b44f7b4f,codex/llama32-no-rotation,clean/pushed; native src/include unchanged
from0f7d083. New source docs/PAPER_NO_ROTATION_BASELINES.md/json pin original
measured0018e3d065,exact command,evidence hashes and scope. Five hot files match
0018; later changes guarded rotations or restored trials. Rotation branch common
commits patch-equivalent except branch default; no missed validated OFF kernel.
All1810018+1511Qwen0268 evidence entries rehashed. New Qwen kernel changes are
not timed on Llama and do not inherit this historical throughput.

Active none,next29. No new Llama model/hardware/native/PPL. Qwen ownsEXP0269:
FP32 singlelayer reference/KV passes,final short speedgate fails; no formal or
fullmodel expansion. Original Qwen paper2030.24/48.17 retained. Reports and audit
in/mnt/d/llm_exp/results/qwen3-block-htp/exp0269;ledgerSHA25620d3ddb9e1152e35e714f45b159f26312e5a79177cae907c31b7952a8af26ff5. Next discuss Qwen prefill FP32 Norm/O cost or draft
paper from selected historical results; no automatic new Llama experiment.
