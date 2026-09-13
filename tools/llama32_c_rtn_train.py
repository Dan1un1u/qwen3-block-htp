#!/usr/bin/env python3
"""Run pinned SpinQuant C training with explicit native-contract adaptations.

Reference code is separately hashed under build/l32-0013/reference. It is not
imported from, nor written into, the user's rotation-quant checkout.
"""
import argparse, json, os, subprocess, sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "build/l32-0013/reference/repos/SpinQuant"
RESULTS = Path("/mnt/d/llm_exp/results/llama32-htp/l32-0013")
MODELS = Path("/mnt/d/llm_exp/models/llama32-htp/l32-0013")
PYTHON = "/home/daniuniu/work/rotation-quant/.venv/bin/python"

def worker(stage):
    sys.path.insert(0, str(REFERENCE))
    import torch
    from transformers import Trainer, set_seed
    from utils import quant_utils
    from train_utils import rotation_calibration as rc
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    audit=json.loads((RESULTS/"rotation-fp32-selective-fp64-audit.json").read_text())
    assert audit["pass_gate"]
    from llama32_rotation_math import rotated_weight_fp32
    from train_utils.quant_linear import QuantizeLinear
    QuantizeLinear.rotated_weight=rotated_weight_fp32
    # Native W4 uses [-7,7]. Keep ordinary A8 [-128,127].
    def ste(ctx, x, scale, maxq):
        hi = int(maxq)
        return scale.to(x.device) * (x / scale.to(x.device)).round().clamp(-7 if hi == 7 else -hi-1, hi)
    quant_utils.STEQuantize.forward = staticmethod(ste)
    class NativeChannelScale(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale, maxq):
            assert int(maxq) == 7
            scaled = x.float() / scale
            ctx.save_for_backward(scaled)
            return (scale * scaled.round().clamp(-7, 7)).to(x.dtype)
        @staticmethod
        def backward(ctx, grad):
            (scaled,) = ctx.saved_tensors
            term = torch.where(scaled < -7, -7, torch.where(scaled > 7, 7, scaled.round()-scaled))
            ds = (grad.float()*term).sum(1,keepdim=True) / (scaled.shape[1]*7)**0.5
            return grad, ds, None
    quant_utils.ChannelScaleQuantize = NativeChannelScale
    original_enable = rc.enable_non_downproj_scale_learning
    def enable(model):
        _, bypass = original_enable(model)
        # Hardware produces one norm carrier shared by Q/K/V and one by Gate/Up.
        for layer in model.model.layers:
            for group in [(layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj), (layer.mlp.gate_proj, layer.mlp.up_proj)]:
                shared = group[0].quantizer.scale
                for wrapper in group[1:]: wrapper.quantizer.scale = shared
        seen = set(); learned = []
        for name, module in model.named_modules():
            if isinstance(module, quant_utils.RotationStaticActQuantizer) and module.bits < 16 and id(module.scale) not in seen:
                seen.add(id(module.scale)); learned.append((name, module.scale))
        assert len(learned) == 48 and len(bypass) == 16
        return learned, bypass
    rc.enable_non_downproj_scale_learning = enable
    if stage == "b_init":
        def initialize_only(self, *args, **kwargs):
            self.state.global_step = 0
            self.state.log_history = [{"stage": "B_initial_SW_only", "updates": 0}]
        Trainer.train = initialize_only
    import optimize_rotation
    optimize_rotation.train()

def run(stage):
    subprocess.run(["python3", "/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py", "preflight", "--source-worktree", str(ROOT)], check=True)
    snapshot=json.loads((RESULTS/"reference-snapshot.json").read_text())
    for name, item in snapshot["files"].items():
        path=REFERENCE.parents[1]/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"], path
    parents={}
    for parent in ([] if stage=="initial" else ["initial"] if stage=="b_init" else ["initial","b_init"]):
        record=json.loads((MODELS/parent/"complete.json").read_text())
        for name,digest in record["artifacts"].items():assert hashlib.sha256((MODELS/parent/name).read_bytes()).hexdigest()==digest
        parents[parent]=hashlib.sha256((MODELS/parent/"complete.json").read_bytes()).hexdigest()
    out=MODELS/stage
    out.mkdir(parents=True, exist_ok=False)
    args=["--input_model", "/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin",
          "--output_rotation_path", str(out/"rotation"), "--output_dir", str(out/"trainer"),
          "--logging_dir", str(out/"logs"), "--model_max_length", "2048",
          "--fp16", "False", "--bf16", "True", "--log_on_each_node", "False",
          "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "8",
          "--logging_steps", "1", "--learning_rate", "1.5", "--activation_scale_learning_rate", "1.0",
          "--weight_decay", "0.0", "--lr_scheduler_type", "cosine", "--gradient_checkpointing", "True",
          "--save_strategy", "no", "--max_steps", "100", "--seed", "42", "--data_seed", "42",
          "--report_to", "none", "--rotation_components", "r1_r2", "--a_quant_mode", "rotation_static",
          "--a_bits", "8", "--a_clip_ratio", "1.0", "--learn_activation_scales", "--w_groupsize", "-1",
          "--no-w_asym", "--no-w_clip", "--k_bits", "16", "--v_bits", "16",
          "--calibration_nsamples", "32", "--calibration_seqlen", "128", "--calibration_seed", "42"]
    if stage=="initial": args += ["--w_bits", "16"]
    else:
        args += ["--w_bits", "4", "--w4_aware_training", "--warmup_steps", "10",
                 "--optimized_rotation_path", str(MODELS/"initial/rotation/R.bin"),
                 "--static_scale_path", str(MODELS/"initial/rotation/quant_scales.pt")]
        if stage=="c": args += ["--learn_weight_scales", "--weight_scale_learning_rate", "0.01", "--initial_weight_scale_path", str(MODELS/"b_init/rotation/initial_quant_scales.pt")]
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REFERENCE), PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="8",CUDA_VISIBLE_DEVICES="0",HF_HOME="/home/daniuniu/work/rotation-quant/cache/huggingface",HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1")
    command=[PYTHON,"-B","-m","torch.distributed.run","--standalone","--nnodes=1","--nproc_per_node=1",str(Path(__file__).resolve()),"--worker",stage,*args]
    record=dict(stage=stage,command=command,parents=parents,source_head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),reference_snapshot_sha256=hashlib.sha256((RESULTS/"reference-snapshot.json").read_bytes()).hexdigest(),seed=42,updates=0 if stage=="b_init" else 100,effective_batch_sequences=8,sequence_length=2048,adaptations=["single GPU accumulation8 replaces two GPU accumulation4", "native symmetric W4 [-7,7]", "48 learned SA shared by QKV and Gate/Up; 96 input sites", "B initial SW only, no unused B or A training/GPTQ", "audited FP32 R1 folds except FP64 O; all R2 FP64; final export FP64"])
    (out/"launch.json").write_text(json.dumps(record,indent=2)+"\n")
    with (out/"run.log").open("x") as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"{stage} failed: {out / 'run.log'}")
    artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"rotation").iterdir() if p.is_file()}
    (out/"complete.json").write_text(json.dumps(dict(**record,artifacts=artifacts),indent=2)+"\n")
    print("TRAINING_STAGE_COMPLETE", stage, flush=True)

if __name__=="__main__":
    if "--worker" in sys.argv:
        i=sys.argv.index("--worker"); stage=sys.argv[i+1]; del sys.argv[i:i+2];worker(stage)
    else:
        p=argparse.ArgumentParser();p.add_argument("stage",choices=["initial","b_init","c"]);run(p.parse_args().stage)
