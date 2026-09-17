#!/usr/bin/env python3
"""EXP0292 independent F16 projections / FP32 residual reference and fixtures."""
import inspect,sys,os,subprocess
from pathlib import Path
import prepare_exp0289 as p
import reference_a16_exp0289 as a
import torch
S=p.ROOT
p.O=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0292")
p.R=Path("/mnt/d/llm_exp/results/qwen3-block-htp/exp0292")
OLD=Path("/mnt/d/llm_exp/models/qwen3-block-htp/exp0289/f16f16")
def preflight():
 z=subprocess.run(["python3","/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py","preflight","--source-worktree",str(S)],check=True,capture_output=True,text=True)
 assert "EXPERIMENT=EXP-0292\n" in z.stdout
p.preflight=preflight
# Q/K and SwiGLU retain the original independent HVX-boundary model.
p.rms=a.rms
source=inspect.getsource(p.layer)
source=source.replace("res=(x+linear(av,w['o'])).half()","res=x.float()+linear(av,w['o']).float()")
source=source.replace("return (res+linear(mid,w['down'])).half(),","return (res+linear(mid,w['down']).float()),")
exec(source,p.__dict__)
p.padded=lambda x: __import__('numpy').pad(x.cpu().numpy().astype('<f4'),((0,64-len(x)),(0,0)))
p.__dict__['OLD_FP16']=OLD
source=inspect.getsource(p.main)
source=source.replace("put(out/f'{n}_weight_f16_hmx.bin',pack_weight(w[n]))","os.link(OLD_FP16/f'layer{i}/{n}_weight_f16_hmx.bin',out/f'{n}_weight_f16_hmx.bin')")
source=source.replace("put(dst/'generation_lm_head_weight_f16_hmx.bin',pack_weight(head))","os.link(OLD_FP16/'generation_lm_head_weight_f16_hmx.bin',dst/'generation_lm_head_weight_f16_hmx.bin')")
source=source.replace("x=F.embedding(torch.tensor(ids.astype('i8'),device='cuda'),emb)","x=F.embedding(torch.tensor(ids.astype('i8'),device='cuda'),emb).float()")
source=source.replace("x=emb[int(fixed[step-1])][None]","x=emb[int(fixed[step-1])][None].float()")
for before,after in [
 ("block_input_f16.bin","block_input_f32.bin"),("block_output_f16.bin","block_output_f32.bin"),
 ("audit_hidden_{step:02d}_f16.bin","audit_hidden_{step:02d}_f32.bin"),
 ("replay_decode_input_00_f16.bin","replay_decode_input_00_f32.bin"),
 ("replay_decode_reference_00_f16.bin","replay_decode_reference_00_f32.bin")]:source=source.replace(before,after)
exec(source,p.__dict__)
if __name__=="__main__":p.main('f16f16')
