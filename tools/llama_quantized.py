"""Load only freshly sealed Llama per-channel quantization output."""
import json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from llama_reference import sha256

def load_quantized(model,root):
 m=json.loads((root/'manifest.json').read_text());assert m['experiment']=='L32-0002' and m['recipe']=='W4A16'
 for n,h in m['files'].items():assert sha256(root/n)==h,n
 for i,layer in enumerate(model.model.layers):layer.load_state_dict(load_file(root/f'layer{i}/dequant.safetensors',device=str(model.device)))
 model.lm_head.weight=torch.nn.Parameter(load_file(root/'head/dequant.safetensors',device=str(model.device))['weight'])
 model.config.tie_word_embeddings=False
 return m

def link_projection(root,index,name,out):
 for suffix in ['hmx','scale_f32']:
  src=root/f'layer{index}'/f'{name}_weight_w4_{suffix}.bin';os.link(src,out/src.name)
