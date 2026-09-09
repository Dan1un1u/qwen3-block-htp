#!/usr/bin/env python3
"""Three-layer on-device chain versus separately executed layer oracles."""
from device_exp0257 import *
import numpy as np

def main():
 preflight();assert json.loads((R/'single_reproduction.json').read_text())['pass_all'];assert json.loads((R/'device_package.json').read_text())['verified_files']==906
 base=REMOTE+'-package-v2';manifest=json.loads((O/'package/manifest.json').read_text());proof={}
 previous=None
 for layer in range(3):
  if layer==0:pkg=base
  else:
   pkg=REMOTE+f'-chain-layer{layer}';adb('shell',f'mkdir {pkg} && ln -s {base}/* {pkg}/')
   # These removals affect only our newly created symlink entries.
   commands=[f'rm {pkg}/layer0',f'ln -s {base}/layer{layer} {pkg}/layer0']
   for f in (O/'package'/f'layer{layer}').iterdir():
    if f.is_file() and ('weight' in f.name or f.name in ['qparams_u8.bin','silu_up_lut_u16.bin','attention_config_all_groups.bin']):
     commands.extend([f'rm {pkg}/{f.name}',f'ln -s {base}/layer{layer}/{f.name} {pkg}/{f.name}'])
   adb('shell',' && '.join(commands))
   inputs=R/f'chain_inputs_layer{layer}';inputs.mkdir()
   z=manifest['layer_qparams'][layer]['block_input']['zero_point']
   for step in range(9):
    name='reference_w4u8_block_input_u8.bin' if step==0 else f'replay_decode_input_{step-1:02d}_u8.bin'
    arr=np.full((64,2048),z,np.uint8);source=np.fromfile(previous/f'step{step:02d}_output.bin',np.uint8).reshape(-1,2048);arr[:len(source)]=source
    arr.tofile(inputs/name);adb('shell',f'rm {pkg}/{name}');adb('push',win(inputs/name),pkg+'/'+name)
  previous=replay(1,4,f'chain_layer{layer}',pkg)
 # Separate normalizer formulation, identical three-layer state ownership.
 candidate=replay(3,4,'slice_nr64',base);scalar=replay(3,6,'slice_scalar_nr64',base)
 for step in range(9):
  name=f'step{step:02d}_output.bin';x=(candidate/name).read_bytes();assert x==(previous/name).read_bytes(),('chain',step);assert x==(scalar/name).read_bytes(),('scalar',step)
  proof[str(step)]=dict(separate_layers_byte_exact=True,scalar_normalization_byte_exact=True,sha256=sha(candidate/name))
 write(R/'slice_gate.json',dict(pass_all=True,layers=[0,1,2],steps=proof,cache_capacity=128,note='Layer0 independent prior integer oracle; separate hardware layers prove integration, not GPU-model equivalence.'))
 print('SLICE_GATE_PASS',flush=True)
if __name__=='__main__':main()
