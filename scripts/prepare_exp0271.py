"""Reuse immutable Qwen weights and build only missing FP32 replay references."""
from common_exp0271 import *
import os,shutil,numpy as np
from reference_exp0269 import layer,verify

def prepare():
 preflight();src=OLD/'sp2';m=verify(src);dst=O/'sp2-fp32';dst.mkdir(parents=True,exist_ok=False)
 for n in m['files']:
  d=dst/n;d.parent.mkdir(parents=True,exist_ok=True);os.link(src/n,d)
 ep=O.parent/'exp0230/C64';em=read(ep/'manifest.json');name='generation_embedding_weight_f16.bin';h=sha(ep/name)
 # Original FP16 embedding retained by C64, not quantized or rotated anew.
 ev=em.get('files',{}).get(name);assert ev is not None,(list(em),name)
 assert h==(ev['sha256'] if isinstance(ev,dict) else ev)
 os.link(ep/name,dst/name)
 single=O.parent/'exp0269/layer0-fp32-a01';past=[None]*3
 for step in range(2):
  inp='reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin'
  x=np.fromfile(single/inp,'<f4').reshape(64,2048)[:64 if not step else 1];(dst/inp).write_bytes((single/inp).read_bytes())
  cos=np.fromfile(dst/('rope_cos_f16.bin' if not step else 'replay_decode_rope_cos_00_f16.bin'),'<f2').reshape(64,128)
  sin=np.fromfile(dst/('rope_sin_f16.bin' if not step else 'replay_decode_rope_sin_00_f16.bin'),'<f2').reshape(64,128)
  for l in range(3):
   x,past[l],d=layer(x,dst/f'layer{l}',cos,sin,past[l]);np.save(dst/f'chain_step{step}_layer{l}_output.npy',x)
   print('REFERENCE',step,l,flush=True)
  out=np.zeros((64,2048),'<f4');out[:len(x)]=x;out.tofile(dst/('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin'))
  for name,val in d.items():np.save(dst/f'chain_step{step}_last_{name}.npy',val)
  for name,val in zip(['k','v'],past[-1]):np.save(dst/f'chain_step{step}_last_cache_{name}.npy',val)
 # Never include the source manifest as a hash of the derived package.
 manifest=dict(experiment='EXP-0271',parent=str(src),parent_manifest_sha256=sha(src/'manifest.json'),embedding_parent=str(ep),embedding_sha256=h,contract='unchanged SP2 weights; original FP16 embedding; FP32 replay reference is consecutive first3 only, ignored by full generation',files={str(f.relative_to(dst)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in dst.rglob('*') if f.is_file()})
 write(dst/'manifest.json',manifest);write(R/'package.json',dict(path=str(dst),manifest_sha256=sha(dst/'manifest.json'),files=len(manifest['files'])));print('PREPARED',flush=True)
if __name__=='__main__':prepare()
