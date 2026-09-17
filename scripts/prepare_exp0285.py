"""Fresh residual storage references; original W4/SP2 weights and scales immutable."""
from common_exp0285 import *
from reference_exp0285 import layer
import numpy as np,os,shutil
def main():
 preflight()
 for name,orig,count in [('half-layer14',O.parent/'exp0269/layer14-fp32-a01',1),('half-full',O/'sp2-fp32',3)]:
  dest=package_path(name);dest.mkdir(parents=True,exist_ok=False);m=read(orig/'manifest.json')
  for n,v in m['files'].items():
   assert sha(orig/n)==v['sha256'],n
   d=dest/n;d.parent.mkdir(parents=True,exist_ok=True);d.symlink_to(orig/n)
  caches=[None]*count
  for step in range(2):
   inf='reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin'
   x=np.fromfile(orig/inf,'<f4').reshape(64,2048).astype('f2');x.tofile(dest/inf.replace('f32','f16'))
   x=x.astype('f4')[:64 if not step else 1]
   cos=np.fromfile(orig/('rope_cos_f16.bin' if not step else 'replay_decode_rope_cos_00_f16.bin'),'<f2').reshape(64,128)
   sin=np.fromfile(orig/('rope_sin_f16.bin' if not step else 'replay_decode_rope_sin_00_f16.bin'),'<f2').reshape(64,128)
   for i in range(count):
    x,caches[i],a=layer(x,orig/f'layer{i}',cos,sin,caches[i])
   y=np.zeros((64,2048),'<f2');y[:len(x)]=x
   outf='reference_w4u8_block_output_f16.bin' if not step else 'replay_decode_reference_00_f16.bin';y.tofile(dest/outf)
   for k,v in a.items():np.save(dest/f'half_step{step:02d}_{k}.npy',v)
  for d in dest.rglob('*'):
   if d.is_file() and not d.is_symlink():
    n=str(d.relative_to(dest));m['files'][n]=dict(sha256=sha(d),bytes=d.stat().st_size)
  m['exp0285']=dict(parent=str(orig),parent_manifest_sha256=sha(orig/'manifest.json'),residual='FP16 stored; FP32 add/norm',quality_claim=False)
  write(dest/'manifest.json',m);write(R/(name+'-package.json'),dict(path=str(dest),manifest_sha256=sha(dest/'manifest.json'),count=count))
  print('PACKAGE_PASS',name,flush=True)
if __name__=='__main__':main()
