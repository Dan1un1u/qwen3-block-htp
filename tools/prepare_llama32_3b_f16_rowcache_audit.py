import sys,os,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from prepare_llama32_3b import preflight,save
from llama_reference import sha256
from execute_llama32_3b_f16 import R,M,adb,windows,read
preflight();p=M.parent/'l32-0047/frontend64-fixed';remote=read(R/'package-frontend64-fixed.json')['remote'];files={}
for i in range(28):
 q=p/f'layer{i}'
 for n in ['k','v']:
  a=np.fromfile(q/f'reference_kv_cache_{n}_hmx_f16_step00.bin','<f2').reshape(8,-1)
  rows=[]
  for h in a[:,:8192]:
   N,K=(64,128) if n=='k' else (128,64)
   w=h.reshape(N//32,K//32,16,32,2).transpose(0,3,1,2,4).reshape(N,K)
   rows.append(w if n=='k' else w.T)
  v=np.zeros((8,128,128),'<f2');v[:,:64]=np.stack(rows)
  # Store separate companion artifacts; leave sealed package/manifest immutable.
  d=R/'rowcache-companion'/f'layer{i}';d.mkdir(parents=True,exist_ok=True)
  for name,data in [(f'kv_cache_{n}_f16.bin',np.zeros_like(v)),(f'reference_kv_cache_{n}_f16.bin',v)]:
   f=d/name
   with f.open('xb') as z:z.write(data.tobytes())
   adb('push',windows(f),remote+f'/layer{i}/'+name)
   assert adb('shell','sha256sum '+remote+f'/layer{i}/'+name).stdout.split()[0]==sha256(f)
   files[str(f.relative_to(R/'rowcache-companion'))]=dict(bytes=f.stat().st_size,sha256=sha256(f))
save(R/'rowcache-companion/manifest.json',dict(experiment='L32-0047',scope='Independent row-major KV crosscheck of native-tail tiling only',files=files))
print('ROW_CACHE_COMPANION_READY',flush=True)
