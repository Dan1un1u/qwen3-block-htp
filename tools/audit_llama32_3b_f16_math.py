import sys,json
from pathlib import Path
import numpy as np
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0047');M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0047/frontend64-fixed')
tag=sys.argv[1];A=R/tag/'audit';out=[]
def metrics(a,b):
 a=a.astype('f8').reshape(-1);b=b.astype('f8').reshape(-1)
 finite=bool(np.isfinite(a).all() and np.isfinite(b).all());err=a-b
 return dict(finite=finite,nrmse=float(np.linalg.norm(err)/max(np.linalg.norm(b),1e-30)),cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),max_abs=float(abs(err).max()),mixed_violation_fraction=float(np.mean(abs(err)>.0625+.002*abs(b))))
for i in range(43):
 a=np.fromfile(A/f'generation_hidden_norm_step{i:02d}_f16.bin','<f2').reshape(2,3072)
 for j,n in enumerate(['hidden','norm']):
  b=np.fromfile(M/f'audit_{n}_{i:02d}_f16.bin','<f2');v=metrics(a[j],b);out.append(dict(step=i,boundary=n,**v))
caches=[]
for i in range(28):
 for n in ['k','v']:
  a=np.fromfile(A/f'generation_step00_layer{i:02d}_{n}_f16.bin','<f2').reshape(8,-1)[:,:64*128]
  b=np.fromfile(M/f'layer{i}/reference_kv_cache_{n}_hmx_f16_step00.bin','<f2').reshape(8,-1)[:,:64*128]
  caches.append(dict(layer=i,kind=n,**metrics(a,b)))
passed=all(v['finite'] and v['nrmse']<=.003 for v in out) and all(v['finite'] and v['cosine']>=.99999 and v['mixed_violation_fraction']<=.01 for v in caches)
result=dict(pass_all=passed,thresholds=dict(composed_nrmse_max=.003,cache_cosine_min=.99999,cache_mixed_violation_fraction_max=.01),boundaries=out,prefill_caches=caches,meaning='Independent mathematical FP16 composition; not bit exact or PPL acceptance')
(R/tag/'mathematical-audit.json').write_text(json.dumps(result,indent=2))
print('MATH_AUDIT',passed,'boundary nrmse',max(v['nrmse'] for v in out),'cache mincos',min(v['cosine'] for v in caches),'maxfrac',max(v['mixed_violation_fraction'] for v in caches))
