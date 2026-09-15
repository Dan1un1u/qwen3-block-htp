"""Read-only reconstruction of matched softmax approximation and component costs."""
import json,hashlib,statistics
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from integer_attention_exp0252 import numpy_oracle,config
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0278/vsum');P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0278/layer14-fp');O=P.parent.parent/'exp0269/layer14-fp32-a01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def diff(a,b):
 d=a.astype('f8')-b.astype('f8');return dict(elements=d.size,differing=int(np.count_nonzero(d)),max_abs=float(np.max(np.abs(d))),mean_abs=float(np.mean(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))
actual=[]
configs=np.fromfile(P/'layer0/attention_config_all_groups.bin','<i4').reshape(8,15)
for step,mode in enumerate(['prefill','decode']):
 for name in ['q','k','cache_k','cache_v']:assert np.array_equal(np.load(P/f'fp32_step{step:02d}_{name}.npy'),np.load(O/f'fp32_step{step:02d}_{name}.npy'))
 q=np.load(P/f'fp32_step{step:02d}_q.npy');k=np.load(P/f'fp32_step{step:02d}_cache_k.npy');v=np.load(P/f'fp32_step{step:02d}_cache_v.npy');valid=np.arange(k.shape[1])[None,:] <= (0 if step==0 else 64)+np.arange(len(q))[:,None]
 fp=[];log=[];avf=[];avl=[]
 for g in range(8):
  args=(q[:,2*g:2*g+2].transpose(1,0,2),k[g],v[g],valid,config(tuple(int(x) for x in configs[g])))
  f=numpy_oracle(*args,'wide_float');l=numpy_oracle(*args,'wide_nr64');fp.append(f['probability']);log.append(l['probability']);avf.append(f['av']);avl.append(l['av'])
 fp=np.concatenate(fp);log=np.concatenate(log);avf=np.concatenate(avf).transpose(1,0,2).reshape(len(q),2048);avl=np.concatenate(avl).transpose(1,0,2).reshape(len(q),2048)
 assert np.array_equal(avf,np.load(P/f'fp32_step{step:02d}_attention.npy'));assert np.array_equal(avl,np.load(O/f'fp32_step{step:02d}_attention.npy'))
 actual.append(dict(mode=mode,same_QKV=True,probability_u8=diff(fp,log),probability_dequantized=diff(fp/255.,log/255.),AV_u8=diff(avf,avl),scope='independent matched actual selected-layer QKV, each AV output exact to own archived physical hardware boundary; cross-arm differences are approximation change'))
g=read(R/'component_gate.json');assert g['pass_all'];fp=[x for x in g['runs'] if x['arm']=='FP'];assert all(x['own_u8_exact'] and x['max_abs']<=2e-6 and x['row_sum_error']<=2e-6 for x in fp)
component={}
for case in ['m64-p0-random','actual-g0']:
 z=read(R/f'component-{case}-formal.json');assert z['pass_all'];component[case]={a:statistics.mean(x['ticks']/x['repeat']/19.2 for x in z['runs'] if x['arm']==a) for a in ['LOG2','FP']}
supp=dict(component_fp_cases=len(fp),component_total_cases=len(g['runs']),max_probability_abs=max(x['max_abs'] for x in fp),max_row_sum_error=max(x['row_sum_error'] for x in fp),float64_u8_threshold_differences=sum(x['float64_u8_threshold_differences'] for x in fp),component_formal_us=component,component_scope='2 heads; primary M64/past0 optimized native shuffle4 versus FP vector; actual-g0 decode; core includes native pack and matched rowmass telemetry; separate pack timer N/A; generic M64/past64 scalar fallback supplementary only',actual_input_approximation=actual,quality_claim=False,softmax_claim='Retained Qwen wide NR64 versus final FP vector rowmass. Prototype scalar rowmass full timings retained separately, not pooled or denominator for final control.',source_manifest_sha256={'FP':sha(P/'manifest.json'),'LOG2':sha(O/'manifest.json')})
with (R/'SOFTMAX_AUDIT.json').open('x') as f:json.dump(supp,f,indent=2)
print(json.dumps(supp,indent=2))
