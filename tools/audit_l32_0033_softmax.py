"""Read-only reconstruction of matched softmax approximation and component costs."""
import json,hashlib,statistics
from pathlib import Path
import numpy as np
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0033');P=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0033/layer7-fp-a02');O=P.parent.parent/'l32-0016/layer7-a02'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def diff(a,b):
 d=a.astype('f8')-b.astype('f8');return dict(elements=d.size,differing=int(np.count_nonzero(d)),max_abs=float(np.max(np.abs(d))),mean_abs=float(np.mean(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))
actual=[]
for step,mode in enumerate(['prefill','decode']):
 for name in ['q','k']:assert np.array_equal(np.load(P/f'fp_step{step}_{name}.npy'),np.load(O/f'fp32_{mode}_{name}.npy'))
 for name in ['k','v']:
  a=np.fromfile(P/f'layer0/reference_kv_cache_{name}_u8.bin','u1').reshape(8,80,64)[:,:65];b=np.fromfile(O/f'layer0/reference_kv_cache_{name}_u8.bin','u1').reshape(8,80,64)[:,:65];assert np.array_equal(a,b)
 fp=np.load(P/f'fp_step{step}_probability.npy');log=np.load(O/f'fp32_{mode}_probability.npy');avf=np.load(P/f'fp_step{step}_attention.npy');avl=np.load(O/f'fp32_{mode}_attention.npy')
 actual.append(dict(mode=mode,same_QKV=True,probability_u8=diff(fp,log),probability_dequantized=diff(fp/255.,log/255.),AV_u8=diff(avf,avl),scope='independent selected-layer matched QKV references; both full selected outputs separately exact on device; AV changes are approximation differences, not hardware implementation errors'))
g=read(R/'component_gate.json');assert g['pass_all'];fp=[x for x in g['runs'] if x['arm']=='FP'];assert all(x['own_u8_exact'] and x['max_abs']<=2e-6 and x['row_sum_error']<=2e-6 for x in fp)
component={}
for case in ['m64-p0-random','m1-p64-random']:
 z=read(R/f'component-{case}-formal.json');assert z['pass_all'];component[case]={a:statistics.mean(x['ticks']/x['repeat']/19.2 for x in z['runs'] if x['arm']==a) for a in ['LOG2','FP']}
supp=dict(component_fp_cases=len(fp),component_total_cases=len(g['runs']),max_probability_abs=max(x['max_abs'] for x in fp),max_row_sum_error=max(x['row_sum_error'] for x in fp),float64_u8_threshold_differences=sum(x['float64_u8_threshold_differences'] for x in fp),component_formal_us=component,component_scope='4 heads; core includes native score conversion/packing and matched rowmass telemetry; immutable score restore excluded from both timers; separate pack timer N/A',actual_input_approximation=actual,quality_claim=False,softmax_claim='Llama retained per-layer log2 division configs include exact mode. Result does not establish every log2 design slower than FP32; it does not test Qwen NR64 port.',source_manifest_sha256={'FP':sha(P/'manifest.json'),'LOG2':sha(O/'manifest.json')})
with (R/'SOFTMAX_AUDIT.json').open('x') as f:json.dump(supp,f,indent=2)
print(json.dumps(supp,indent=2))
