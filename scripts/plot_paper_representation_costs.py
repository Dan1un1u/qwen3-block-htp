import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/d/llm_exp/results/paper-no-rotation-ablation-20260915');R.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
base=Path('/mnt/d/llm_exp/results');paths={'Qwen':{'stream':base/'qwen3-block-htp/exp0276/SUMMARY.json','cost':base/'qwen3-block-htp/exp0277/SUMMARY.json','softmax':base/'qwen3-block-htp/exp0278/vsum/SUMMARY.json'},'Llama':{'stream':base/'llama32-htp/l32-0031/SUMMARY.json','cost':base/'llama32-htp/l32-0032/SUMMARY.json','softmax':base/'llama32-htp/l32-0033/SUMMARY.json'}}
data=[];bind={}
for model,ps in paths.items():
 for kind,p in ps.items():
  z=read(p);bind[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for mode in ['prefill','decode']:
   if kind=='stream':e=next(iter(z['scopes'].values()))['effects'][mode]['SPLIT'];ratio=e['ratio']
   else:e=z['trajectories']['fixed']['effects'][mode];ratio=e['sp2_over_a8_wall' if kind=='cost' else 'fp_over_log2_wall']
   data.append(dict(model=model,kind=kind,mode=mode,ratio=ratio,ci95=e['ci95']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(12.6,3.7),sharey=True)
ylabels=['Qwen prefill','Qwen decode','Llama prefill','Llama decode'];order=[('Qwen','prefill'),('Qwen','decode'),('Llama','prefill'),('Llama','decode')]
for ax,kind,title in zip(axes,['stream','cost','softmax'],['A3: split / copacked SP2','A5: SP2 / ordinary A8','A9: FP32 / retained log2']):
 ax.axvline(0,color='#8c959f',lw=1);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 for y,(model,mode) in enumerate(order):
  v=next(v for v in data if v['model']==model and v['mode']==mode and v['kind']==kind);x=100*(v['ratio']-1);lo,hi=[100*(a-1) for a in v['ci95']];color='#2563a8' if model=='Qwen' else '#be5920'
  ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o' if mode=='prefill' else 's',color=color,capsize=4,markersize=6,lw=1.5)
  ax.annotate(f'{x:+.2f}%',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9,color=color)
 ax.set_title(title,fontsize=11);ax.set_xlabel('Full-model Host wall change (%)');ax.set_yticks(range(4),ylabels);ax.set_ylim(3.65,-.65)
axes[0].set_xlim(-.5,2.0);axes[1].set_xlim(-.5,2.0);axes[2].set_xlim(-6.3,3)
fig.text(.5,.02,'95% paired bootstrap CI; each panel has its own matched control. A3: greedy. A5/A9: fixed token input. FP32 residual, no rotation.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.1,1,1))
for suffix in ['png','svg','pdf']:fig.savefig(R/('paired_representation_costs.'+suffix),dpi=220,bbox_inches='tight')
with (R/'paired_representation_costs.json').open('x') as f:json.dump(dict(input_sha256=bind,values=data),f,indent=2)
print('PLOT_READY',R,flush=True)
