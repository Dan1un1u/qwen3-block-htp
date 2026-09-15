import json,numpy as np
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0274/consumer-row1-a02')
z=json.loads((R/'formal.json').read_text());assert z['pass_all']
fig,axes=plt.subplots(1,2,figsize=(8.6,3.3),layout='constrained');colors=['#64748b','#0284c7'];rng=np.random.default_rng(274);idx=rng.integers(0,10,(20000,10))
for ax,mode in zip(axes,['prefill','decode']):
 for j,(arms,label) in enumerate([(['B00','B10'],'Operator-level schedule'),(['B01','B11'],'Collaborative pipeline')]):
  yy=[];lower=[];upper=[]
  for arm in arms:
   t=np.array([next(v[mode+'_ns'] for v in z['runs'] if v['cycle']==c and v['arm']==arm) for c in range(10)])/1e6
   y=t.mean();ci=np.quantile(t[idx].mean(1),[.025,.975]);yy.append(y);lower.append(y-ci[0]);upper.append(ci[1]-y)
  x=np.array([0,1])+(-.025 if j==0 else .025);ax.errorbar(x,yy,yerr=[lower,upper],marker='o',capsize=4,color=colors[j],label=label,linewidth=1.8)
 ax.set_xticks([0,1],['Modular interfaces','Native producers']);ax.set_title(mode.capitalize());ax.set_ylabel('Complete Host wall (ms'+(' / token' if mode=='decode' else ' / 64 tokens')+')');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Qwen: full-model format × pipeline ablation',fontsize=12)
fig.savefig(R/'factorial.png',dpi=200);fig.savefig(R/'factorial.svg');fig.savefig(R/'factorial.pdf')
