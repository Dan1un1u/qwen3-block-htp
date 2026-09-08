import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0250');s=json.loads((R/'summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12.6,5.8),layout='constrained')
colors={'OFF':'#3065ac','R3':'#cf6b27'}
modes=['carrier','unbounded','wide_float','wide_exact','wide_sole','sole']
labels=['Real score','Rounded\nno clip','Raw U8\nfloat exp','Raw U8\ninteger exp','New SOLE','Old SOLE']
ax=axes[0]
for j,arm in enumerate(['OFF','R3']):
 ys=[s['final_ppl'][arm+'_'+m] for m in modes];xs=np.arange(len(modes))+(j-.5)*.32
 ax.bar(xs,ys,width=.3,color=colors[arm],label=arm)
 for x,y in zip(xs,ys):ax.annotate(f'{y:,.1f}',(x,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8,color=colors[arm])
for name,style in [('F',':'),('C64','--')]:ax.axhline(s['final_ppl'][name],color='#666666',linestyle=style,label=name)
ax.set_yscale('log');ax.set_ylim(10,max(s['final_ppl'].values())*1.8);ax.set_xticks(range(len(modes)),labels,rotation=15);ax.set_ylabel('PPL (log scale; lower is better)');ax.set_title('Frozen paths on independent 128 documents');ax.legend(ncol=4,fontsize=9);ax.grid(axis='y',alpha=.18)
ax=axes[1]
keys=list(s['ladder']['OFF']);short=['Score rounding / gain','First U8 saturation','Second U8 saturation','Integer exponent','SOLE reciprocal','New / old SOLE']
for j,arm in enumerate(['OFF','R3']):
 ratios=[s['ladder'][arm][k]['ppl_ratio'] for k in keys];cis=np.array([s['ladder'][arm][k]['ratio_ci95'] for k in keys]);ys=np.arange(len(keys))+(j-.5)*.19
 ax.errorbar(ratios,ys,xerr=np.array([ratios-cis[:,0],cis[:,1]-ratios]),fmt='o',capsize=3,color=colors[arm],label=arm)
ax.axvline(1,color='#555555',linestyle='--');ax.set_xscale('log');ax.set_yticks(range(len(keys)),short);ax.invert_yaxis();ax.set_xlabel('Paired PPL ratio, pointwise 95% CI');ax.set_title('Conditional comparisons');ax.grid(axis='x',alpha=.18)
fig.suptitle('EXP-0250 | Removing the second U8 score saturation',fontsize=15)
fig.supxlabel('2048 targets; frozen software arithmetic outside attention. No device accuracy or speed measurement.',fontsize=9)
fig.savefig(R/'raw_score_repair.png',dpi=180);fig.savefig(R/'raw_score_repair.pdf');plt.close(fig)
print('PLOTS_WRITTEN')
