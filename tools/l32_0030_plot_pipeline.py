import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0030')
z=json.loads((R/'SUMMARY.json').read_text());scopes=list(z['scopes']);arms=['QKV','FFN','EPILOGUE','LOOKAHEAD'];colors=['#386cb0','#e78a35','#50a47a','#9765aa']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(12.5,4.6))
for ax,mode in zip(axs,['prefill','decode']):
 for j,(a,c) in enumerate(zip(arms,colors)):
  v=[z['scopes'][s]['effects'][mode][a] for s in scopes];y=np.array([100*(q['ratio']-1) for q in v]);lo=np.array([100*(q['ci95'][0]-1) for q in v]);hi=np.array([100*(q['ci95'][1]-1) for q in v]);x=np.arange(3)+(j-1.5)*.10
  ax.errorbar(x,y,yerr=[y-lo,hi-y],marker='o',ms=4,lw=1.3,capsize=3,label=('QKV (inactive control)' if a=='QKV' else a),color=c)
 ax.axhline(0,color='#666666',lw=.8,ls='--');ax.set_xticks(range(3),[f'{scopes[0]} layer',f'{scopes[1]} layers',f'Full {scopes[2]} + frontend']);ax.set_title(mode.capitalize());ax.set_ylabel('Host wall increase when disabled (%)');ax.grid(axis='y',alpha=.18)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),frameon=False,ncol=4,fontsize=9)
fig.suptitle(z['experiment']+' | Conditional pipeline effects',fontsize=14)
fig.text(.5,.018,'Native formats, SP2, FP32 residual, no rotation. Paired 95% bootstrap CIs; fixed 10 rounds. Effects are not additive.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.95))
for ext in ['png','svg','pdf']:fig.savefig(R/('pipeline_scaling.'+ext),dpi=180)
