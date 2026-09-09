#!/usr/bin/env python3
"""One fixed ten-pair uncertainty follow-up; retain all first-campaign data."""
from device_exp0259 import *
preflight();g=read(R/'formal_gate.json');assert g['integrity_pass']
rows=[]
for i in range(10):
 for rep in [1,10]:
  for arm in (['off','stream'] if i%2==0 else ['stream','off']):
   tag=f'confirmation/round{i+11:02}_r{rep}_{arm}';p=R/tag/'validated.json';z=read(p) if p.exists() else run(arm,rep,tag);rows.append(dict(round=i+11,**z))
allrows=[z for z in g['rows'] if z['arm'] in ['off','stream']]+rows
rng=np.random.default_rng(259);stats={}
for scope,rr in [('confirmation_only',rows),('all_twenty_pairs',allrows)]:
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   ids=sorted(set(z['round'] for z in rr));pairs=np.array([[next(z[mode] for z in rr if z['round']==i and z['repeat']==rep and z['arm']==a) for a in ['off','stream']] for i in ids]);ratio=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratio[rng.integers(0,len(ids),(10000,len(ids)))],axis=1),[.025,.975]);stats[f'{scope}_r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratio)),ci95=ci.tolist())
eligible=all(v['ci95'][1]<=1.1 for k,v in stats.items() if k.startswith('all_twenty_pairs'))
write(R/'confirmation_gate.json',dict(integrity_pass=True,rows=rows,statistics=stats,fullmodel_escalation_allowed=eligible,additional_rounds_fixed_before_capture=10,reason='initial r1 decode CI upper1.10959 straddles1.10; no retuning, no discarded data',no_further_sampling_if_uncertain=True))
print('CONFIRMATION_COMPLETE',eligible,json.dumps(stats),flush=True)
