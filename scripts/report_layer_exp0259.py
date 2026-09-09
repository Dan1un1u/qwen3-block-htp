#!/usr/bin/env python3
from device_exp0259 import *
preflight();assert read(R/'numerical_gate.json')['pass_all']
data={};total=0
for phase,n in [('short',5),('formal',10)]:
 g=read(R/f'{phase}_gate.json');assert g['integrity_pass'];rng=np.random.default_rng(259);rows=[]
 for i in range(n):
  for rep in [1,10]:
   for arm in ARMS:
    p=R/f'{phase}/round{i+1:02}_r{rep}_{arm}';ps=[q for q in records(p/'stdout.jsonl') if q.get('record')=='exp0240_profile'];physical(ps,arm,False);assert len(ps)==rep*9;total+=len(ps)
    gold=read(R/f'audit_{"off" if arm=="off" else "r3"}/validated.json')['output_hashes'];assert [q['output_hash'] for q in ps]==gold*rep
    z=read(p/'validated.json');assert z['prefill_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill');assert z['decode_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode');rows.append(dict(round=i+1,**z))
    if phase=='formal':
     d=data.setdefault(f'{arm}_r{rep}',{'prefill':[],'decode':[]})
     for mode in d:d[mode].append(normalized([q for q in ps if q['mode']==mode]))
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   for arm in ARMS[1:]:
    for control in ['off']+(['r3'] if arm!='r3' else []):
     pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [control,arm]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);v=g['performance'][f'{arm}_vs_{control}_r{rep}_{mode}'];assert abs(np.median(ratios)-v['paired_ratio'])<1e-12 and np.max(abs(ci-v['ci95']))<1e-12
assert total==5940
perf=read(R/'formal_gate.json')['performance'];eligible=all(perf[f'stream_vs_off_r{rep}_{mode}']['ci95'][1]<=1.10 for rep in [1,10] for mode in ['prefill_ns','decode_ns'])
write(R/'layer_summary.json',dict(experiment='EXP-0259',scope='layer0 M64+8M1 replay, no end-to-end projection',implementation_gate='pass_byte_exact',timed_RPCs=total,numerical_eligible=False,speed_eligible=eligible,selected_before_formal='stream',performance=perf,data=data))
write(R/'layer_independent_integrity.json',dict(pass_all=True,timed_RPCs=total,statistics_independently_recomputed=True,physical_and_ledger_checks=True,all_output_hashes_exact=True,whole_numerical_capture_checks=108,fullmodel_escalation_allowed=eligible))
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rs])
lines=['# EXP0259 single-layer R3 implementation ablations','','Same layer0, M64 plus eight real replay decode steps, repeat1/repeat10,5short10formal. Constant+vector includes fusedKoperand; stream restores five workers on readiness events and relocates bias buffers. Exact sqrtf and originalHMX retained.108 complete capture/output comparisons and5940timedRPC pass. Original ideal-R3 numerical gate still failed; noPPL acceptance.','',table(['comparison','control_us','candidate_us','paired ratio','CI95'],[[k,f'{v["control_ns"]/1000:.3f}',f'{v["candidate_ns"]/1000:.3f}',f'{v["paired_ratio"]:.6f}',str(v['ci95'])] for k,v in perf.items()]),'',f'Fullmodel escalation eligible: {eligible}. Not model quality or baseline acceptance. E2E unmeasured at this layer scope.']
(R/'LAYER_REPORT.md').write_text('\n'.join(lines)+'\n')
for rep in [1,10]:
 for mode in ['prefill','decode']:
  ds=[data[f'{arm}_r{rep}'][mode] for arm in ARMS];keys=set.intersection(*(set(x) for d in ds for x in d));rr=[]
  for k in sorted(keys):rr.append([k,*[f'{statistics.median(x[k] for x in d):.6f}' for d in ds]])
  lines+=['',f'## r{rep} {mode}',table(['field',*ARMS],rr)]
(R/'LAYER_PROFILE.md').write_text('\n'.join(lines)+'\n');print('LAYER_REPORT_PASS',eligible,flush=True)
