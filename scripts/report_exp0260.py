#!/usr/bin/env python3
from profile_exp0260 import *
from measure_exp0218 import OVERVIEW
preflight();assert read(R/'slice_gate.json')['pass_all'];assert read(R/'full_formal_gate.json')['integrity_pass']
data={};loops={};calls={1:0,28:0}
for count,prefix in [(1,'layer'),(28,'full')]:
 gold=read(R/('layer_control/validated.json' if count==1 else 'full_control/validated.json'))
 for phase,n in [('short',5),('formal',10)]:
  g=read(R/f'{prefix}_{phase}_gate.json');rows=[]
  for i in range(1,n+1):
   for rep in [1,10]:
    for arm in [0,2]:
     p=R/f'{prefix}_{phase}/round{i:02d}_r{rep}_a{arm}';rs=records(p/'stdout.jsonl');ps=[q for q in rs if q.get('record') in ['exp0240_profile','generation_profile']];physical(ps,count,arm);assert len(ps)==rep*(9 if count==1 else 16);calls[count]+=len(ps);z=read(p/'validated.json')
     assert z['prefill_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill') and z['decode_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode')
     if count==1:assert z['output_hashes']==gold['output_hashes']*rep
     else:
      fs=[q for q in rs if q.get('generation_sequence_complete')];assert len(fs)==rep
      codes=[q['selected_logit_half_bits'] for q in rs if 'selected_logit_half_bits' in q];assert codes==gold['selected_codes']*rep
      assert [q['token_ids'] for q in fs]==gold['token_sequences']*rep
      for j,f in enumerate(fs):assert f['total_host_wall_ns']==sum(q['host_wall_ns'] for q in ps[j*16:j*16+16]) and f['generation_loop_wall_ns']>=f['total_host_wall_ns']
      for q in ps:
       assert q['boundary_ddr_write_bytes']==0 and q['backend']=='standalone_fastrpc_dsp' and q['qnn']=='none'
       for j in range(28):
        l=q[f'slice_layer_{j}'];assert l['hidden_ddr_read_bytes']==l['hidden_ddr_write_bytes']==0
      if phase=='formal':
       key=f'a{arm}_r{rep}';d=data.setdefault(key,{'prefill':[],'decode':[]})
       for mode in d:d[mode].append(normalized([q for q in ps if q['mode']==mode]))
       u=loops.setdefault(key,{'model':[],'loop':[],'startup':[]});u['model'].append(statistics.mean(f['total_host_wall_ns'] for f in fs));u['loop'].append(statistics.mean(f['generation_loop_wall_ns'] for f in fs));u['startup'].append(next(q for q in rs if q.get('record')=='generation_startup'))
     rows.append(dict(round=i,**z))
  rng=np.random.default_rng(260)
  for rep in [1,10]:
   for mode in ['prefill_ns','decode_ns']:
    pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [0,2]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);v=g['performance'][f'r{rep}_{mode}'];assert abs(np.median(ratios)-v['paired_ratio'])<1e-12 and np.max(abs(ci-v['ci95']))<1e-12
assert calls=={1:2970,28:5280}
# Current control must reproduce the sealed old C64 implementation, not just itself.
p=R.parent/'exp0230';assert sha(p/'evidence_sha256.json')=='bb90bd21fb0d9ba979fcad9f6224b8dd739d9939f493636f5d85b4d8b07160f2';seal=read(p/'evidence_sha256.json')
old=p/'formal/round_01_C64.jsonl';assert sha(old)==seal[str(old.relative_to(p))]
goldcodes=[q['selected_logit_half_bits'] for q in records(old) if 'selected_logit_half_bits' in q];goldtokens=[q['token_ids'] for q in records(old) if q.get('generation_sequence_complete')]
assert goldcodes==read(R/'full_control/validated.json')['selected_codes'] and goldtokens==read(R/'full_control/validated.json')['token_sequences']
parent=R.parent/'exp0259';assert sha(parent/'EVIDENCE_SHA256.json')=='2f891f9aa35e7fa5bece5180e8e3a81fe4fd1253dc3749565dd226b462541a07';seal259=read(parent/'EVIDENCE_SHA256.json')['files'];assert sha(parent/'historical_provenance.json')==seal259['historical_provenance.json']['sha256'];hist=read(parent/'historical_provenance.json');fseries=[];aseries=[];histfiles={str(old):sha(old)}
for name,h in sorted(hist.items()):
 if 'exp0218' not in name:continue
 assert sha(name)==h;histfiles[name]=h;fseries.append(normalized([q for q in records(name) if q.get('record')=='generation_profile' and q['mode']=='prefill']))
for i in range(1,11):
 p=parent/f'full_formal/round{i:02d}_r10_a1/stdout.jsonl';assert sha(p)==seal259[str(p.relative_to(parent))]['sha256'];histfiles[str(p)]=sha(p);aseries.append(normalized([q for q in records(p) if q.get('record')=='generation_profile' and q['mode']=='prefill']))
assert len(fseries)==len(aseries)==10

def val(ss,fields):return statistics.median(sum(x[k] if k.endswith('_us') else x[k]/19.2 for k in fields) for x in ss)
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
series=[fseries,data['a2_r10']['prefill'],aseries];mod=[];module=[]
for label,(_,fields) in zip(labels,OVERVIEW):
 vs=[val(ss,fields) for ss in series];cells=[f'{v:.1f} ({100*v/val(ss,["host_us"]):.2f}%)' for v,ss in zip(vs,series)];mod.append([label,*cells,f'{100*(vs[1]/vs[2]-1):+.2f}%' if vs[2] else 'N/A']);module.append(dict(module=label,F16_us=vs[0],W4F16_us=vs[1],W4A8_us=vs[2]))
compact=table(['模块','F16A16 EXP0218','W4A16 C64 EXP0260','W4A8 R3 EXP0259','A8相对W4A16增速'],mod);(R/'module_table.md').write_text(compact+'\n');through={}
for k,d in data.items():
 pre=val(d['prefill'],['host_us']);dec=val(d['decode'],['host_us']);loop=statistics.median(loops[k]['loop'])/1000;model=statistics.median(loops[k]['model'])/1000
 through[k]=dict(prefill_tokens=64,prefill_host_us=pre,prefill_tokens_per_second=64e6/pre,decode_tokens=15,decode_host_us=dec*15,decode_host_us_per_token=dec,decode_tokens_per_second=1e6/dec,selected_output_tokens=16,model_host_us=model,model_effective_tokens_per_second=16e6/model,generation_loop_us=loop,generation_loop_tokens_per_second=16e6/loop)
summary=dict(experiment='EXP-0260',execution_state='completed',evidence_validity='valid',local_gate='pass',implementation_eligible=True,speed_eligible=read(R/'full_formal_gate.json')['speed_eligible'],model_quality='unchanged_not_reassessed',baseline_promoted=False,device_PPL=None,layer_timed_RPCs=2970,fullmodel_timed_RPCs=5280,fullmodel_layer_ledgers=147840,throughput=through,performance=read(R/'full_formal_gate.json')['performance'],modules=module)
write(R/'summary.json',summary);write(R/'numeric_diagnostics.json',dict(data=data,loops=loops));write(R/'historical_provenance.json',histfiles);write(R/'independent_integrity_checks.json',dict(pass_all=True,layer_timed_RPCs=2970,fullmodel_timed_RPCs=5280,selected_token_and_FP16_code_checks=5280,independent_CIs_reconstructed=True,old_C64_control_reproduced=True,complete_ledgers=True))
lines=['# EXP0260 W4F16 decode exact conversion and bulk-transfer optimization','','Freeze original C64 per-output-channel W4, all FP16 math and cache semantics. OPT0 retains current control, OPT1 batches exact storage conversion around unchanged scalar expf/ordered FP32 sum/division, OPT2 adds HVX zero/copy for dynamic decode Attention. W4F16 M1 only; other recipes/prefill kernels unchanged. No new quantizer, grouping, rotation, approximation or PPL acceptance.','','Initial legacy global-audit option was incompatible with replay, repaired with dedicated component audit; failed logs retained. Exhaustive finite FP16 roundtrip and probability-midpoint tests found V79vcvt drops negativezero sign. Explicit IEEE sign preservation repaired it; final finite63488patterns and46080midpoint-neighbour tests pass, plus oldSoftmax entire probability buffers exact. Sentinel failure was not observed real-model quality regression and no gate was loosened.','','Layer/slice live hidden and allstep KV exact, fullmodel control reproduces old sealed EXP0230 selectedtokens/FP16codes. All2970layer plus5280fullmodel timedRPCs validated. Same8MiB VTCM, no timed intermediateDDR/spill/fulllogit export, oneRPC/fullpass. Independent ledgers/statistics and code/provenance hashes retained.','','Formal comparisons are rotated paired5short10formal, bothrepeat1/10. Primary is repeat10 fullmodel decode; prefill guard<=10percent. No optional additional sampling used.','',table(['scope','control μs','candidate μs','paired cost change','95% interval'],[[k,f'{v["control_ns"]/1000:.3f}',f'{v["candidate_ns"]/1000:.3f}',f'{100*(v["paired_ratio"]-1):+.3f}%',str(v['ci95'])] for k,v in summary['performance'].items()]),'','M64 overview: μs(Host share); F16EXP0218 and W4A8R3EXP0259 historical nonpaired, new W4F16C64 candidate is not Selected EXP0166. W4A8 previous numerical failure unchanged. Per-row medians need not sum; individual raw additive ledgers checked.','',compact]
end=['','Actual warm28layer E2E,64prefill+15continuousdecode,16outputs. Host includes embedding/norm/head/greedy/FastRPC; generationloop includes deviceHost loop/log serialization, excludes model loading/ADB/externaltokenizer. No layer extrapolation.','',table(['arm/repeat','prefill tok/s','decode tok/s','16outputs/Host tok/s','generationloop tok/s'],[[k,*[f'{v[f]:.6f}' for f in ['prefill_tokens_per_second','decode_tokens_per_second','model_effective_tokens_per_second','generation_loop_tokens_per_second']]] for k,v in through.items()])]
(R/'REPORT.md').write_text('\n'.join(lines+end)+'\n');full=lines+['','Complete field comparison: ticks=19.2/μs; overlapping counters are not additive.']
for rep in [1,10]:
 for mode in ['prefill','decode']:
  ds=[data[f'a{a}_r{rep}'][mode] for a in [0,2]];keys=set.intersection(*(set(x) for d in ds for x in d));rr=[]
  for k in sorted(keys):
   vs=[statistics.median(x[k] for x in d) for d in ds];rr.append([k,*[f'{v:.6f}' for v in vs],f'{vs[1]-vs[0]:+.6f}'])
  full+=['',f'## r{rep} {mode}','',table(['field','control','candidate','delta'],rr)]
(R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(full+end)+'\n');print(json.dumps(summary,indent=2))
