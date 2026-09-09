#!/usr/bin/env python3
"""Independent reconstruction and complete PC027 single-layer R4 report."""
from device_exp0262 import *
import yaml,hashlib
from exp0240_report import MODULES,coherent_center

def flatten(x,prefix=''):
 out={}
 for k,v in x.items():
  if isinstance(v,dict):out.update(flatten(v,prefix+k+'.'))
  elif isinstance(v,(float,int)) and not isinstance(v,bool):out[prefix+k]=v
 return out

def main():
 preflight();gate=read(R/'numerical_gate.json');assert gate['component_pass'] and gate['full_layer_ideal_R4_gate_pass']
 # Recheck native reference binary against the authority-pinned old seal.
 index=yaml.safe_load((M/'experiments/index.yaml').read_text());entry=next(e for e in index['experiments'] if e['id']=='EXP-0240');old=R.parent/'exp0240';sealpath=old/'EVIDENCE_SHA256.json'
 assert sha(sealpath) in yaml.safe_dump(entry),'reference evidence identity'
 seal=read(sealpath)
 for n in gate['converter_provenance']:
  p=Path(n);assert sha(p)==seal[p.name]['sha256'];assert read(p)['converter_sha256']==gate['independent_converter_sha256']
 assert sha(S/'build/reference/qbh_hmx_u8_reference.so')==gate['independent_converter_sha256']
 data={};allps=[];total=0
 for phase,n in [('short',5),('formal',10)]:
  g=read(R/(phase+'_gate.json'));assert g['integrity_pass'];rows=[]
  for i in range(n):
   for rep in [1,10]:
    for arm in [0,1]:
     p=R/f'{phase}/round{i+1:02d}_r{rep}_a{arm}';ps=[q for q in records(p/'stdout.jsonl') if q.get('record')=='exp0240_profile'];assert len(ps)==rep*9
     physical(ps,arm,False);total+=len(ps);allps.extend(ps)
     gold=read(R/f'audit_a{arm}/validated.json')['output_hashes'];assert [q['output_hash'] for q in ps]==gold*rep
     z=read(p/'validated.json');assert z['prefill_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill') and z['decode_ns']==statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode')
     rows.append(dict(round=i+1,**z))
     cmd=(p/'command.txt').read_text();assert 'QBH_DENSE_R4_AUDIT' not in cmd and 'QBH_DENSE_R3_AUDIT' not in cmd
     if phase=='formal':
      for mode in ['prefill','decode']:
       qs=[q for q in ps if q['mode']==mode];ks=set(flatten(qs[0]))
       ds=[flatten(q) for q in qs];assert all(set(q)==ks for q in ds)
       data.setdefault(f'r{rep}_{mode}_a{arm}',[]).append({k:statistics.mean(q[k] for q in ds) for k in sorted(ks)})
  rng=np.random.default_rng(261)
  for rep in [1,10]:
   for mode in ['prefill_ns','decode_ns']:
    pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [0,1]] for i in range(n)])
    ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);p=g['performance'][f'r{rep}_{mode}'];assert abs(float(np.median(ratios))-p['paired_ratio'])<1e-12 and np.max(abs(ci-p['ci95']))<1e-12
 assert total==2970
 perf=read(R/'formal_gate.json')['performance'];modules={};breakdown={}
 for mode in ['prefill','decode']:
  modules[mode]={}
  for arm in [0,1]:
   center,indices=coherent_center(data[f'r10_{mode}_a{arm}']);host=center['host_wall_ns']/1000
   values={name:sum(center[k] for k in fields)/19.2 for name,fields in MODULES};values['Host–DSP boundary']=host-center['invocation_ticks']/19.2;values['Complete Host wall']=host
   assert abs(sum(v for k,v in values.items() if k!='Complete Host wall')-host)<1e-7
   modules[mode][str(arm)]=dict(round_indices=indices,host_us=host,modules_us=values)
  ds=data[f'r10_{mode}_a1'];breakdown[mode]={k:statistics.median(d[k] for d in ds)/19.2 for k in ['dense_r4_prepare_ticks','dense_r4_matmul_ticks','dense_r4_layout_ticks','dense_r4_finish_ticks']}
 runtime=read(read(R/'runtime_l1.json')['manifest']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
 # All native code in the measured ABI125 binary must remain unchanged.
 subprocess.run(['git','diff','--exit-code',runtime['source_head'],head,'--','src','include','CMakeLists.txt'],cwd=S,check=True)
 physical_summary={k:sorted({q[k] for q in allps}) for k in ['vtcm_requested_bytes','vtcm_acquired_bytes','vtcm_peak_plan_bytes','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','block_invocation_count','ledger_unattributed_ticks','dense_r4_hmx_calls','dense_r4_audit_bytes','dense_r3_mode','dense_r3_optimization']}
 summary=dict(experiment='EXP-0262',source_head=head,runtime_source_head=runtime['source_head'],source_branch='codex/exp-0262-w4u8-r4-native-layout-pipeline',scope='one real layer0 M64+8M1 teacher-input replay; no fullmodel',short_rounds=5,formal_rounds=10,timed_RPCs=total,ABI=125,numerical_gate='pass_R4_components_and_conditional_wholeblock',known_R3_numerical_failure='unchanged',physical_gate='pass',measurement_integrity='pass',speed_eligible=all(p['ci95'][1]<=1.10 for p in perf.values()),stop_before_fullmodel=any(p['stable_over10'] for p in perf.values()),performance=perf,module_centers=modules,R4_breakdown_us=breakdown,physical=physical_summary,baseline_promoted=False,model_quality='not assessed',device_PPL=None,E2E_tokens_per_second=None)
 write(R/'summary.json',summary);write(R/'independent_integrity.json',dict(pass_all=True,timed_RPCs=total,all_output_hashes_exact=True,all_layer_ledgers_checked=True,paired_statistics_independently_recomputed=True,independent_converter_pinned=True,native_source_unchanged=True));write(R/'counter_round_means.json',data)
 lines=['# EXP0262 full-width R4 single-layer cost','','Branch '+summary['source_branch']+'; reporting source '+head+'; native runtime '+runtime['source_head']+' ABI125. Evidence '+str(R)+'; candidate artifacts '+str(O/'r4')+'.','', 'Two paired arms: current optimized R3 OPT2 + wideNR64/nativeW4, and the same path plus full6144 R4 before middle A8. Five short and ten alternating formal pairs, repeat1/repeat10; one prefillM64 and eight consecutive teacher-input M1 steps per repetition. KV starts empty and is computed/appended on device. No frozen-snapshot decode, discarded warmups, selection, extra rounds or fullmodel extrapolation.','', 'Primary medians of round means; paired effect is median within-round ratio, 10000bootstrap seed261. Decode latency averages the eight positions. Additive modules use the identical two median-Host-ranked rounds for all fields. All2970RPCs and layer ledgers independently reconstructed.','', '|Scope|Control us|R4 us|Paired time change|Ratio95% CI|','|---|---:|---:|---:|---|']
 for k,v in perf.items():lines.append(f"|{k}|{v['control_ns']/1000:.3f}|{v['candidate_ns']/1000:.3f}|{100*(v['paired_ratio']-1):+.2f}%|[{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]|")
 lines+=['',f"Speed eligible: {summary['speed_eligible']}; stop before fullmodel: {summary['stop_before_fullmodel']}. Apply the unchanged10percent rule to paired complete Host wall; this result is not an intrinsic R4 lower bound.",'', 'R4 is H12 tensor H512, full6144 coverage. Matrix entries are dense parity/Paley signs; no FWHT/butterfly. Stage1 normalizedH512 operates on12groups, stage2 normalizedH12 mixes allgroups. Intermediate and final HMX casts use declared FP16 normalization constants and rounding. The full transform preserves unquantized xW; deployed Down is freshly RTN quantized from original W R with oneFP32scale/output, signed[-7,7]. OtherC64weights unchanged, no group scales/W4toS8. Middle static range is unchanged, so this fixture establishes cost and implementation correctness, not calibrated accuracy or PPL improvement.','', 'Prepare includes unquantized FP16 SwiGLU LUT gather/packing and constant sign matrix materialization. Layout transposes stage1 for the second GEMM; finish reorganizes/quantizes into native-U8 Down input. Native layout uses exact HVX deal/scatter; finish uses gather, unchanged quantization and native stores. Decode FP16 SwiGLU is produced into the HMX input on Gate/Up readiness. Prefill uses two 256KiB input/output slots; next layout and previous finish are scheduled during the current HMX command. Matmul_ticks is exposed submit/wait cost, not total engine compute. Pipeline HVX ticks report scheduled overlap work, not fully hidden time. Dense factor GEMMs alone cannot stand in for complete cost.','', '|R4 phase, us|M64 prefill|M1 decode|','|---|---:|---:|']
 for k in breakdown['prefill']:lines.append(f"|{k}|{breakdown['prefill'][k]:.3f}|{breakdown['decode'][k]:.3f}|")
 lines+=['','Numerical: fresh original shards verified, H12/H512 orthogonality and explicit dense-factor equivalence, xW identity max1.34e-14, independentW4unpack. All65536SwiGLU LUT entries correctly rounded with Decimal80 and independently certified using mpmath100. All9realsteps HMX eachstage<=1FP16ULP+minnormal; middlequantization, nativeDown and finalresidual exact versus independent arithmetic. Conditional fullblock versus scalarR4 after sameHMXstate max1LSB,mincosine0.999773 passes2/.999. Known idealR3 failure remains; this is not fullmodel quality.','', 'PreR4 live data exact between arms; decode dead padding excluded explicitly, prefill allrows checked. K/V prefill carrier hashes exact and persistent cache metadata advances correctly every invocation. Intra-arm everytimedoutput matches frozen auditedhash. Otherrecipes and baselines unchanged.','', 'Physical: requested/granted8MiB, existing peak plan6682752bytes; phase-dead HMXactivation768KiB and two1MiB expansion arenas reused, no VTCM allocation growth. No timed intermediateDDR/spill/audit, oneFastRPC perlayerstep and oneHMXowner; nativeW4 Down unchangedformat. On device the fullmatrix72MiB is never allocated or fetched; denseH512 512KiB constant materialization and H12 padded2KiB are included. Prefill9/decode2R4 HMXcommands; per-row arithmetic and total tile pairs unchanged. Host-DSP boundary=Hostwall-DSPinvocation perrecord.','', 'Retained recovery: runner_parse_failure.txt records a Python declaration replacement error before staging. Native vector and pipeline audits exact against sealed EXP0261; samebinary original R4 also exact. No numerical relaxation or weight/hash replacement.','', 'Bounded native-layout and pipeline optimization completed. Remaining prefill overhead includes FP16 LUT preparation and gather/quantization traffic. No fullmodel or baseline promotion. Device PPL and R4 E2E token/s: N/A (single layer only).']
 report='\n'.join(lines)+'\n';(R/'REPORT.md').write_text(report)
 overview=[]
 labels={"Host–DSP boundary":"Host–DSP 边界","Complete Host wall":"完整 Host wall"}
 ordered=[name for name,_ in MODULES]+['Embedding','Final model RMSNorm','LM head + greedy','Host–DSP boundary','Complete Host wall']
 for mode in ['prefill','decode']:
  tab=[f'## {mode} repeat10 module overview','','Units us, share complete Host wall; F16/W4A16 N/A because no equivalent layer0 paired measurements. Historical fullmodel timings are not substituted. R4 is included in Gate/Up+SwiGLU activation attribution.','','|模块|F16A16|W4A16|W4A8+R3 control|W4A8+R3+R4|A8相对W4A16增速|','|---|---|---|---:|---:|---|']
  for name in ordered:
   vals=[]
   for arm in ['0','1']:
    c=modules[mode][arm];v=c['modules_us'].get(name);vals.append('N/A: outside layer' if v is None else f"{v:.1f} ({100*v/c['host_us']:.2f}%)")
   tab.append('|'+labels.get(name,name)+'|N/A|N/A|'+'|'.join(vals)+'|N/A|')
  overview+=tab+['']
 (R/'MODULE_TABLES.md').write_text('\n'.join(overview)+'\nE2E token/s: N/A, single-layer experiment.\n')
 lines+=overview
 lines+=['','## Complete normalized numeric counters','','Counts and bytes in native units, *_ticks in qtimer ticks (19.2ticks/us), *_ns in ns. All numeric telemetry retained below, including nested per-layer records. Independent per-field medians do not sum; engine work, DMA and waits can overlap. Legacy unused reference fields are telemetry, not correctness authority; numerical_gate.json is authoritative for implementation comparisons.']
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   a=data[f'r{rep}_{mode}_a0'];b=data[f'r{rep}_{mode}_a1'];assert set(a[0])==set(b[0]);lines += ['',f'### repeat{rep} {mode}','','|Counter|Control|R4|Change|','|---|---:|---:|---:|']
   for k in a[0]:
    x=statistics.median(d[k] for d in a);y=statistics.median(d[k] for d in b);delta=f'{100*(y/x-1):+.3f}%' if x else 'N/A zero denominator';lines.append(f'|{k}|{x:.6f}|{y:.6f}|{delta}|')
 (R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(lines)+'\n')
 docs=S/'docs/experiments';(docs/'EXP-0262-RESULTS.md').write_text(report);(docs/'EXP-0262-PROFILE.md').write_text((R/'FULL_PROFILING_REPORT.md').read_text())
 print('REPORT_PASS',json.dumps(summary['performance']),json.dumps(breakdown),flush=True)
if __name__=='__main__':main()
