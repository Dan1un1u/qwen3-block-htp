#!/usr/bin/env python3
"""Independent full-model profiling closure, PC079 repeat10 gate only."""
from device_exp0265 import *
from exp0240_report import coherent_center
from summarize_exp0217 import normalized
from measure_exp0218 import OVERVIEW
import yaml

def flat(x,prefix=''):
 d={}
 for k,v in x.items():
  if isinstance(v,dict):d.update(flat(v,prefix+k+'.'))
  elif type(v) in [int,float]:d[prefix+k]=v
 return d

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def val(center,fields):return sum(center[k] if k.endswith('_us') else center[k]/19.2 for k in fields)

def main():
 preflight();assert read(R/'slice_gate.json')['pass_all'] and read(R/'full_gate.json')['pass_all'];allps=[];data={};counters={};loops={};calls=0
 for phase,n in [('short',5),('formal',10)]:
  gate=read(R/(phase+'_gate.json'));assert gate['integrity_pass'];rows=[]
  for i in range(n):
   for rep in [1,10]:
    for arm in [0,1]:
     path=R/f'{phase}/round{i+1:02d}_r{rep}_a{arm}';rs=records(path/'stdout.jsonl');ps=[q for q in rs if q.get('record')=='generation_profile'];fs=[q for q in rs if q.get('generation_sequence_complete')];assert len(ps)==16*rep and len(fs)==rep;physical(ps,28,arm,6,False)
     gold=read(R/('full_control' if arm==0 else 'full_previous')/'validated.json');codes=[(q['selected_token_id'],q['selected_logit_half_bits']) for q in rs if 'selected_logit_half_bits' in q];assert codes==[tuple(x) for x in gold['selected_codes']]*rep
     for j,f in enumerate(fs):
      assert f['all_steps_pass'] and f['token_ids']==gold['token_ids'];assert f['total_host_wall_ns']==sum(p['host_wall_ns'] for p in ps[j*16:j*16+16]);assert f['generation_loop_wall_ns']>=f['total_host_wall_ns']
     for p in ps:assert p['repeat_count']==1 and p['backend']=='standalone_fastrpc_dsp' and p['qnn']=='none'
     calls+=len(ps);allps.extend(ps);z=read(path/'validated.json');assert z['prefill_ns']==statistics.mean(p['host_wall_ns'] for p in ps if p['mode']=='prefill') and z['decode_ns']==statistics.mean(p['host_wall_ns'] for p in ps if p['mode']=='decode');rows.append(dict(round=i+1,**z));cmd=(path/'command.txt').read_text();assert 'QBH_DENSE_R4_AUDIT' not in cmd and 'QBH_DENSE_R3_AUDIT' not in cmd
     if phase=='formal':
      key=f'a{arm}_r{rep}';data.setdefault(key,{});counters.setdefault(key,{})
      for mode in ['prefill','decode']:
       qs=[q for q in ps if q['mode']==mode];data[key].setdefault(mode,[]).append(normalized(qs));fsq=[flat(q) for q in qs];assert all(set(q)==set(fsq[0]) for q in fsq);counters[key].setdefault(mode,[]).append({k:statistics.mean(q[k] for q in fsq) for k in sorted(fsq[0])})
      v=loops.setdefault(key,dict(model_ns=[],loop_ns=[],startup=[]));v['model_ns'].append(statistics.mean(f['total_host_wall_ns'] for f in fs));v['loop_ns'].append(statistics.mean(f['generation_loop_wall_ns'] for f in fs));v['startup'].append(next(q for q in rs if q.get('record')=='generation_startup'))
  rng=np.random.default_rng(265)
  for rep in [1,10]:
   for mode in ['prefill_ns','decode_ns']:
    pairs=np.array([[next(z[mode] for z in rows if z['round']==i+1 and z['repeat']==rep and z['arm']==a) for a in [0,1]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);v=gate['performance'][f'r{rep}_{mode}'];assert abs(np.median(ratios)-v['paired_ratio'])<1e-12 and np.max(abs(ci-v['ci95']))<1e-12
  assert gate['repeat10_speed_eligible']==all(v['ci95'][1]<=1.1 for k,v in gate['performance'].items() if k.startswith('r10_'))
 assert calls==5280
 # Historical full-model columns retain sealed source files, never substitute a layer.
 hist=read(R.parent/'exp0259/historical_provenance.json');sealed(259,['historical_provenance.json']);hs={'F16A16':[],'W4A16':[]}
 for name,h in sorted(hist.items()):
  p=Path(name);assert sha(p)==h;qs=[q for q in records(p) if q.get('record')=='generation_profile' and q['mode']=='prefill'];assert qs and all(q['block_invocation_count']==28 for q in qs);hs['F16A16' if 'exp0218' in name else 'W4A16'].append(normalized(qs))
 assert all(len(v)==10 for v in hs.values());write(R/'historical_provenance.json',hist)
 centers={};through={};breakdown={}
 for key,ds in data.items():
  centers[key]={}
  for mode in ['prefill','decode']:
   c,ix=coherent_center(ds[mode]);mods={name:val(c,fields) for name,fields in OVERVIEW};assert abs(sum(list(mods.values())[:-1])-c['host_us'])<1e-6;centers[key][mode]=dict(round_indices=ix,host_us=c['host_us'],modules_us=mods)
  pre=centers[key]['prefill']['host_us'];dec=centers[key]['decode']['host_us'];model=statistics.median(loops[key]['model_ns']);loop=statistics.median(loops[key]['loop_ns']);through[key]=dict(prefill_tokens=64,prefill_host_us=pre,prefill_tokens_per_second=64e6/pre,decode_tokens=15,decode_host_us=dec*15,decode_host_us_per_token=dec,decode_tokens_per_second=1e6/dec,selected_output_tokens=16,complete_model_host_us=model/1000,complete_model_effective_tokens_per_second=16e9/model,generation_loop_us=loop/1000,generation_loop_tokens_per_second=16e9/loop,cold_startup_median_ns={k:statistics.median(x[k] for x in loops[key]['startup']) for k in loops[key]['startup'][0] if k!='record'})
  if key.startswith('a1'):breakdown[key]={mode:{k:statistics.median(x['dense_r4_'+k+'_ticks'] for x in ds[mode])/19.2 for k in ['prepare','matmul','layout','finish','parallel_work','parallel_join']} for mode in ['prefill','decode']}
 hc={k:coherent_center(v)[0] for k,v in hs.items()};labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE（含R3）','QK–Softmax–AV','O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU（含R4）','Down','Final residual','KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall'];module_rows=[]
 for label,(name,fields) in zip(labels,OVERVIEW):
  vals=[val(hc[k],fields) for k in ['F16A16','W4A16']]+[centers['a1_r10']['prefill']['modules_us'][name]];hosts=[hc[k]['host_us'] for k in ['F16A16','W4A16']]+[centers['a1_r10']['prefill']['host_us']];cells=[f'{v:.1f} ({100*v/h:.2f}%)' for v,h in zip(vals,hosts)];module_rows.append([label,*cells,f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else 'N/A：零分母'])
 compact=table(['模块','F16A16 EXP0218','W4A16 EXP0166','W4A8＋R3＋R4 EXP0265','A8相对W4A16增速'],module_rows);(R/'MODULE_TABLE.md').write_text(compact+'\n')
 perf=read(R/'formal_gate.json')['performance'];eligible=all(v['ci95'][1]<=1.1 for k,v in perf.items() if k.startswith('r10_'));summary=dict(experiment='EXP-0265',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),native_runtime=read(read(R/'runtime_l28.json')['manifest']),ABI=127,actual_layer_count=28,short_rounds=5,formal_rounds=10,timed_fullmodel_RPCs=calls,layer_ledgers_checked=calls*28,repeat10_speed_eligible=eligible,repeat1_auxiliary_only=True,performance_policy='PC079',performance=perf,throughput=through,module_centers=centers,R4_breakdown_us=breakdown,numerical_implementation_gate='pass_frozen_R4_exact_OPT3_and_independent_slice_reference',known_ideal_R3_gate='failed_unchanged_diagnostic_scope',physical_gate='pass',measurement_integrity='pass',model_quality='R4 sample does not answer France capital; no PPL evaluation',device_PPL=None,baseline_promoted=False,text_output=read(R/'text_output.json'))
 write(R/'summary.json',summary);write(R/'counter_round_means.json',counters);write(R/'generation_loop_times.json',loops);write(R/'independent_integrity.json',dict(pass_all=True,timed_fullmodel_RPCs=calls,layer_ledgers=calls*28,all_selected_tokens_and_codes_exact=True,all_physical_and_KV_checks=True,statistics_independently_recomputed=True,repeat1_no_veto=True,historical_hashes_verified=len(hist)))
 intro=['# EXP0265 optimized dense R4 full-model diagnostic E2E','','User-approved PC079 changes performance gating to formal paired repeat10 only; repeat1 stays auxiliary and cannot veto. Sealed EXP0264 is eligible under this new rule, its original recorded classification is unchanged. Three-layer gates completed before this28-layer integration.','', 'Frozen R3 OPT2 + wideNR64/native per-channel W4 control versus same path with full6144 dense R4 OPT6. Fresh original Down W R quantized RTN[-7,7]/singleFP32scale peroutput for28layers; otherC64 weights, head/embedding/norms, staticA8 ranges and fixed rotatedEOSseed unchanged. Full Hadamard is H12 tensor H512 via two dense HMX factors, explicit FP16 normalization and intermediate rounding; no butterfly or groupquantization. Prefix is the existing frozen R3 diagnosticseed, not recomputed through R4. This costfixture has no R4 recalibration or quality-optimized Down quantizer.','', 'Export:28 originalFP64 xW invariance checks, independentW4unpack, layer0 byteexact toEXP0261,65536SwiGLU LUT entries perlayer Decimal80/independentmpmath100. Changed87 files; all906 packagefiles verified locally/remotely and parent/original hashes retained. Threeconsecutive layers,9steps: noR4 reproduces sealedEXP0259; OPT3,OPT6 and repeat exact. Exposed lastlayer actualR4 factors compareFloat64; SDKDown/residual exact; conditionalidealR4 outputmax1LSB/cos>=.999. All finiteFP16quantization checked perlayer inuntimed audits. KnownidealR3 wholelayerfailure unchanged, no PPL or model-quality acceptance.','', 'Full28layers OPT3/OPT6/audit/repeat produce identical16feedbacktokens and selectedlogitcodes. Fullaudit exercises all63488finiteFP16values on all28layers eachstep. Only selected maxima codes are compared in generation, not fullvocabulary logits. Five short and ten alternatingformal pairs repeat1/10,5280fullmodelRPCs and147840layerledgers verified. Everysequence starts fresh then appends persistentKV continuously. OneFastRPC perfullmodelstep;8MiB requested/granted,zero timedintermediateDDR/spill/audit, no interlayerhiddenDDR and a singleHMXowner.','', 'Primary values are medians of ten round means. Effect is median within-round time ratio with10000pairedbootstrap seed265. No outlier deletion, optional resampling or post-result tuning. Each module table uses the identical two median-Host-ranked rounds for allfields within eacharm; rows close toHostwall. Historical F16 EXP0218 and Selected W4A16 EXP0166 are nonpaired matchingfullmodel M64 references, not currentpairedcontrols.','',table(['Scope','R3 control us','R3+R4 us','Paired change','Ratio95%CI','Role'],[[k,f'{v["control_ns"]/1000:.3f}',f'{v["candidate_ns"]/1000:.3f}',f'{100*(v["paired_ratio"]-1):+.2f}%',f'[{v["ci95"][0]:.6f},{v["ci95"][1]:.6f}]','auxiliary' if k.startswith('r1_') else 'gate'] for k,v in perf.items()]),'',f'Repeat10 speedeligible: {eligible}. Repeat1 cannot veto. Numerical/physical/evidence rules remain unchanged; no baseline promotion.','',compact]
 text=read(R/'text_output.json');intro+=['','Text sample, before first EOS (timing deliberately continues16outputs afterEOS for fixedworkload, so later fragments are not scored):','','Prompt: What is the capital of France? Answer briefly.','','R3: '+text['R3']['raw_text'].split('<|im_end|>')[0],'','R3+R4: '+text['R3_R4']['raw_text'].split('<|im_end|>')[0],'','R4 produces decodable text but does not answer this question beforeEOS. This is a semantic failure for this sample, not a formatting failure. No broader quality or PPL conclusion from one speedfixture prompt.','', 'R4 phase wall telemetry is included in Gate/Up+SwiGLU. Matmul is exposedsubmit/wait, not totalengine compute. Aggregateparallelwork overlapswall and mustnotbeadded; no component extrapolation toE2E.']
 for rep in [1,10]:intro+=['',table(['R4 phase us',f'r{rep} prefill',f'r{rep} decode'],[[k,f'{breakdown[f"a1_r{rep}"]["prefill"][k]:.3f}',f'{breakdown[f"a1_r{rep}"]["decode"][k]:.3f}'] for k in breakdown[f'a1_r{rep}']['prefill']])]
 end=['','Actual E2E (completeHost;64prefill includes fixedEOS,15continuousdecode afterfirst output):','',table(['arm/repeat','prefill64 tok/s','decode15 tok/s','16outputs/modelHost tok/s','16outputs/generationloop tok/s'],[[k,*[f'{v[n]:.3f}' for n in ['prefill_tokens_per_second','decode_tokens_per_second','complete_model_effective_tokens_per_second','generation_loop_tokens_per_second']]] for k,v in through.items()]),'', 'Allthroughput uses measured complete embedding/28layers/finalnorm/nativeW4head/greedy/FastRPC. Hostdenominators and token counts in summary.json; generationloop additionally includes ondeviceHostloop and profilingserialization. Hot inference excludes tokenizer/detokenizer,ADB and modelload; coldstartup separately recorded. No perlayer extrapolation or PPL.']
 (R/'REPORT.md').write_text('\n'.join(intro+end)+'\n');full=intro+['','Complete numeric fields: *_ticks19.2ticks/us; *_ns nanoseconds; counts/bytes raw. Counter medians below are independent and need not sum, overlapping engines are not additive. Nested layer fields retained.']
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   a=counters[f'a0_r{rep}'][mode];b=counters[f'a1_r{rep}'][mode];assert set(a[0])==set(b[0]);full+=['',f'## r{rep} {mode}','',table(['Field','R3 control','R3+R4','Difference'],[[k,f'{statistics.median(x[k] for x in a):.6f}',f'{statistics.median(x[k] for x in b):.6f}',f'{statistics.median(x[k] for x in b)-statistics.median(x[k] for x in a):+.6f}'] for k in a[0]])]
 (R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(full+end)+'\n')
 for a,b in [('REPORT.md','EXP-0265-RESULTS.md'),('FULL_PROFILING_REPORT.md','EXP-0265-PROFILE.md')]: (S/'docs/experiments'/b).write_bytes((R/a).read_bytes())
 print('REPORT_PASS',json.dumps(dict(performance=perf,throughput=through,eligible=eligible)),flush=True)
if __name__=='__main__':main()
