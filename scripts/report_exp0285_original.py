import sys,json,statistics
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/qwen3-block-htp');R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0285/original-paired')
sys.path.insert(0,str(S/'scripts'))
from summarize_exp0217 import normalized
from report_exp0273 import GROUPS
from device_exp0285 import records
def read(n):return json.loads((R/n).read_text())
def dump(n,v):
 with (R/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
arms=['FP32','FP16'];raw={a:[] for a in arms};rounds={a:[] for a in arms};native=set();profiles_checked=0
for phase,n in [('short',5),('formal',10)]:
 z=read(phase+'.json');assert z['pass_all'] and len(z['runs'])==n*2
 for cycle in range(n):
  pair={}
  for a in arms:
   d=R/f'{phase}/{cycle:02d}-{a}';v=json.loads((d/'validated.json').read_text());protocol=json.loads((d/'protocol.json').read_text())
   native.add(protocol['source_head']);ps=[p for p in records(d/'stdout.jsonl') if p.get('record')=='generation_profile'];assert len(ps)==160
   assert v['selected_codes']==json.loads((R.parent/f'full-{a}-audit-a01/validated.json').read_text())['selected_codes']
   for p in ps:
    assert p['fp32_residual']==(2 if a=='FP32' else 3)
    assert p['block_invocation_count']==28 and p['vtcm_requested_bytes']==p['vtcm_acquired_bytes']==8388608
    assert all(p[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode','boundary_ddr_write_bytes'])
    z=normalized([p]);assert sum(sum(z[k] for k in keys) for _,keys in GROUPS)==p['invocation_ticks']
   profiles_checked+=len(ps);pair[a]=ps
   if phase=='formal':raw[a]+=ps;rounds[a].append(v)
  for x,y in zip(pair['FP32'],pair['FP16']):
   for k in ['weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count','hmx_u8s8_tile_pair_count','vtcm_peak_plan_bytes']:
    assert x[k]==y[k],(phase,cycle,k,x[k],y[k])
assert len(native)==2
out=dict(experiment='EXP-0285',model='Qwen3-1.7B',recipe='native W4A8 SP2 no rotation',measured_sources=sorted(native),profiles_checked=profiles_checked,quality_evaluated=False,baseline_promoted=False,arms={},effects={},auxiliary=read('auxiliary.json'),reserved_hole_bytes=262144,live_residual_bytes={'FP32':524288,'FP16':262144},vtcm_peak_plan_bytes=max(v['vtcm_peak_plan_bytes'] for v in raw['FP16']))
rng=np.random.default_rng(285);idx=rng.integers(0,10,(20000,10))
for a in arms:
 out['arms'][a]={}
 for mode in ['prefill','decode']:
  ps=[p for p in raw[a] if p['mode']==mode];host=statistics.mean(p['host_wall_ns']/1000 for p in ps);values=[]
  for p in ps:
   z=normalized([p]);values.append([sum(z[k] for k in keys)/19.2 for _,keys in GROUPS])
  ticks=np.mean(values,axis=0);rows=[dict(module=n,us=float(t),host_share_pct=float(t/host*100)) for (n,_),t in zip(GROUPS,ticks)]
  bound=host-float(ticks.sum());assert bound>=0
  rows+=[dict(module='Host-DSP boundary',us=bound,host_share_pct=bound/host*100),dict(module='Complete Host wall',us=host,host_share_pct=100)]
  out['arms'][a][mode]=dict(host_us=host,e2e_tps=(64 if mode=='prefill' else 1)*1e6/host,modules=rows)
for mode in ['prefill','decode']:
 x=np.array([z[mode+'_ns'] for z in rounds['FP32']]);y=np.array([z[mode+'_ns'] for z in rounds['FP16']])
 ratio=float(y.mean()/x.mean());ci=np.quantile(y[idx].mean(1)/x[idx].mean(1),[.025,.975]).tolist()
 out['effects'][mode]=dict(wall_ratio=ratio,wall_change_pct=(ratio-1)*100,tps_change_pct=(1/ratio-1)*100,wall_ratio_ci95=ci,ten_percent_gate=ci[1]<=1.1)
dump('SUMMARY.json',out)
lines=['# EXP-0285: Original latest FP32 baseline versus FP16 residual','',
'Qwen3-1.7B latest no-rotation native-W4 SP2 baseline, parent EXP0284 SP2 arm. Original sealed EXP0284 binary mode2 FP32 control versus EXP0285 binary mode3 FP16 storage; projection scale/add and ordered RMSNorm remain FP32. Norm still produces A8. Embedding begins in FP16; both residual additions round to FP16 at storage in candidate. No rotations, quantizer fitting, weight changes or model-quality/PPL evaluation. No baseline promotion. This original-binary comparison is primary; the first same-binary comparison remains supplementary because common storage dispatch slowed its FP32 control.','',
'Primary fixed original M64 prompt +15 decode steps, with identical token trajectory. Complete Host wall includes embedding,28layers, finalNorm, full head,greedy and FastRPC; excludes cold loading,tokenizer andADB. One warmup and one auxiliaryrepeat1 per arm, five short and ten alternatingAB/BA formalrepeat10 pairs, no optional stopping. Headline/module values use arithmetic mean of all formal measurements. Paired bootstrap20000 seed285.','',
'| Residual storage | Prefill token/s | Decode token/s | Prefill Host ms | 15-decode Host ms |',
'|---|---:|---:|---:|---:|']
for a in arms:
 z=out['arms'][a];lines.append(f"| {a} | {z['prefill']['e2e_tps']:.4f} | {z['decode']['e2e_tps']:.4f} | {z['prefill']['host_us']/1000:.6f} | {z['decode']['host_us']*15/1000:.6f} |")
lines+=['','| Phase | FP16 Host wall change | Throughput change | Paired95% wall-ratio CI |','|---|---:|---:|---|']
for mode,z in out['effects'].items():lines.append(f"| {mode} | {z['wall_change_pct']:+.3f}% | {z['tps_change_pct']:+.3f}% | [{z['wall_ratio_ci95'][0]:.6f}, {z['wall_ratio_ci95'][1]:.6f}] |")
lines+=['',
'Independent implementation checks: actual layer14 prefill/decode and chain3 outputs exact; Q/K, AV, postNorm, Gate/Up and low/high SP2 live carriers exact; repeat10 deterministic. Full28 fixed trajectory: all448 per-layer FP16 byte hashes exact to independent CPU, all16 final hidden and normalized vectors exact, full-vocabulary head selected IDs/codes exact. FP32 full-model control88 exported files exact against sealed EXP0284. These establish implementation correctness, not model quality.',
f"All {profiles_checked} short/formal token profiles have complete additive ledgers and declared physical gates. Both arms have equal weight DMA bytes, HMX commands/tile pairs and VTCM arena peak {out['vtcm_peak_plan_bytes']} bytes; grant8388608. No explicit timed intermediate DDR or spills reported by runtime counters.",
'',
'Memory qualification: FP16 live residual occupies262144 bytes versus524288 for M64; it is physically packed, not FP32 with fake rounding. A reserved unused262144-byte hole keeps subsequent HMX/DMA addresses identical to the control. Total VTCM plan is therefore NOT reduced in this experiment. Initial compact allocation caused a cDSP HMX memory access fault at raw native-W4 QKV accumulation (BadVA0xFF3FE800); restoring downstream placement fixed it. Exact hardware address constraint not isolated, no general bank-boundary claim. Full compaction is future work.',
'',
'Compiler qualification: both inherited FP32 and candidate assembly contain vector spills on worker stacks; runtime intermediate-DDR counters do not instrument compiler stack traffic. No full-sized FP32 residual tensor or explicit conversion buffer is introduced. Do not interpret zero reported spills as proof of zero compiler stack traffic.',
'',
'Retained recovery evidence: initial build script permission invocation corrected by explicit bash; missing global NumPy corrected using existing quantization environment; helper forward declarations fixed; first two candidate device attempts faulted before producing outputs. Header-rejection hypothesis was disproved by logcat DSP_RUNNING/precise HMX exception. Padded-placement candidate passes independent references. First CPU hash export used standard FNV offset instead of project-specific basis; a02 recomputed all hashes from the independent CPU trajectory, old results retained. Original-baseline fairness check additionally found altered generated code slowed the new FP32 branch. A fresh fixed5short/10formal campaign pairs the unchanged original sealed binary with the unchanged candidate; both output references and binary hashes are verified. Initial same-binary campaign is retained in the parent directory, not pooled. No threshold relaxation or discarded timing rounds.',
'',
'Repeat1 auxiliary timing is retained in SUMMARY.json; only repeat10 formal timing supports the speed conclusion.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
lines=['# Complete FP32 / FP16 residual module tables','','Microseconds; percentage of complete Host wall. Decode is per token.']
for mode in ['prefill','decode']:
 lines+=['',f'## {mode}','','| Module | FP32 residual | FP16 residual |','|---|---:|---:|']
 for a,b in zip(out['arms']['FP32'][mode]['modules'],out['arms']['FP16'][mode]['modules']):
  lines.append(f"| {a['module']} | {a['us']:.2f} ({a['host_share_pct']:.2f}%) | {b['us']:.2f} ({b['host_share_pct']:.2f}%) |")
(R/'MODULE_TABLES.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:out[k] for k in ['effects','profiles_checked','vtcm_peak_plan_bytes']},indent=2))
