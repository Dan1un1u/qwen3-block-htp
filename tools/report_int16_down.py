"""Audit all three-arm rounds and reconstruct complete E2E/module comparisons."""
import sys,json,hashlib,statistics
from pathlib import Path
import numpy as np
MODEL=sys.argv[1];Q=MODEL=='qwen3-block-htp';S=Path('/home/daniuniu/work')/MODEL;R=Path('/mnt/d/llm_exp/results')/MODEL/('exp0284' if Q else 'l32-0040');ID='EXP-0284' if Q else 'L32-0040';ARMS=['A8','SP2','INT16'];LAYERS=28 if Q else 16
sys.path.insert(0,str(S/('scripts' if Q else 'tools')))
if Q:
 from summarize_exp0217 import normalized
 from report_exp0273 import GROUPS
else:
 from report_llama32_pipeline_profile import MODULES as GROUPS

def read(p):return json.loads(p.read_text())
def profiles(p):
 out=[]
 for line in p.read_text().splitlines():
  try:z=json.loads(line)
  except ValueError:continue
  if isinstance(z,dict) and z.get('record')=='generation_profile':out.append(z)
 return out
def module_ticks(p):
 if Q:p=normalized([p]);return [sum(p[k] for k in keys) for _,keys in GROUPS]
 return [sum(p[k] for k in keys)-(p['generation_final_norm_ticks'] if keys==['generation_lm_head_ticks'] else 0) for _,keys in GROUPS]
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
for name in ['single_gate.json','slice_gate.json','full_gate.json']:assert read(R/name)['pass_all']
out=dict(experiment=ID,trajectories={},profiles_checked=0,quality_claim=False,baseline_promoted=False,shared_native_SP2_pipeline=True);native=set();physical=[]
rng=np.random.default_rng(284 if Q else 40);idx=rng.integers(0,10,(20000,10));stdout='stdout.jsonl' if Q else 'stdout.txt'
for trajectory in ['fixed','greedy']:
 formal={a:[] for a in ARMS};rounds={a:[] for a in ARMS}
 for phase,n in [('short',5),('formal',10)]:
  summary=read(R/f'{trajectory}_{phase}.json');assert summary['pass_all'] and len(summary['runs'])==n*3
  for c in range(n):
   pair={}
   for a in ARMS:
    d=R/f'{trajectory}-{phase}/{c:02d}-{a}';v=read(d/'validated.json');cfg=read(d/'protocol.json');native.add(cfg['runtime']['seal']['source_head']);ps=profiles(d/stdout);assert v['pass_all'] and len(ps)==160
    oracle=read(R/f'full-{a}-{trajectory}-audit/validated.json')
    if Q:assert v['selected_codes']==oracle['selected_codes']
    for i,p in enumerate(ps):
     assert p['block_invocation_count']==LAYERS and p['vtcm_requested_bytes']==p['vtcm_acquired_bytes']==8388608 and p['vtcm_peak_plan_bytes']<=8388608
     assert all(p[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode','boundary_ddr_write_bytes'])
     ticks=module_ticks(p);assert sum(ticks)==p['invocation_ticks'];assert p['host_wall_ns']/1000>=sum(ticks)/19.2
    pair[a]=ps;out['profiles_checked']+=len(ps)
    if phase=='formal':formal[a].extend(ps);rounds[a].append(v)
   for j in range(160):
    x,y,z=[pair[a][j] for a in ARMS]
    for k in ['weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count','vtcm_peak_plan_bytes']:assert x[k]==y[k]==z[k],(c,j,k)
    assert y['hmx_u8s8_tile_pair_count']==z['hmx_u8s8_tile_pair_count'];delta=y['hmx_u8s8_tile_pair_count']-x['hmx_u8s8_tile_pair_count'];assert delta==((344064 if Q else 262144) if x['mode']=='prefill' else 0)
    if c==0 and phase=='formal' and j<2:physical.append(dict(trajectory=trajectory,mode=x['mode'],A8_pairs=x['hmx_u8s8_tile_pair_count'],SP2_INT16_pairs=y['hmx_u8s8_tile_pair_count'],weight_bytes=x['weight_ddr_read_bytes'],peak=x['vtcm_peak_plan_bytes'],hmx_commands=x['hmx_command_count']))
 arms={};effects={}
 for a in ARMS:
  arms[a]={}
  for mode in ['prefill','decode']:
   ps=[p for p in formal[a] if p['mode']==mode];host=statistics.mean(p['host_wall_ns']/1000 for p in ps);ticks=np.array([module_ticks(p) for p in ps],dtype='f8').mean(0)/19.2
   rows=[dict(module=name,us=float(us),host_share_pct=float(us/host*100)) for (name,_),us in zip(GROUPS,ticks)]
   boundary=host-float(ticks.sum());assert boundary>=0;rows.extend([dict(module='Host–DSP boundary',us=boundary,host_share_pct=boundary/host*100),dict(module='Complete Host wall',us=host,host_share_pct=100)])
   arms[a][mode]=dict(host_us=host,e2e_tps=(64 if mode=='prefill' else 1)*1e6/host,modules=rows)
 for mode in ['prefill','decode']:
  t={a:np.array([z[mode+'_ns'] for z in rounds[a]]) for a in ARMS};effects[mode]={}
  for num,den in [('SP2','A8'),('INT16','A8'),('INT16','SP2')]:
   ratio=float(t[num].mean()/t[den].mean());ci=np.quantile(t[num][idx].mean(1)/t[den][idx].mean(1),[.025,.975]).tolist();effects[mode][num+'_over_'+den]=dict(wall_ratio=ratio,overhead_pct=100*(ratio-1),ci95=ci,ten_percent_gate=ci[1]<=1.10)
 out['trajectories'][trajectory]=dict(arms=arms,effects=effects)
assert len(native)==1;out.update(measured_source=next(iter(native)),physical=physical,numerical_scope='INT16 independent component/range, selected layer, consecutive3 and whole-model own-reference gates; A8/SP2 prior oracles reproduced; no PPL')
write(R/'SUMMARY.json',out)
lines=[f'# {ID}: A8 / SP2 / Uniform INT16 Down','', 'Only the SwiGLU-to-Down reconstruction grid and its corresponding middle scale change. Native per-channel W4, FP32 residual, all non-Down boundaries and no-rotation settings remain frozen. SP2 and INT16 use the same fused LUT, two-byte physical carrier, native packed-W4 matrix path and pipeline. No new DSP kernel or online conversion. The uniform grid uses ties-to-even and symmetric [-32767,32767] codes; the existing SP2 grid/tie rule is unchanged. Both retain the same nominal frozen clipping alpha, with FP32-stored scales. Ordinary A8 is the original frozen quantizer, not newly recalibrated.','', 'Each trajectory: one warmup per arm, repeat1 auxiliary, five short and ten rotated formal repeat10 rounds. Primary fixed-token and supplementary greedy are separate estimates; no optional stopping. Complete M64+15 Host wall includes embedding, all layers, final norm, head, greedy, FastRPC, excludes cold loading/ADB/external tokenizer. Headline and module times are arithmetic means of the ten round means. Paired bootstrap20000 with fixed seed.','', '| Trajectory | Down input | Prefill token/s | Decode token/s | Prefill Host ms | 15-decode Host ms |','|---|---|---:|---:|---:|---:|']
for trajectory,v in out['trajectories'].items():
 for a,z in v['arms'].items():lines.append(f"| {trajectory} | {a} | {z['prefill']['e2e_tps']:.4f} | {z['decode']['e2e_tps']:.4f} | {z['prefill']['host_us']/1000:.6f} | {z['decode']['host_us']*15/1000:.6f} |")
lines+=['','| Trajectory / phase | Comparison | Host overhead | Paired 95% CI of wall ratio |','|---|---|---:|---|']
for trajectory,v in out['trajectories'].items():
 for mode,comp in v['effects'].items():
  for name,z in comp.items():lines.append(f"| {trajectory} / {mode} | {name} | {z['overhead_pct']:+.3f}% | [{z['ci95'][0]:.6f}, {z['ci95'][1]:.6f}] |")
lines+=['',f"All {out['profiles_checked']} short/formal token-boundary profiles verified. Same weight DDR bytes, HMX command counts and VTCM peak across arms. SP2 and INT16 HMX tile work is equal; relative to A8, extra prefill tile pairs {(344064 if Q else 262144)}, extra decode pairs0. This does not reveal HMX internal W4 implementation.", '', 'Numerical implementation correctness does not establish model quality. No PPL or quality acceptance. This comparison measures efficient support for the software-prescribed SP2 representation; it does not establish SP2 superiority over uniform INT16.', '', 'The reused internal flag remains SP2 mode8 for both SP2 and INT16; manifests, LUTs, scales and arm labels distinguish the two quantization contracts. Failed tooling attempts, if any, remain in the evidence directory; see closure notes.']
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
lines=['# Complete module tables','', 'Microseconds and share of complete Host wall; decode is per token. All formal rounds included.']
for trajectory,v in out['trajectories'].items():
 for mode in ['prefill','decode']:
  lines+=['',f'## {trajectory} / {mode}','','| Module | A8 | SP2 | Uniform INT16 |','|---|---:|---:|---:|']
  for rows in zip(*(v['arms'][a][mode]['modules'] for a in ARMS)):
   lines.append('| '+rows[0]['module']+' | '+' | '.join(f"{z['us']:.1f} ({z['host_share_pct']:.2f}%)" for z in rows)+' |')
with (R/'MODULE_TABLES.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({t:v['effects'] for t,v in out['trajectories'].items()},indent=2))
