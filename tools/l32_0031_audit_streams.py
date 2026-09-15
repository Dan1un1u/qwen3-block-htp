import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from run_llama32_frontend import records as parse_records
def records(p):return parse_records(Path(p).read_text())
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0031')
count=0;seals=set();steps=[]
for phase,n in [('short',5),('formal',10)]:
 for c in range(n):
  arm=[]
  for a,mask in [('ALL',0),('SPLIT',4)]:
   d=R/f'l16-{phase}-a01/{c:02d}-{a}';cfg=json.loads((d/'protocol.json').read_text());seal=cfg['runtime']['seal'];assert not seal['paper_trace']
   for name,h in seal['files'].items():
    p=Path(cfg['runtime']['archive'])/Path(name).name
    if str(p) not in seals:assert hashlib.sha256(p.read_bytes()).hexdigest()==h;seals.add(str(p))
   ps=[v for v in records(d/'stdout.txt') if v.get('record')=='generation_profile'];assert len(ps)==160
   for v in ps:
    assert v['paper_format_disable']==mask and v['paper_pipeline_disable']==0
    assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
    assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode'])
   arm.append(ps);count+=len(ps)
  for x,y in zip(*arm):
   assert x['mode']==y['mode']
   for k in ['weight_ddr_read_bytes','w4u8_decode_direct_n_weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count']:assert x[k]==y[k],k
   delta=y['hmx_u8s8_tile_pair_count']-x['hmx_u8s8_tile_pair_count'];assert delta==(262144 if x['mode']=='decode' else 0),(x['mode'],delta)
   if phase=='formal' and c==0:steps.append(dict(mode=x['mode'],all_pairs=x['hmx_u8s8_tile_pair_count'],split_pairs=y['hmx_u8s8_tile_pair_count'],weight_bytes=x['weight_ddr_read_bytes']))
for n in ['single_gate.json','slice_gate.json','full_gate.json']:assert json.loads((R/n).read_text())['pass_all']
z=dict(pass_all=True,timed_profiles=count,paired_profiles=count//2,extra_decode_pairs_per_layer=16384,extra_decode_pairs_full=262144,weight_bytes_identical=True,worker_commands_identical=True,physical_gate=True,independent_numerical_exact=True,quality_claim=False,first_formal_steps=steps)
with (R/'FINAL_AUDIT.json').open('x') as f:json.dump(z,f,indent=2)
print({k:v for k,v in z.items() if k!='first_formal_steps'})
