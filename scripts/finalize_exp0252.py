#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,subprocess,statistics,math
import numpy as np
import device_exp0252 as d
from exp0240_report import FIELDS
S=d.S;R=d.R;M=Path(str(S)+'-project-memory')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=S,text=True).strip()
def read(n):return json.loads((R/n).read_text())
def write(n,z):
 with (R/n).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def records(tag):return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if l.startswith('{') and json.loads(l).get('record')=='exp0240_profile']
d.preflight();sw=read('software_summary.json');hw=read('hardware_summary.json');aud=read('numerical_audit.json');pr=read('formal_provenance.json');assert not git('status','--porcelain')
assert aud['eligible']=={'control':True,'nr64':True,'r3_control':False,'r3_nr64':False}
assert not git('diff',pr['source_head'],'HEAD','--','src','include','CMakeLists.txt')
assert not git('diff','a6a8326fb694bab2ec3659c02b77a3d459527394','HEAD','--','scripts/attention_exp0252.py','scripts/integer_attention_exp0252.py','scripts/data_exp0252.py')
assert read('nr64_denominator_check.json')['entries']==37224464 and read('nr64_denominator_check.json')['max_probability_lsb']==1
for n in ['checks/numerical.json','independent_data_audit.json','development_complete.json','collection_pass.json']:assert read(n)['pass_all']
for n in ['F','C64','OFF_wide_sole','OFF_wide_exact','R3_wide_sole','R3_wide_exact']:assert read('checks/reproduction_'+n+'.json')['exact']
for phase in ['development','final']:
 for recipe in ['F','C64']:assert read(f'checks/{phase}_{recipe}_weights.json')['unchanged']
for f in ['dataset_freeze.json','freeze.json']:
 z=read(f);assert z['protocol_sha256']==sha(M/'docs/experiments/EXP-0252.md')
 for n,h in z['files'].items():assert sha(R/n)==h
 for n,h in z['references'].items():assert sha(n)==h
for name,ppl in sw['final_ppl'].items():
 score=read('scores/final_'+name+'.json');assert score['freeze_sha256']==sha(R/'freeze.json');assert score['checks']['repeat_exact'] and score['checks']['causal_exact'] and score['checks']['prefix_immutable']
 assert abs(math.exp(statistics.mean(v for row in score['samples'] for v in row['nll']))-ppl)<1e-9
calls=0
for phase,n in [('short',5),('formal',10)]:
 for i in range(n):
  for rep in [1,10]:
   for cell in ['control','nr64']:
    rs=records(f'{phase}_{i:02d}_r{rep}_{cell}');gold=records('audit_'+cell);assert len(rs)==rep*9
    for j,x in enumerate(rs):
     assert x['output_hash']==gold[j%9]['output_hash'] and x['wide_score_mode']==(1 if cell=='control' else 4)
     assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']==6682752
     assert x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==x['intermediate_spill_fill_count']==x['u8_attention_audit_ddr_write_bytes']==0
     assert x['block_invocation_count']==1 and x['w4u8_decode_direct_n_projection_count']==7 and x['scan_total_kv_length']==64+j%9 and x['dsp_status']==3
     assert sum(x[k] for k in FIELDS)==x['invocation_ticks']==x['ledger_named_ticks'] and x['ledger_unattributed_ticks']==0
    calls+=len(rs)
assert calls==2970
rng=np.random.default_rng(252)
for rep in [1,10]:
 for mode in ['prefill','decode']:
  vals={c:np.array([statistics.mean(x['host_wall_ns']/1000 for x in records(f'formal_{i:02d}_r{rep}_{c}') if (x['replay_step']==0)==(mode=='prefill')) for i in range(10)]) for c in ['control','nr64']};ratios=vals['nr64']/vals['control'];v=hw['performance'][f'repeat{rep}_{mode}']
  assert abs(float(np.median(vals['control']))-v['control_us'])<1e-8 and abs(float(np.median(vals['nr64']))-v['nr64_us'])<1e-8
  assert abs((float(np.median(ratios))-1)*100-v['paired_regression_percent'])<1e-8
  ci=np.quantile(np.median(ratios[rng.integers(0,10,(10000,10))],axis=1),[.025,.975]);assert np.max(abs(ci-v['paired_ratio_ci95']))<1e-10
assert hw['speed_gate']=='pass'
for arms in hw['module_centers'].values():
 for m in arms.values():assert abs(sum(v for k,v in m['modules_us'].items() if k!='Complete Host wall')-m['host_us'])<1e-8
pins={'exp0250':'e82e72cec87543319561493abd975cf828bcdea1b2ee46b18c57fcb53e206b0a','exp0251':'e3d088b0ece57e5c36baa3a0ac3098306e950af73d4d162f7495fccf4afc7334'};parents={}
for exp,pin in pins.items():
 root=R.parent/exp;assert sha(root/'EVIDENCE_SHA256.json')==pin;z=json.loads((root/'EVIDENCE_SHA256.json').read_text())
 for name,v in z['files'].items():assert sha(root/name)==v['sha256']
 parents[exp]=dict(ledger_sha256=pin,verified_files=len(z['files']))
models={}
for cell,pin in [('control','8f42f9e07f49d90504f8845e9f01ffc796c133c567794d8bf15af8778f93eaff'),('r3','c196670085429fb749cd2c391fa3d4c79eee9c3644a222cce3def002475609f1')]:
 root=d.MODELS/cell;assert sha(root/'manifest.json')==pin;z=json.loads((root/'manifest.json').read_text())
 for name,v in z['files'].items():assert sha(root/name)==v['sha256']
 models[cell]=dict(manifest_sha256=pin,files=len(z['files']))
for path in d.binaries():assert sha(path)==pr['binaries'][path.name]==sha(R/'binaries/attempt1'/path.name)
remote=d.adb('shell',f'cd {d.REMOTE} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so');assert all(h in remote for h in pr['binaries'].values());assert d.adb('shell','cat /proc/sys/kernel/random/boot_id')==pr['boot']
checks=dict(pass_all=True,NR64_max_probability_LSB=1,denominator_exponent_entries=37224464,software_old_controls_reproduced=6,final_data_reconstruction_and_disjoint=True,all_final_PPLs_recomputed=True,all_timed_outputs_physical_and_ledger_pass=True,timed_RPCs=calls,audit_RPCs=117,statistics_and_CI_independently_recomputed=True,native_source_and_binary_immutable=True,software_core_unchanged_since_first_inference=True,parent_evidence=parents,models=models)
write('independent_integrity_checks.json',checks)
write('ARTIFACT_PROVENANCE.json',dict(source_head=git('rev-parse','HEAD'),runtime=pr,inference_source_heads=sw['inference_source_heads'],protocol_sha256=sha(M/'docs/experiments/EXP-0252.md'),models=models,parent_evidence=parents,ABI=116,toolchain='SDK6.6 Tools19.0.07 V79 NDKr26c',constant_reciprocal_table_bytes=256))
summary=dict(experiment='EXP-0252',execution_state='completed',evidence_validity='valid',local_gate='pass',gate_role='NR64_arithmetic_and_no_R3_single_layer_speed_only',adoption_status='not_applicable',source_head=git('rev-parse','HEAD'),runtime_source_head=pr['source_head'],inference_source_heads=sw['inference_source_heads'],software=sw,hardware=hw,R3_numerical_gate='fail',R3_prefill_final_max_LSB={m:rows[0]['boundaries']['final']['max_lsb'] for m,rows in aud['r3_amplification'].items()},baseline_promoted=False,device_PPL=None,E2E_tokens_per_second=None,next_action='discuss_remaining_R3_amplification_and_A8_quality_before_integration')
write('summary.json',summary)
lines=['# EXP-0252 normalization and dense R3 joint ablation','','NR64 (fixed64-bin Q30 reciprocal plus one integer Newton step) approaches exact integer normalization, passes independent hardware arithmetic and no-R3 speed gates. Dense R3 whole-layer gate remains failed even with exact normalization. No promotion, slice or full-model device execution.','',f"Source {summary['source_head']}; runtime {pr['source_head']}. Evidence {R}. Frozen original per-output-channel W4 and EXP0246 calibration/prefix/attention scales unchanged. C64 names the retained W4A16 quantization reference; this is not group64 deployment.",'', '## Independent software PPL','', 'Fresh128documents/2048targets,32 per en/zh wiki/news cell; all8 arms frozen before scoring. Existing6 development controls reproduce EXP0250 per-token scores. PPL is conditional software arithmetic, not DSP PPL.','', '|Configuration|PPL|Ratio vs F16|','|---|---:|---:|']
for name,ppl in sw['final_ppl'].items():lines.append(f"|{name}|{ppl:.6f}|{ppl/sw['final_ppl']['F']:.4f}|")
for group in ['OFF','R3']:
 v=sw['ladder'][group]['wide_sole->wide_nr64'];lines.append(f"{group} NR64/SOLE PPL ratio {v['ppl_ratio']:.6f},95% CI {v['ratio_ci95']}. " + ('Improvement supported on this panel.' if v['ratio_ci95'][1]<1 else 'Interval includes1; PPL improvement is not statistically established on this panel.'))
lines.append(f"R3 NR64 remains{100*(sw['vs_F']['R3_wide_nr64']['ppl_ratio']-1):.2f}% above matched F16. All A8 arms fail unchanged quality point gates. NR64 versus exact PPL differences have intervals including1; a lower point estimate is not evidence that reciprocal approximation improves mathematical accuracy.")
lines+=['','Paired document NLL/PPL ratios,95%intervals and language/domain cells: SOFTWARE_REPORT.md. Candidate fixed before any final scoring; no final-data selection. Existing overall5%/eachcell10% quality gates remain unchanged.','', '## Same-input hardware error amplification','', '|Normalizer|Probability max LSB|AV max LSB|O max LSB|Residual max LSB|Final max LSB|Final cosine|','|---|---:|---:|---:|---:|---:|---:|']
for mode,rows in aud['r3_amplification'].items():
 b=rows[0]['boundaries'];lines.append('|'+mode+'|'+'|'.join(str(b[n]['max_lsb']) for n in ['probability','AV','O','residual','final'])+f"|{b['final']['cosine']:.8f}|")
lines+=['','These are actual HMX dense R3 versus independently verified Float64-dense reference, on identical layer input and each declared normalizer. Dense matrix1FP16ULP+minnormal and Q/K1code gates pass, but whole2LSB/cosine0.999 does not. Exact normalization also fails; further reciprocal refinement alone cannot remove this captured residual divergence.','', 'Each of12 arm x9step raw-QK/probability/AV boundary sets independently matches NumPy int64 exactly. Same actual Q/K under SOLE/exact/NR64 checked; O/residual/final are actual DSP captures. NR64 vs exact prefill probability difference<=1LSB, final up to3LSB; that is an approximation diagnostic, not its implementation oracle. HVX NR64 vs independent NR64 arithmetic and untimed scalar complete-layer reference are exact. Repeated NR64 output and independently packed K cache/verified V cache pass. Old EXP0251 controls reproduce.','', '## Hardware speed','', 'Five short and ten rotated paired formal rounds, repeat1/repeat10,2970timed layerRPCs. Exact division and numerically ineligible R3 arms are audit-only. Primary repeat10 median paired-round mean Host wall, paired ratios/10000bootstrap seed252. No outlier or warmup deletion; repeat1 remains separately reported.','', '|Scope|SOLE us|NR64 us|Paired change|Ratio95% CI|','|---|---:|---:|---:|---|']
for key,v in hw['performance'].items():lines.append(f"|{key}|{v['control_us']:.3f}|{v['nr64_us']:.3f}|{v['paired_regression_percent']:+.2f}%|{v['paired_ratio_ci95']}|")
lines+=['','Primary repeat10 intervals are below1.1; speed gate passes, no extra pairs required. Both include1, so no demonstrated speedup/penalty. Repeat1 prefill uncertainty is wider and does not replace the declared repeat10 primary.8MiB requested/acquired,peak6,682,752bytes,zero timed intermediate DDR/spill/audit,oneRPC and7nativeW4projections; every additive ledger closes exactly. Constant reciprocal table256bytes, per-row LUT construction and Newton arithmetic included.','', '## Limits and continuation','', 'Current runtime ABI116, layer0 M64+eightM1, capacity72. No full-model device PPL/text/E2E or baseline promotion. Conditional further integration is not entered while R3 numerical failure and model-quality acceptance remain unresolved. No costly mode5 refinement, butterfly, altered weights/activation scales or hidden scalar fallback in timing.','', 'All execution attempts succeeded. PPL model processes loaded before later hardware/reporting-only commits; core PPL source files are byte-identical since a6a8326. Per-score heads retained and verified. Exact probability division is a reference, not an assumed efficient device candidate. Source/evidence checks in independent_integrity_checks.json, provenance in ARTIFACT_PROVENANCE.json.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'FULL_PROFILE.md').write_text((R/'HARDWARE_PROFILE.md').read_text()+'\n'+(R/'SOFTWARE_REPORT.md').read_text()+'\n'+(R/'REPORT.md').read_text())
(R/'NEXT_DIRECTION.md').write_text("# After EXP0252\n\nKeep NR64 as an experimentally validated low-overhead normalization candidate, no promotion. Its same-input probability closely matches exact normalization, and exact normalization still leaves R3 whole-layer mismatch. Discuss the remaining score rounding/exponent threshold sensitivity with identical captured Q/K, then consider an explicitly bounded finer exponent representation while retaining native W4 and U8 AV output. Do not spend another iteration improving reciprocal alone toward exactness as a cure for R3 amplification. Full-model device acceptance remains downstream of numerical/physical/speed eligibility, a consecutive-layer slice and longer-cache coverage; independent PPL remains primary. Any next method requires separately frozen protocol, no calibration/final-data tuning or hidden mixed precision.\n")
print('FINAL_INTEGRITY_PASS',summary['source_head'])
