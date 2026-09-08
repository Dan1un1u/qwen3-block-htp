#!/usr/bin/env python3
"""Independent collection verification, provenance and closure documents; run after report."""
from pathlib import Path
import json,hashlib,subprocess,statistics
import numpy as np
S=Path('/home/daniuniu/work/qwen3-block-htp');M=Path(str(S)+'-project-memory');R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0251')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=S,text=True).strip()
z=json.loads((R/'summary.json').read_text());a=json.loads((R/'numerical_audit.json').read_text());pr=json.loads((R/'formal_provenance.json').read_text())
assert not git('status','--porcelain')
assert not git('diff',pr['source_head'],'HEAD','--','src','include','CMakeLists.txt')
assert z['source_head']==git('rev-parse','HEAD')
assert a['eligible']=={'control':True,'wide':True,'r3_wide':False}
assert all(x['exact'] for arm in a['integer_boundaries'].values() for x in arm)
assert all(x['whole']['changed']==0 and x['repeat_exact'] for x in a['wide_whole'])
from exp0240_report import FIELDS
calls=0;files=0;allrs=[];rng=np.random.default_rng(251)
def records(tag):return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if l.startswith('{') and json.loads(l).get('record')=='exp0240_profile']
for phase,n in [('short',5),('formal',10)]:
 for i in range(n):
  for rep in [1,10]:
   for cell in ['control','wide']:
    rs=records(f'{phase}_{i:02d}_r{rep}_{cell}');gold=records('audit_'+cell);assert len(rs)==9*rep
    for j,x in enumerate(rs):
     assert x['output_hash']==gold[j%9]['output_hash']
     assert x['wide_score_mode']==int(cell=='wide') and x['dsp_status']==3
     assert x['u8_attention_audit_ddr_write_bytes']==0
     assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608
     assert x['vtcm_peak_plan_bytes']==6682752
     assert x['block_invocation_count']==1 and x['w4u8_decode_direct_n_projection_count']==7
     assert x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==x['intermediate_spill_fill_count']==0
     assert x['scan_total_kv_length']==64+j%9
     assert sum(x[k] for k in FIELDS)==x['invocation_ticks']==x['ledger_named_ticks'] and x['ledger_unattributed_ticks']==0
    allrs.extend(rs);calls+=len(rs);files+=1
assert calls==2970 and files==60
for rep in [1,10]:
 for mode in ['prefill','decode']:
  values={}
  for cell in ['control','wide']:
   values[cell]=[statistics.mean(x['host_wall_ns']/1000 for x in records(f'formal_{i:02d}_r{rep}_{cell}') if (x['replay_step']==0)==(mode=='prefill')) for i in range(10)]
  ratios=np.array(values['wide'])/values['control'];v=z['performance'][f'repeat{rep}_{mode}']
  assert abs(statistics.median(values['control'])-v['control_us'])<1e-9
  assert abs(statistics.median(values['wide'])-v['wide_us'])<1e-9
  assert abs((statistics.median(ratios)-1)*100-v['paired_regression_percent'])<1e-9
  ci=np.quantile(np.median(ratios[rng.integers(0,10,(10000,10))],axis=1),[.025,.975]);assert np.max(abs(ci-v['paired_ratio_ci95']))<1e-12
assert z['speed_gate']=='pass'
for mode,arms in z['module_centers'].items():
 for cell,m in arms.items():assert abs(sum(v for k,v in m['modules_us'].items() if k!='Complete Host wall')-m['host_us'])<1e-8
parent_pins={'exp0247':'8752c15d95d9bb5dced802cda7d7d64cb92897b1c6c16fc9a81f909f80ec0965','exp0248':'9df15e39189ee4f9904c765321370b5262abdf46cde70e1f12ae39cd6eb9fc1e','exp0250':'e82e72cec87543319561493abd975cf828bcdea1b2ee46b18c57fcb53e206b0a'}
parents={}
for parent,pin in parent_pins.items():
 root=R.parent/parent;ledger=root/'EVIDENCE_SHA256.json';assert sha(ledger)==pin
 data=json.loads(ledger.read_text())
 for name,item in data['files'].items():assert sha(root/name)==item['sha256'],name
 parents[parent]=dict(ledger_sha256=pin,files_verified=len(data['files']))
for name,pin in pr['binaries'].items():
 assert sha(R/'binaries/attempt1'/name)==pin
 live=S/('hexagon_ReleaseG_toolv19_v79' if 'skel' in name else 'android_ReleaseG_aarch64')/'ship'/name;assert sha(live)==pin
import device_exp0251 as device
remote=device.adb('shell',f'cd {device.REMOTE} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
assert all(pin in remote for pin in pr['binaries'].values())
assert device.adb('shell','cat /proc/sys/kernel/random/boot_id')==pr['boot']
models={}
for cell,pin in [('control','8f42f9e07f49d90504f8845e9f01ffc796c133c567794d8bf15af8778f93eaff'),('r3','c196670085429fb749cd2c391fa3d4c79eee9c3644a222cce3def002475609f1')]:
 root=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0247')/cell;p=root/'manifest.json';assert sha(p)==pin;models[cell]=dict(manifest_sha256=pin,manifest=json.loads(p.read_text()))
checks=dict(pass_all=True,all_timed_RPCs=calls,all_timed_processes=files,all_outputs_match_audited_reference=True,all_physical_and_additive_ledgers_exact=True,independent_pair_statistics_and_intervals_exact=True,all_parent_ledgers_and_files_unchanged=parents,native_source_unchanged_since_runtime_build=True,local_and_remote_binary_hashes_and_boot_verified=True,scope='M64 then eight M1 steps; layer0 only; no model-quality acceptance')
(R/'independent_integrity_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
provenance=dict(experiment='EXP-0251',protocol_sha256=sha(M/'docs/experiments/EXP-0251.md'),source_head=git('rev-parse','HEAD'),runtime=pr,models=models,parent_evidence=parents,toolchain='SDK6.6 Tools19.0.07 HTPv79 NDKr26c',ABI=115,analysis_python='/home/daniuniu/.cache/qwen3-block-htp-analysis-py/bin/python',GPU_oracle_python='/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python')
(R/'ARTIFACT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
report=['# EXP-0251 hardware speed of wide-score repair','','Only the no-R3 wide-score repair is numerically eligible and passes the10% speed gate. Plain dense HMX R3 + wide repair still fails the independent whole-layer gate and has no formal timing result.','', 'Device PJZ110 / SM8750 / HTP V79. Frozen EXP0247 C64 per-output-channel W4 packages (C64 is prefix length, not weight group size), native HMX weight.n. Five short + ten rotated paired formal rounds, each repeat1 and repeat10:2970 timed layer RPCs. M64 prefill then8 real M1 cache appends, one layer0. No discarded rounds.','', '|Scope|Old us|Wide repair us|Paired time change|95% change interval|','|---|---:|---:|---:|---|']
for key,v in z['performance'].items():report.append(f"|{key}|{v['control_us']:.3f}|{v['wide_us']:.3f}|{v['paired_regression_percent']:+.2f}%|[{(v['paired_ratio_ci95'][0]-1)*100:+.2f}%, {(v['paired_ratio_ci95'][1]-1)*100:+.2f}%]|")
report+=['','Host wall cells are separate medians of paired-round means; paired changes use the median of within-round ratios, so dividing displayed median times gives a different percentage. Primary repeat10 confidence intervals are entirely below+10% and contain zero; no demonstrated speed penalty or speedup. No extra rounds required.','', 'Numerical audit: all45 arm/step raw-QK, probability and AV boundary sets match the independent NumPy integer oracle exactly, including cached decode. W0 all9 outputs exactly match untimed scalar-wide reference and repeated capture; C0 reproduces sealed EXP0248 control. K cache independently packed and V cache matched to the oracle. Formal output hashes match these audited references. Wide exponent range/mask/future-max/shift-invariance proof retained from sealed EXP0250.','', 'R3 limit: dense matmul itself passes1FP16ULP+min-normal and Q/K differences<=1code, but prefill whole-layer max11LSB/cosine0.9927222 and several decode steps3-7LSB/cosine0.98547-0.99056 fail the unchanged2LSB/cosine0.999 gate. Exact downstream arithmetic from actual captured Q/K does not imply equality to Float64 rotation. No mode5 refinement or scalar fallback was timed.','', 'Every timed invocation requests/acquires8MiB VTCM, peak6,682,752bytes; zero intermediate DDR/spill and audit exports;7 native W4 projections and1RPC. Additive ledger exactly closes. Complete repeat1/repeat10 additive module, overlapping engine/DMA/wait and physical counters in FULL_PROFILE.md; stable recipe tables in USER_PROFILE.md.','', 'Recovery: first build hit a Host telemetry printf type mismatch for the new uint32 mode; corrected U32 emission without arithmetic changes and rebuilt successfully. Initial collector used system Python missing NumPy; no device call occurred in that attempt, reran using existing analysis environment. Both failed logs retained. No data deletion, threshold relaxation, quantizer or weight change.','', 'Deployment scope: compiled layer0 ABI115; wide mode capacity<=72, testedM64+eightM1 only. Do not treat this experimental cache as a full-model runtime. Other recipes frozen, no baseline promotion. Device PPL and full-model E2E token/s:N/A. Software EXP0250 PPL is not a device measurement.','', 'Source '+git('rev-parse','HEAD')+'; runtime '+pr['source_head']+'.']
(R/'REPORT.md').write_text('\n'.join(report)+'\n')
(R/'NEXT_DIRECTION.md').write_text("# After EXP0251\n\nWide-score repair maps correctly to the tested native-U8 hardware path with no demonstrated Host-wall penalty and passes the10% single-layer speed gate. Keep it as an experimental implementation; model-quality acceptance is unresolved.\n\nDiscuss next scope: validate longer cached contexts/all layers and device PPL before promotion; separately examine whether actual dense-HMX rounding plus current log2 exponent/SOLE quantization amplifies one-code Q/K changes. EXP0250 software residual precision ablations are the starting evidence. Any higher-precision exponent/reciprocal or R3 compensation needs a new explicit experiment and frozen comparisons. Do not silently reintroduce expensive mode5 refinement, butterfly/FWHT, grouped weights or mixed precision. No new algorithm or full-model experiment is authorized by this closure.\n")
print(json.dumps(checks,indent=2))
