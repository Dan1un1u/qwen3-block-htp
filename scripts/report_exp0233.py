#!/usr/bin/env python3
"""Verify and close official-tool software experiment, preserving failed attempts."""
import json,math,subprocess,hashlib
from pathlib import Path
import numpy as np
from data_exp0233 import RESULT,OUTPUT,SOURCE,MEMORY,write,verified,preflight
from autoround_exp0233 import frozen,sha,read_weight

def main():
 preflight();frozen();assert not (RESULT/'closure.json').exists()
 primary=json.loads((RESULT/'summary_primary.json').read_text());name='summary_combined.json' if primary['reserve_trigger'] else 'summary_primary.json'
 final=json.loads((RESULT/name).read_text());assert not final['reserve_trigger'] and final['independent_raw_token_reduction_count']==36
 for n,h in final['inputs'].items():assert sha(RESULT/n)==h
 dev={v:json.loads((RESULT/f'software/development_{v}.json').read_text()) for v in ['F','C64','AR-P','AR-G']}
 for v,r in dev.items():
  assert r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6
  if v in ['F','C64']:assert r['exact_prior_development_regression']
  assert abs(math.exp(math.fsum(x for s in r['samples'] for x in s['nll'])/2048)-r['ppl'])<1e-10
 assert json.loads((RESULT/'quantization_oracle.json').read_text())['pass_all']
 assert json.loads((RESULT/'full_model_calibration_audit.json').read_text())['pass_all']
 packages={}
 for v in ['AR-P','AR-G']:
  rec=json.loads((RESULT/v/'package.json').read_text());root=Path(rec['root']);assert sha(root/'manifest.json')==rec['manifest_sha256'];m=json.loads((root/'manifest.json').read_text())
  assert len(m['projections'])==196 and len(rec['blocks'])==28 and all(r['finite'] and r['calibration_samples']==512 for r in rec['blocks'])
  for n,e in m['files'].items():assert sha(root/n)==e['sha256'],n
  for n,e in m['projections'].items():
   layer,short=n.split('/');a=read_weight(root/layer,short,tuple(e['shape']));assert hashlib.sha256(a.tobytes()).hexdigest()==e['effective_FP16_sha256']
  packages[v]=rec
 commands=[]
 for p in sorted((RESULT/'commands').glob('*.log')):
  m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['log_sha256']
  assert sha(OUTPUT/'artifacts'/m['source_head']/'source.tar')==m['source_archive_sha256'];commands.append(m)
 assert len(commands)==len(list((RESULT/'commands').glob('*.json')))
 artifacts={str(p.relative_to(OUTPUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
 write('artifacts_sha256.json',dict(root=str(OUTPUT),files=artifacts,all392projection_roundtrips=True))
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
 text=['# EXP0233 independent AutoRound reference','',f"Software final: {final['status']}; {final['documents']} documents / {final['targets']} targets. Reserve used: {primary['reserve_trigger']}. No promotion.",'',
  '|Stratum|F16|C64|AR-P|AR-G|AR-P/F16 [95%CI]|AR-G/F16 [95%CI]|','|---|---:|---:|---:|---:|---|---|']
 for n,g in final['statistics'].items():
  vals=g['ppl'];a=g['vs_F16']['AR-P'];b=g['vs_F16']['AR-G']
  text.append(f"|{n}|{vals['F']:.6f}|{vals['C64']:.6f}|{vals['AR-P']:.6f}|{vals['AR-G']:.6f}|{a['ppl_ratio']:.6f} {a['ratio_ci95']}|{b['ppl_ratio']:.6f} {b['ratio_ci95']}|")
 text+=['','## Matched-control interpretation','',f"Overall versus C64: {final['statistics']['overall']['vs_C64']}",'',
  'AR-P is official AutoRound blockwise learned rounding with a project-format[-7,7] quantization adapter, oneFP32scale peroutputrow; AR-G uses unchanged official int_sym full-range[-8,7], signed FP32scales andgroup128. Both use the exact existingC64 512x128calibration positions,200iterations/block,batch8,seed233,official best training-MSE parameter choice. C64 uses GPTQ and originalFP32solver weights; AutoRound AMP rounds originalweights toFP16 first and uses iterative optimization. AR-P isolates format compatibility, not solver alone; AR-G is a multifactored native-tool reference. Both freeze originalgamma/embedding/norms and the inheritedEXP221head. This is not the best unconstrained full-tool recipe or a causal estimate for grouping alone.',
  '', 'Official intel/auto-round v0.5.1 commit73669aa50871bca9f244ad99faa9979790f7c729 was archived with Apache2license; every installed source module matches the pinned checkout. Isolated dependency installation reused frozen torch2.8.0 andtransformers4.51.0 paths without changing old environments. All effective arguments/source archives are retained. Setup dependency and JSONdtype serialization failures remain in commands/recovery, without training-score tuning.',
  '', '## Correctness and data','',
  'Independent NumPy tests cover project/native grids, zero/one-sided rows,FP16/FP32 and finite nonzero STE gradients. All392projection nibble/scalereconstructions exactly match exported effectiveFP16weight hashes;56blocks contain512finite teacher/student calibration outputs, first4stored. Frozen nontransformer parameters/buffers match before/after training and when reloaded for scoring. Canonical evaluation repeat/causal/CE checks pass; F/C64development reproduces all prior pertokenNLL/top1 exactly. Independent complete-model calibration forwards match all84 stored teacher/student layer states exactly(maxNRMSE0), and the two tool branches share exactFP16teacher states. Every36 final PPLaggregate independently reduced from rawtoken NLL.',
  '', 'Development is the reusedEXP230128document panel, descriptive only. Final1024documents were frozen before training/scoring and independently reconstructed, excluding every prior calibration/training/evaluation role throughEXP232 using document/text/32grams. Same tokenizer64prompt+16targets, masks, eagerFP16backend for all. Gates overall5% and all language/domain/cell10% use paired stratified documentbootstrap5000seed233. No candidate selected from finalPPL; reserve rule fixed. The unused inherited dataset scale_search_reference metadata is explicitly corrected in recovery/dataset_metadata_erratum.json without changing dataset/freeze hashes; actual calibration is512x128 for both candidates. The same PC052panel may be reused by the next two diagnostics with explicit disclosure, never to select a deployed model.',
  '', 'Development PPL: '+str({v:r['ppl'] for v,r in dev.items()}),
  '', '## Execution boundary and continuation','',
  'Host-only quality experiment; no DSP code/build/device/profiling changes. DequantizedPyTorch speed is not W4runtime throughput. Otherrecipes frozen. Next authorized phaseEXP234 group12864K, thenEXP235 C64sensitivity. No automaticpromotion or newoptimization direction.']
 with (RESULT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
 ref=verified('exp0230','full_profiling_report.md');table=verified('exp0230','module_table.md').read_text()
 profile=f"""# EXP0233 complete profiling record: software-only

Closure source {head}; artifacts{OUTPUT}; evidence{RESULT}. F/C64/AR-P/AR-G software teacher forcing64+16; no target-device execution.

|Required section|Repeat1 control/candidate/delta|Repeat10 control/candidate/delta|Reason|
|---|---|---|---|
|Complete Hostwall,prefill,continuousdecode|N/A|N/A|No DSPexecution|
|DSPinvocation,setup,teardown,additiveblockstages,unattributed|N/A|N/A|No timestamps|
|ProjectionDMA,HMX,unpack,waits,workers,lifetimes|N/A|N/A|No newruntime|
|QK,Softmax,AV,packing,waits,tasks|N/A|N/A|No newruntime|
|GateUp,SwiGLU,Down,slots,publication|N/A|N/A|No newruntime|
|DDRbytes,DMA,VTCM,FastRPC,HMX,spill|N/A|N/A|No hardwareclaim|
|Devicehashes,LSB,physicalgates,5short10formal|N/A|N/A|Softwarequalityonly|

Missing values are not zeros; overlapping counters are not additive. Independent software quantizer/packing/repeat/mask/CE checks and full provenance are in REPORT.md.

Historical EXP230 fullprofiling reference {ref}, SHA256{sha(ref)}. The table below is unchanged historical evidence: W4F16 C64candidate(notpromoted), F16F16/W4U8 nonpairedEXP218. DifferentW4bytes prohibit activation-only attribution. It is not an AutoRound profile.

{table}

AR-P/AR-G E2E prefill/decode token/s:N/A. HistoricalC64:64tokens/63262.995us=1011.649859tok/s;15tokens/1389448.9595us=10.795647tok/s. No softwaretime extrapolation.
"""
 with (RESULT/'full_profiling_report.md').open('x') as f:f.write(profile)
 closure=dict(experiment='EXP-0233',execution_state='completed',evidence_validity='valid',local_gate='pass' if all(v=='pass' for v in final['status'].values()) else 'fail',
  adoption_status='pending',source_head=head,quality=final,development={v:r['ppl'] for v,r in dev.items()},packages=packages,
  dataset_sha256=sha(RESULT/'dataset.json'),data_freeze_sha256=sha(RESULT/'dataset_freeze.json'),report_sha256=sha(RESULT/'REPORT.md'),full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),
  artifact_files=len(artifacts),artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),baseline_promoted=False,other_recipes='frozen',next_direction='approved_EXP234_then_EXP235')
 write('closure.json',closure)
 write('evidence_sha256.json',{str(p.relative_to(RESULT)):sha(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
 print('EXP0233_CLOSED',final['status'],flush=True)
if __name__=='__main__':main()
