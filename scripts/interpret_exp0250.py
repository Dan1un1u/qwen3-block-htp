from pathlib import Path
import json,math,hashlib
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0250');s=json.loads((R/'summary.json').read_text());local=json.loads((R/'local_wide_precision.json').read_text())
finding=(f"The proposed repair is effective on this independent panel: R3 full integer attention PPL {s['final_ppl']['R3_sole']:.2f} -> {s['final_ppl']['R3_wide_sole']:.2f}, OFF {s['final_ppl']['OFF_sole']:.2f} -> {s['final_ppl']['OFF_wide_sole']:.2f}. The second score saturation is the dominant measured failure. R3 with frozen Q/K recalibration improves the repaired path relative to OFF. However repaired R3 is still {100*(s['vs_F']['R3_wide_sole']['ppl_ratio']-1):.1f}% above F16 PPL; the exact-normalization reference is {100*(s['vs_F']['R3_wide_exact']['ppl_ratio']-1):.1f}% above F16. Both fail the unchanged accuracy reference. SOLE adds a significant conditional penalty in both groups, while integer-exponent pointwise intervals include zero loss. The repair succeeds numerically; A8 quality acceptance remains unmet.")
lines=['','## Interpretation','',finding,'',
 'Masked row maximum is now computed on raw native U8 scores. The difference is explicitly signed16 before multiplication and exponent coding, avoiding the second U8 saturation. The bound255*18+4=4594 fits int16. This changes downstream score handling only; native W4 linear semantics and first HMX U8 score output are retained.',
 'A simple same-range example: raw scores[160,180], multiplier5. The old second conversion maps both to255, making their difference zero. The repaired path retains a difference100 in F3 units before exponent coding. Moving row-max subtraction later cannot reconstruct that difference once both codes have saturated.',
 'Full-model software results must be interpreted against the paired controls below. They do not demonstrate a new device kernel or recover precision already lost in the first raw U8 conversion. The latter is isolated separately by unbounded versus wide_float.',
 '', '| Complete integer path | Old PPL | Repaired PPL | Repaired / old (95% CI) |', '|---|---:|---:|---:|']
for arm in ['OFF','R3']:
 d=s['ladder'][arm]['sole->wide_sole'];ci=d['ratio_ci95']
 lines.append(f"| {arm} | {s['final_ppl'][arm+'_sole']:.6f} | {s['final_ppl'][arm+'_wide_sole']:.6f} | {d['ppl_ratio']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] |")
lines+=['', 'First-stage raw-U8 versus unbounded per-token NLL/top1 exactness on the final panel: '+json.dumps(s['first_raw_U8_outputs_exact'])+'. Equality on this finite panel does not prove saturation is impossible on other inputs or longer contexts.']
lines+=['', 'Eight exposed development controls exactly reproduce the parent experiment. All16 independent arms were specified before inference, so every arm is reported. No range or candidate tuning used the final panel. Better conditional PPL after rounding or exponent encoding can reflect interacting errors, and is not proof that those approximations are error-free.',
 '', '## Same-input Softmax precision on retained exposed development traces', '', '| Group | Standard Softmax + U8: mean row L1 / mass | Integer exponent + exact division | Integer exponent + SOLE |', '|---|---:|---:|---:|']
for arm,layers in local['variants'].items():
 parts=[]
 for name in ['wide_float','wide_exact','wide_sole']:
  l1=sum(v['methods'][name]['mean_row_L1'] for v in layers)/28;mass=sum(v['methods'][name]['row_mass_mean'] for v in layers)/28;parts.append(f'{l1:.6f} / {mass:.6f}')
 lines.append('| '+arm+' | '+' | '.join(parts)+' |')
lines+=['', 'Each entry is an equal-layer average over all28 retained EXP0249 carrier-trajectory prefill traces for four exposed development documents. Within each group the source scores and masks are identical. L1 is relative to standard Softmax of the repaired, first-stage-U8 scores. Codes are divided by255 for this local diagnostic; the slightly clipped frozen AV probability scales remain unchanged in model PPL. These traces are explanatory and are not additional independent final samples.',
 '', '## Next direction and limits', '',
 'Prioritize checking a device realization of raw U8 masked maximum and widened HVX score differences while preserving native HMX W4 and the existing memory/RPC contract. Keep the existing single-layer Host-wall >10% slowdown stop. Before full-model deployment, re-evaluate exponent and reciprocal precision with a small fixed candidate set on development inputs; exact division here is a diagnostic upper reference for normalization, not a demonstrated fast DSP implementation. Do not equate software dense R3 with the expensive EXP0248 device refinement or silently inherit its timing.',
 'The quality reference remains <=5% overall PPL increase and <=10% in every cell relative to F16, with pointwise uncertainty reported. A successful numerical repair does not mean A8 quality acceptance. Remaining carrier, V/AV and nonlinear errors are still bundled outside these conditional comparisons. Other recipes and weights remain frozen. No new experiment, hardware deployment or baseline promotion is implied by this proposal.',
 '', 'Artifacts: raw_score_repair.png/.pdf, summary.json, local_wide_precision.json, independent_integrity_checks.json, FULL_PROFILE.md. E2E token/s N/A.']
with (R/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')
(R/'FULL_PROFILE.md').write_text((R/'profile_frozen_template.md').read_text())
next_lines=['# EXP0250 后续方向','',finding,'',
 '本轮只实现软件参考：原始HMX U8分数先做masked row maximum，以signed16差值乘multiplier，再进入整数指数/SOLE，去掉第二次U8饱和。其他recipe、C64 per-channel W4、校准参数与原始前缀均冻结。',
 '已完成16个冻结配置的新128文档/2048目标PPL，8个旧对照逐token精确复现，独立整数参考、因果/重复/CE、权重及前缀不可变检查通过。数值门槛通过只表示实现/诊断有效，不能视为A8质量验收或设备部署完成。',
 '主要PPL结果与置信区间见 EXP-0250-RESULTS.md。下一步建议先核实原生U8输出加HVX宽差值的设备映射和开销，并把整数指数/SOLE的精度改进作为冻结的小型对照。精确除法当前只作诊断参考，尚未证明设备低成本实现。保留原生HMX W4、8MiB VTCM、无中间DDR、一RPC、单层prefill/decode任一稳定慢超过10%停下讨论；不直接跳全模设备。',
 '仍须解决软件与实际HMX dense R3算术的差异，不能复用EXP0248昂贵refinement后宣称速度可接受。若attention差值修复落地，再检查K/V/AV载体与剩余非线性误差。',
 'Active none,next251待讨论；新实验尚未授权。无设备运行/新权重/参数调优/基线提升；设备构建缓存仍为EXP0248实验layer0 ABI114。E2E token/s N/A。','',json.dumps(s['final_ppl'],indent=2)]
(R/'NEXT_DIRECTION.md').write_text('\n'.join(next_lines)+'\n')
print('REPORT_AUGMENTED')
