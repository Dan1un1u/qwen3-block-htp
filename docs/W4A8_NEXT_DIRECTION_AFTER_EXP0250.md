# EXP0250 后续方向

The proposed repair is effective on this independent panel: R3 full integer attention PPL 8272.14 -> 41.87, OFF 11271.88 -> 61.56. The second score saturation is the dominant measured failure. R3 with frozen Q/K recalibration improves the repaired path relative to OFF. However repaired R3 is still 42.0% above F16 PPL; the exact-normalization reference is 32.1% above F16. Both fail the unchanged accuracy reference. SOLE adds a significant conditional penalty in both groups, while integer-exponent pointwise intervals include zero loss. The repair succeeds numerically; A8 quality acceptance remains unmet.

本轮只实现软件参考：原始HMX U8分数先做masked row maximum，以signed16差值乘multiplier，再进入整数指数/SOLE，去掉第二次U8饱和。其他recipe、C64 per-channel W4、校准参数与原始前缀均冻结。
已完成16个冻结配置的新128文档/2048目标PPL，8个旧对照逐token精确复现，独立整数参考、因果/重复/CE、权重及前缀不可变检查通过。数值门槛通过只表示实现/诊断有效，不能视为A8质量验收或设备部署完成。
主要PPL结果与置信区间见 EXP-0250-RESULTS.md。下一步建议先核实原生U8输出加HVX宽差值的设备映射和开销，并把整数指数/SOLE的精度改进作为冻结的小型对照。精确除法当前只作诊断参考，尚未证明设备低成本实现。保留原生HMX W4、8MiB VTCM、无中间DDR、一RPC、单层prefill/decode任一稳定慢超过10%停下讨论；不直接跳全模设备。
仍须解决软件与实际HMX dense R3算术的差异，不能复用EXP0248昂贵refinement后宣称速度可接受。若attention差值修复落地，再检查K/V/AV载体与剩余非线性误差。
Active none,next251待讨论；新实验尚未授权。无设备运行/新权重/参数调优/基线提升；设备构建缓存仍为EXP0248实验layer0 ABI114。E2E token/s N/A。

{
  "F": 29.48685047467392,
  "C64": 29.665686253733227,
  "OFF_carrier": 49.9431430300986,
  "OFF_unbounded": 51.267662972385274,
  "OFF_wide_float": 50.98174560213134,
  "OFF_score": 7658.5484388235445,
  "OFF_wide_exact": 52.3883729685423,
  "OFF_wide_sole": 61.55943464749742,
  "OFF_sole": 11271.881494705096,
  "R3_carrier": 39.089125959289866,
  "R3_unbounded": 39.027082388015444,
  "R3_wide_float": 39.027082388015444,
  "R3_score": 6371.563253147737,
  "R3_wide_exact": 38.96114843607058,
  "R3_wide_sole": 41.87363262103216,
  "R3_sole": 8272.14274430138
}
