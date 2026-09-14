# L32-0017：FP32 残差向量化和流水恢复

用户明确速度优先，授权先修流水和向量实现。本轮已完成4个实现阶段：
HVX O/Down epilogue、四上下文 FP32 prenorm、固定16行转置和decode
向量广播、FP16 embedding 原地向量展开。原0012-SP2 control保留。
实测源码59348e01bb10fc38c7974550345377b5449ac9c1，两个分支中仅no-rotation变更。

## 实现和合同
原生packed W4 HMX保留signed24分量，向量重构S32（SP2 low+256high-32768sum），
转FP32、乘每通道scale并加FP32残差。Down prefill复用两个VTCM原始结果槽：
HMX发布后HVX消费，最终FP32 store和barrier后才确认可复用；每个HMX命令
在bias DMA槽重用前等待对应消费结束。Decode保留同次矩阵乘承载low/high。
O和decode Down在HMX worker中使用转交的HVX上下文，完成后归还。

两处prefillNorm由main+3个HVX worker分担16行；转置在已闲置normalized
VTCM缓冲内进行，不增加峰值。按token并行保留原逐通道FP32加法顺序，
没有树形重关联或FMA。Decode用vdelta广播避免HVX->scalar->HVX逐通道往返。
FP16 gamma转FP32和post-Norm A8 native格式向量生产；量化舍入临界带
回到原标量除法，保持原U8码合同。Embedding从行尾向前向量展开防止别名覆盖。
原权重/scales/SP2、attention/KV/head保持冻结；无旋转、重新校准或PPL。

## 算术与物理验证
Layer0/7/15 prefill64+decode1、连续3层、完整16层输出逐值精确，KV精确。
最终完整生成tokenID/U8logit码与封存FP32 oracle一致，原0012control也一致。
36次硬件CLI：35成功、1失败被保留；436个token边界，其中正式320个全部通过。
prefill/decode的Host/DSP可加和台账全部闭合。
FP32 VTCM8098272B，control8229344B；8MiB限制保持，中间DDR/spill=0，
W4展开=0，FP16 HMX投影回退=0，R3/R4=0。没有改变浮点recipe。

失败：第一版O零点只广播到标量乘法操作数一个半字，导致奇数lane缺补偿。
SDK模拟定位后广播两个半字，后续所有代表层和完整生成精确。保留失败二进制/
输出。其余编译、目录、parent manifest问题均在硬件前停止并修复；
manifest实际与本地目标完全相同，仅历史parent字段阻止同包复用。
运行器现允许已验证的目标或声明parent，逐文件hash仍检查，未接受未知hash。

## 正式性能
同一构建，固定十轮AB/BA，M64+15连续decode，bootstrap20000次、seed17017。
时间包括embedding、16层、finalnorm、head、greedy、FastRPC，排除加载和外部tokenizer。

| 阶段 | 0012-SP2 token/s | FP32优化后 token/s | Host耗时增加 | 配对95% ratio CI | 10%gate |
|---|---:|---:|---:|---|---|
| Prefill |1996.44|1639.74|21.75%|[1.204287,1.227944]|未通过|
| Decode |45.73|42.26|8.21%|[1.075607,1.088402]|通过|

未优化L32-0016历史单次功能速度51.5686/25.2278tok/s，仅辅助参照，非配对正式比较。
当前不能整体性能验收，不能覆盖0012的最优速度结论，也没有质量或默认baseline晋升。

## 剩余瓶颈
完整prefill两处Norm 10.221ms，对照2.011ms，增加8.210ms；O增加1.208ms。
Down仅增加0.0925ms（约3.0%），decode Down增加0.0037ms（约0.14%）。
Embedding由原型约5ms降至62us，对照44us。
Gate/Up本次记录减少2.479ms；不同数值轨迹/缓冲布局下的观测，不是独立核优化结论。
净prefill增加6.974ms，decode每token增加1.795ms，后者主要也是Norm。
下一方向仍以速度为主：FP32 Norm重排/归约及O出口；不要自动恢复PPL校准或旋转。
若改变归约顺序，应明确新实际算术并独立验证，不静默放宽现有精确检查。

完整模块表：/mnt/d/llm_exp/results/llama32-htp/l32-0017/PROFILE.md
正式原始结果：formal-a01/；构建：build-59348e01-16/；summary.json。
Ledger SHA256 ae1572aca736152bbf59a9a5e445b979a1549d3ce25d880a78bfcef051c19dcf，
223个证据文件及7个复用模型manifest。旧失败、旧速度结论均保留。
