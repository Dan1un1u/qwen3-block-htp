# Qwen3 HTP — native-W4 SP2 流水迁移

当前分支 `codex/exp-0267-sp2-native-w4-pipeline` 保存 EXP-0267 的 Qwen3 实现与结果。方法来自已封存的 Llama L32-0012；历史 Qwen3 分支及两条 Llama 分支均保持原提交。

[方法结论](docs/NATIVE_W4_SP2_METHOD.md) · [实验结果](docs/experiments/EXP-0267-RESULTS.md) · [完整 profiling](docs/experiments/EXP-0267-PROFILE.md)

Qwen3-1.7B 全28层、M64+15连续decode、KV容量128，五轮short及十轮正式配对测试。repeat10为主，repeat1仅参考。优化SP2相对U8：prefill墙钟−15.59%，decode+1.23%，均通过10%门槛；相对上一版SP2重叠流水，prefill−5.24%。实测prefill2020.27token/s、decode47.88token/s。原U8为1705.31及48.47token/s。

SP2通过低位/高位整数分解复用原生packed W4乘法。decode使用空闲物理行承载两个分量；prefill通过Gate/Up与SwiGLU、Down HMX与HVX的并行覆盖额外工作。Qwen版本复用MLP阶段已空闲的HMX激活缓冲，四种实现峰值均为8,365,824B，无层间张量DDR搬运。本轮验证量化合同实现与性能，没有评估PPL或提升默认精度状态。

源码实测版本及四个二进制哈希见结果报告与 `D:/llm_exp/results/qwen3-block-htp/exp0267`。原始权重不变，新SP2元数据位于 `D:/llm_exp/models/qwen3-block-htp/exp0267`。

任何后续工作先运行权威project-memory bootstrap，并按输出顺序读取四个authority文件：

```bash
/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/bootstrap.sh /home/daniuniu/work/qwen3-block-htp
```

实验已结束。新修改或真机运行需先登记新的已批准实验并通过preflight。Qwen专用构建入口为 `scripts/build_qwen3_sp2.sh 28`；EXP-0267脚本和结果目录是不可覆盖的复现证据。`tools/recipe.py`仍只检查历史配方，不是SP2启动器。

[Llama父分支概览](docs/LLAMA32_PARENT_OVERVIEW.md)仅保留历史上下文；后续Llama工作使用其独立worktree与project-memory。
