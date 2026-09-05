# 三种 recipe 统一轻量评测方案（提案，尚未执行）

F16F16、W4F16、W4U8 使用同一原始 checkpoint、tokenizer、输入 token IDs、模板、上下文范围及 greedy 规则。W4F16/W4U8 的 W4 权重和尺度必须逐文件一致；各 recipe 的实际 LM-head 数值路径也包含在质量评测中。W4U8 质量允许失败，但必须报告失败，不通过改数学或筛样本掩盖。软件数学复现、可读文本和模型质量分别记录，不能互相替代。

## 1. 固定速度基准

近期统一 M64 prefill + 15 次连续 decode（共返回 16 个 token），batch=1、greedy、关闭 thinking；三种 recipe 使用同一提示和相同次数的 KV 更新。三种运行顺序轮转，一次预热、五轮短测，正式结果十轮。每轮重建会话，decode 内持续反馈实际生成 token，不重放冻结缓存。一次 session 内保留权重和资源。

Prefill = 64 / 完整 token-in/token-out Host wall；decode = 15 / 完整连续 decode Host wall，不计 prefill。同时归档每次调用与完整循环的 Host 时间，单独记录 tokenizer、模型加载、prepare、首 token 文本可见延迟和 detokenizer，避免把热态推理称为包含冷启动的用户请求耗时。正常 profiling 不开启 logits/hidden 审计。未来支持更长 A16 会话后再另立 M64+64 标准，不能与旧的 15/192-step 数字混算。

对用户只展示固定三种 recipe 模块表（微秒、Host 占比、W4U8 相对 W4F16 增速），末尾附各 recipe 实测 E2E token/s；详细计数器、完整 ledger 和原始数据留档。历史数据必须注明非配对，统一速度排名只使用同一轮转测试的新数据。

## 2. 质量主指标：短上下文条件 NLL

冻结中英文各半的小语料清单，每段 64 context tokens + 16 continuation tokens。正式轻量集 32 段，共 512 个计分 token；每次迭代快速集固定取其中 8 段、每段前 8 个目标 token，共 64 个计分 token。另留 8 段未用于调参的验收集，recipe 晋升前执行。语料来源、版本、许可、文本和 token IDs 的 SHA256 一次固化；不根据 candidate 表现换题或调阈值。

使用 teacher forcing：所有 recipe 接收相同真值前缀，KV 仍由自己的 transformer 计算。统计平均 NLL、exp(mean NLL)、相对原始 BF16 teacher 的 ΔNLL、相对本 recipe 冻结基准的 ΔNLL，并按中文/英文分别列出。这里的 PPL 是明确限定 64→80 上下文范围的小样本条件 PPL，不能当作完整 WikiText/C4 成绩。每个文档边界重新开始，目标 token 只计一次。

BF16 teacher 在主机上一次运行并缓存结果；设备的 F16F16、W4F16、W4U8 才是待比较的实现。仅检查主机重新量化模型无法发现实际 DSP 算术差异，不能替代设备分数。额外保留 teacher top-1 一致率、目标 token 排名和 U8 logits 饱和/并列比例作为诊断。

## 3. 可解释的小任务集

固定 24 个短题：8 个短阅读/信息抽取、8 个单步常识或算术、8 个格式/指令遵循，中英文各半。只要求短答案，每题最多 16 output tokens，用固定规则做 exact-match、数值或 JSON 字段检查；报告原始通过数/总数和具体回退题目。快速集固定 8 题。另留 4 个中文/英文开放提示展示输出，由人抽查，不用在线 LLM judge，也不将主观可读性当作准确率。

8/24 题的结果粒度很粗，不能宣称通用 benchmark 能力。原始 BF16 和两个 A16 recipe 首次共同跑完后才能判断这套题的区分度；题集版本冻结后，任何改题产生新版本，旧分数保留。W4U8 失败照实列出，不阻塞另两个 recipe 的评测。

## 4. 成本与判定

每轮普通实验先跑快速集；正式候选及 baseline 建立时跑 512-target-token + 24-task 轻量集。质量测量无须重复十次，只额外重复固定两个样本验证确定性；速度测量单独重复。每个 recipe 一次加载权重，循环运行全部样本，以减少 4 GB 级 F16F16 包装/部署/映射开销。目标是常规快测数分钟内、完整轻量集约十分钟内完成全部 recipe；这是工程预算，首次实测后登记实际时间，不作现成性能承诺。

先建立 BF16 teacher、F16F16、Selected W4F16、当前 W4U8 四列初始分数，不先假定两个 A16 已经达标。调度/布局等声称数学不变的实验继续要求既有实现一致性；真实数值变化的实验用固定样本逐项 ΔNLL 与任务回退评审，先标红再判断，不能为了通过临时放宽阈值。建议初始告警线为相对本 recipe 基准 ΔNLL > 0.02 nats/token 或任务少通过 >=2/24；这些只是待首次测量校准的告警提案，不是已批准的硬门槛。

## 5. 需要补的接口

本次 EXP-0217 仅补齐固定 M64+15 的 F16F16 生成。上述评测尚需一个独立注册的实现步骤：三 recipe 共用样本 manifest、teacher-forcing token 输入、目标 token log-prob 输出和多样本会话 runner。LM head 可在已有分块扫描中用数值稳定的 streaming log-sum-exp 与目标 logit 累积，只返回小标量；质量模式独立计时、不污染正式速度。W4U8 必须按真实 output qparam 解码其量化 logits，不能直接对 U8 编码求 softmax，也不能改用主机 FP16 head 偷换 recipe。短任务若模板不是恰好 64 tokens，还需显式支持 logical M≤64、位置和 mask；不能静默插入有意义的 padding。该接口属于下一步提案，当前未声称已实现或已评测。

参考：
- Qwen 官方模板与 non-thinking 模式：https://github.com/QwenLM/Qwen3
- Hugging Face 对固定上下文 PPL、tokenization 与上下文影响的说明：https://huggingface.co/docs/transformers/perplexity
- EleutherAI 对 loglikelihood 与 generate_until 的接口划分：https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md
