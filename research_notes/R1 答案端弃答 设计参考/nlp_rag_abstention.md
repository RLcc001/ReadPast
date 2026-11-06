# NLP / RAG 中“有证据 vs 无证据”对比的免训练弃答与答案核验规则（R1 设计参考）

> 阅读层级标注：**[全文]** = 通过 arXiv HTML / ar5iv 读了正文（部分长文只读到前约 100k 字符，标为 **[全文·前段]**）；**[摘要]** = 只读了摘要页或检索摘要；**[检索摘要]** = 只见到搜索结果片段，未打开正文。所有数字均来自所列来源，未自行推算的数字不写。
> 检索日期：2026-11-03。

---

## 问题 1：上下文 vs 参数知识的对比（CAD、AdaCAD、COIECD、CDA、PMI 类打分、冲突检测、“忠实度”分数）——精确公式与 α/阈值设定

### Takeaway
这一族方法的核心量都是 **log p(y | c, x) − log p(y | x)**（PMI / CXMI / 对比 logit），与 R1 的 log p'_E(o*) − log p'_Q(o*) 同构；但几乎没有工作把它的**符号（阈值 0）**直接当成硬判决来弃答。原版 CAD 用固定 α（0.5 或 1，按任务类型手设），后续 AdaCAD / COIECD / CPMI 的主要动机恰恰是“固定强度的对比在**无冲突**样本上明显掉点”（CAD 在无冲突 NQ-Synth 上 88.20→64.00），于是改为按 JSD、熵或信息量约束**自适应门控**——这与 R1 在可答且答对样本上 16–20% 误弃答是同一个病。

### Cited Findings

#### 总表（贯穿五个问题的密集目录）

| 方法 | 对比形式（with vs without） | 判决规则 / 阈值 | 阈值怎么定 | 是否评估“过度弃答/伤及无冲突样本” | 包装方式 | 来源 |
|---|---|---|---|---|---|---|
| CAD (Shi et al. 2023) | (1+α)·logit(y|c,x) − α·logit(y|x) | 无硬阈值，软重加权 | α 固定：摘要 0.5，知识冲突 1 | 报告了 FLAN-T5 在无冲突 NQ 上下降 | “simple inference-time method”，免训练 | [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739) |
| AdaCAD (2024) | α_t = JSD(p(·|x) ‖ p(·|c,x)) | 无阈值；长文本 warmup 下限 λ=0.3 | λ=0.3 取自无冲突 QA 上 JSD 的最大值 | 是，专门在无冲突 NQ-Synth 上评估 | CAD 的自适应修正 | [AdaCAD](https://arxiv.org/html/2409.07394) |
| COIECD (2024) | 用 I(y_t)=−log p(y_t|x,c) 与无上下文熵 H₁ 的差构造约束集 | 约束集外视为“冲突 token” | λ=0.25、α=1，NQ 上扫参后固定 | 是，无冲突子集上与 Regular 比较 | 冲突检测 + 自适应解码 | [COIECD](https://arxiv.org/html/2402.11893) |
| CDA (2024/25) | 参数 prompt、上下文 prompt、null prompt、弃答 prompt 四路分布 | 软混合，弃答权重 w^a = 1 − w^p − w^c；无硬阈值 | 动量 α=0.7；校准用 null prompt | 是：F1_ans / F1_abs 分开报告 | “首个带弃答的上下文对比解码”，免训练 | [CDA](https://arxiv.org/html/2412.12527) |
| CPMI (van der Poel 2022) | log p(y|y<t,x) − λ·𝟙{H≥τ}·log p(y|y<t) | 熵门控 τ 决定何时启用 PMI | 在 500 条带标注 XSum 子集（属测试集）上调参 | 是，比较全程 PMI 与门控 PMI 的 ROUGE 代价 | 解码策略 | [CPMI](https://ar5iv.labs.arxiv.org/html/2210.13210) |
| PMI_DC (Holtzman 2021) | argmax_i P(y_i|x,dom)/P(y_i|dom) | 选项打分，无阈值 | 无（仅模板） | 报告 3 个数据集不占优 | 打分规则 | [PMI_DC](https://ar5iv.labs.arxiv.org/html/2104.08315) |
| ClashEval TPC (2024) | 有上下文答案 vs 无上下文答案的平均 token 概率 | 先验概率更高就用无上下文答案；无阈值 | 校准版改比较“百分位” | 报告 Prior Bias（误用先验的比例）上升 | 基准 + 简单免训练方法 | [ClashEval](https://arxiv.org/html/2404.10198) |
| Du et al. 2024 persuasion score | KL(p(A|c,q) ‖ p(A|q)) | 分析指标，非判决 | — | — | 分析/度量 | [Du et al.](https://arxiv.org/html/2404.04633) |
| FilCo CXMI (2023) | M(o|t⊕q) / M(o|q) | 比值 > 阈值才保留上下文片段 | 正文写 λ=0.0，脚注说 1.0 最好（论文自相矛盾） | — | 训练数据过滤（需要金标答案） | [FilCo](https://arxiv.org/html/2311.08377) |
| LUMINA (ICLR 2026) | MMD(p(·|q,d), p(·|q,d′))，d′=随机文档 | H_t = λ·I − (1−λ)·E，连续分数 | λ=0.5 默认（扫参似在评测集上） | 只报 AUROC/AUPRC；阈值 F1 用 oracle 阈值 | 免训练检测器 + 统计检验 | [LUMINA](https://arxiv.org/html/2509.21875v3) |
| ReDeEP (ICLR 2025) | ECS（注意力头对上下文利用）与 PKS（FFN 前后 logit-lens JSD） | 回归加权 α·PKS − β·ECS | 在验证集上选超参/回归系数 | AUC 等检测指标 | 机制可解释性检测器 | [ReDeEP](https://arxiv.org/html/2410.11414) |
| Sufficient Context (ICLR 2025) | 用 autorater 判 context 是否充分（不是概率对比） | logistic 回归组合“充分性 + 自评置信”，低于阈值弃答 | 按目标 coverage 取阈值 | 是：coverage–selective accuracy 曲线 | “新视角”分析 + 选择性生成 | [Sufficient Context](https://arxiv.org/html/2411.06037) |
| Counterfactual prompting (EMNLP-F 2024) | 原 RAG 答案 vs 反事实 prompt 下重答（**不是**有/无上下文） | 答案不变就保留，变了就丢弃；无阈值 | — | 是：Risk / Carefulness / Coverage | RAG 风险控制框架 + 基准 | [CF prompting](https://arxiv.org/html/2409.16146) |
| Feng et al. Compete (ACL 2024) | 加入其他 LLM 生成的冲突证据后重答 | 多数情况下答案改变就弃答；无阈值 | 无需 held-out | 是：R-Acc / ER / A-Acc / A-F1 | 多 LLM 协作 | [Feng et al.](https://arxiv.org/html/2402.00367) |
| BERT SQuAD 2.0 | s_null 与最优非空 span 分数 | ŝ_ij > s_null + τ 才作答 | τ 在 dev 集上调到 F1 最大 | EM/F1 混合 | 标准做法 | [BERT](https://ar5iv.labs.arxiv.org/html/1810.04805) |
| Conformal abstention (2024) | 自一致性匹配计数（不是有/无上下文） | 分数 < λ̂ 弃答 | 共形风险控制，有分布无关风险上界 | 是：风险 vs 弃答率 | 带统计保证的阈值选择 | [Conformal](https://arxiv.org/html/2405.01563) |

#### CAD（Context-Aware Decoding，“Trusting Your Evidence”，Shi et al. 2023）[全文]
- 公式：y_t ∼ softmax[(1+α)·logit_θ(y_t | c, x, y<t) − α·logit_θ(y_t | x, y<t)]；论文把它解释为 PMI 调整：上下文条件概率乘以（有上下文 / 无上下文）概率比的 α 次方，“outputs that become much more likely when the context is included are preferred”；α=0 退化为普通解码，并类比 classifier-free guidance — [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739)
- α 的选择：摘要任务 α=0.5，知识冲突任务（MemoTrap、NQ-Swap、NQ）α=1；原文说 “α=0.5 generally yielded good results across all settings…but a slightly higher α is more effective in the knowledge conflict setting”；正文未描述 dev 集调参或自适应 α — [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739)
- 伤害无冲突样本：原文称除了 “a minor decrease observed for FLAN-T5 on the non-conflict NQ dataset” 外全面提升；FLAN-T5 11B 在 NQ 上 85.5→82.5，3B 81.8→80.3，作者归因于 NQ 在 FLAN 的指令微调数据中 — [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739)
- 收益示例：LLaMA-30B CNN-DM ROUGE-L 25.8→31.8、factKB 76.8→87.8；GPT-Neo 20B MemoTrap 37.1→57.3、NQ-Swap 16.1→36.8（注：引言称 LLaMA-30B 提升 2.9 倍，但表 3 的 9.6→37.7 约 3.9 倍，论文未解释） — [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739)
- 包装：可用于任何现成 LM、“without any additional training”；需要有/无上下文两套 logits（两次前向是推断，论文未明说） — [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739)

#### AdaCAD（2024）[全文]
- α^jsd_t = JSD(p_θ(y_t|x,y<t) ‖ p_θ(y_t|c,x,y<t))，采样 y_t ∼ p(y|c,x,y<t)·[p(y|c,x,y<t)/p(y|x,y<t)]^{α^jsd_t}；JSD ∈ [0,1]；长文本加 warmup 下限 α = max(α^jsd, λ)，λ=0.3，“matches the maximum JSD on non-conflicting QA data” — [AdaCAD](https://arxiv.org/html/2409.07394)
- 关键动机数据（Llama3-70B，无冲突 NQ-Synth）：greedy 88.20，CAD 64.00（约 24% 相对下降），AdaCAD 86.40；平均 α_max 在 NQ-Swap（高冲突）为 0.45、NQ-Synth 为 0.28 — [AdaCAD](https://arxiv.org/html/2409.07394)
- Llama3-70B 上 greedy / CAD / AdaCAD：NQ 44.13/34.05/45.43；TriviaQA 90.20/54.60/88.80；PopQA 86.10/75.16/85.68（AdaCAD 在部分无冲突集上仍略低于 greedy） — [AdaCAD](https://arxiv.org/html/2409.07394)
- 跨模型和 QA 数据集平均：比 CAD 高 14.21 分、COIECD 4.82、ConfCD 5.86、greedy 2.41；把 CAD 的 α 在 500 条验证集上调优（NQ 0.25、HotpotQA 0.50）后，AdaCAD 仍平均高约 3.4 分 — [AdaCAD](https://arxiv.org/html/2409.07394)

#### COIECD（2024）[全文·前段]
- 信息量 I(y_t) = −log p(y_t|x,c,y<t)，与无上下文熵 H₁ 的偏移 I − H₁；p_δ = softmax(I − H₁)；上界 u = λ·max p_δ，下界 l = (1/λ)·min p_δ；约束集 C 外的 token 视为冲突 — [COIECD](https://arxiv.org/html/2402.11893)
- 自适应解码：g = log p₂ − log p₁（p₂ 有上下文，p₁ 无上下文）；非冲突 token 用 log p₁ + α·g，冲突 token 用 log p₂ + α·g — [COIECD](https://arxiv.org/html/2402.11893)
- 超参：λ=0.25、α=1，在 NQ 上扫参后称 “consistently provide robust improvements”；理论中的常数 γ 从未赋值 — [COIECD](https://arxiv.org/html/2402.11893)
- CAD 在无冲突数据上掉点（LLaMA2-13B）：SQuAD EM 80.50→78.53，NQ EM 73.05→67.50；FLAN-T5-3B SQuAD EM 79.56→68.37；COIECD 本身在无冲突 NQ 上也略低于 Regular（72.37 vs 73.05 EM） — [COIECD](https://arxiv.org/html/2402.11893)
- 理论依托：Stable Entropy Hypothesis、Locally Typical Set；作者承认计算量约翻倍 — [COIECD](https://arxiv.org/html/2402.11893)

#### CDA：Contrastive Decoding with Abstention（“When to Speak, When to Abstain”）[全文·前段]
- 用输出分布熵 H 度量不确定性；**用 null prompt（“[QUESTION]”“[CONTEXT]” 占位符）计算基线熵 H̄^p、H̄^c 做校准**，得相关性信号 r^p、r^c（抓取工具给出的式子为 max(H − H̄, 0)/H̄，符号方向需对照原文核实），再归一化成权重 w^p、w^c — [CDA](https://arxiv.org/html/2412.12527)
- 输出分布：d^o = w^p·d^p + w^c·d^c + (1 − w^p − w^c)·d^a，其中 d^a 来自显式弃答指令；两路知识都不确定时弃答权重变大；**解码时没有硬阈值**；动量版 w_t ← 0.7·w_{t−1} + 0.3·w_t — [CDA](https://arxiv.org/html/2412.12527)
- 唯一阈值 η=0.7（10 次采样中至少 8 次正确）只用于构建测试床，判断“参数知识是否存在” — [CDA](https://arxiv.org/html/2412.12527)
- 消融：去掉 null-prompt 校准后 NQ 的 RS 从 62.95 跌到 47.89（CDA，Llama3）——**校准是关键** — [CDA](https://arxiv.org/html/2412.12527)
- 成本约为 greedy 的两倍（多出上下文、null、弃答 prompt 的前向） — [CDA](https://arxiv.org/html/2412.12527)

#### CPMI（Mutual Information Alleviates Hallucinations in Abstractive Summarization）[全文]
- score = log p(y|y<t,x) − λ·𝟙{H(p(·|y<t,x)) ≥ τ}·log p(y|y<t)：**只在模型不确定（熵 ≥ τ）时才启用 PMI**；τ=0 即全程 PMI — [CPMI](https://ar5iv.labs.arxiv.org/html/2210.13210)
- 选定值：tranS2S λ≈0.1312、τ≈3.5618；bartS2S λ≈0.6560、τ≈3.5987；在 Maynez et al. 的 500 条标注 XSum 子集上搜索（该子集属测试集，结果可能偏乐观） — [CPMI](https://ar5iv.labs.arxiv.org/html/2210.13210)
- 代价对比：全程 PMI 使 ROUGE-L 降约 3.13%，门控 CPMI 只降约 0.977% — [CPMI](https://ar5iv.labs.arxiv.org/html/2210.13210)

#### PMI_DC（Surface Form Competition，Holtzman et al. 2021）[全文]
- exp PMI_DC = P(y|x,domain)/P(y|domain)，选项选择 argmax_i P(y_i|x,dom)/P(y_i|dom)；domain premise 是条件前缀的结尾，如 CommonsenseQA 用 “the answer is:”；动机是 surface form competition — [PMI_DC](https://ar5iv.labs.arxiv.org/html/2104.08315)
- 无阈值；16 个划分、13 个数据集，在 GPT-3 175B 上领先次优方法 40 个百分点以上；HellaSwag、ARC Easy、BoolQ 上不占优 — [PMI_DC](https://ar5iv.labs.arxiv.org/html/2104.08315)
- 相关：Calibrate Before Use 用 content-free 输入（如 “N/A”）拟合校准参数，使该输入的预测在各答案上均匀，GPT-3/GPT-2 平均准确率最多提升 30.0 个绝对点 [摘要] — [Zhao et al. 2021](https://arxiv.org/abs/2102.09690)

#### 冲突与忠实度分数
- Du et al.（ACL 2024）persuasion score：ψ(c,q) = KL(p(A|c,q) ‖ p(A|q))（文中称 half-pointwise MI）；susceptibility = I(C; A | q)；答案分布用下一 token 分布近似，每个关系 600 个 context 做蒙特卡洛估计；熟悉实体的 susceptibility 显著更低（Pythia-6.9b：open 查询 73/122，closed 查询 61/122） [全文·前段] — [Du et al.](https://arxiv.org/html/2404.04633)
- ClashEval：GPT-4o 在检索内容错误时 “over 60% of the time” 推翻自己正确的先验；先验越自信越不采纳上下文，斜率 −0.1 到 −0.45 [全文] — [ClashEval](https://arxiv.org/html/2404.10198)
- ReDeEP：ℋ = Σ α·PKS − Σ β·ECS，α、β 是回归系数，超参在 RAGTruth 验证集选；LLaMA2-7B RAGTruth AUC 0.7458（RAGAS 0.7290）；LLaMA2-13B AUC 0.8244 [全文·前段] — [ReDeEP](https://arxiv.org/html/2410.11414)
- LUMINA：外部上下文分数 = 以检索文档 d 与**随机文档 d′**（实验中用另一条数据的检索文档）为条件的下一 token 分布之间的平方 MMD（cosine 核，top-100 token）；论文没有解释为何不用空上下文；H_t = λ·I − (1−λ)·E，λ=0.5；免训练，两次前向；RAGTruth AUROC：Llama2-13B 0.8569 vs ReDeEP 0.8055，Llama3-8B 0.7446 vs 0.7495（略低） [全文] — [LUMINA](https://arxiv.org/html/2509.21875v3)

### Inferences
- R1 的判据 Δ = log p'_E(o*) − log p'_Q(o*) > 0 就是 PMI_DC / CXMI / CAD 对比项在选中选项上的**符号检验**。FilCo 脚注把比值阈值 1.0（对数域的 0）描述为 “naturally distinguishes context that adds to or reduces output probability”，是“阈值 0”最直接的先例，但 FilCo 用的是金标答案、用于训练数据过滤，不是推理时弃答。
- CAD → AdaCAD / COIECD / CPMI 这条演化线提供了与 R1 误弃答问题直接对应的修正思路：**只在两路分布确实有分歧（JSD 大）或模型不确定（熵高）时启用对比**。AdaCAD 的 λ 下限“取无冲突 QA 上的最大 JSD”，给出了一个无需标签、可从无冲突数据统计得到的门控尺度。
- CDA 和 Calibrate Before Use 说明，比较两个条件（有证据 / 只有问题）之前，先用 content-free / null prompt 对两边**各自校准**很重要（CDA 去掉校准后 RS 降约 15 分）。R1 直接比较 p'_E 与 p'_Q 的原始 log 概率，两边的置信度尺度不同，这是误弃答的一个可能来源。ClashEval 也明说有上下文概率 “extremely right-tailed”、先验 “nearly uniform”。
- LUMINA 用随机文档而非空上下文作对照，可以避免 prompt 长度和格式的混杂。对应到 R1，可以考虑用“随机或打乱的记忆 + 帧”作对照，而不是“只给问题”（推断，文献未直接比较这两种对照）。

### Gaps
- 原计划中的 “Adaptive Contrastive Decoding in RAG for Handling Noisy Contexts”（ACD，Kim et al. 2024）未核实：我抓取的 arXiv 2406.17519 实际是 CLeHe（“Entropy-Based Decoding for Retrieval-Augmented LLMs”），因此 ACD 的公式未写入。CLeHe 的公式：y_t ∼ softmax[(1+β)Σ_j w_j log p(y_t|d_j,x) − β log p^{l*}(y_t|x)]，w 为按文档熵的 softmax（τ=0.1/0.25），β 在 WebQ 验证子集上选 — [CLeHe](https://arxiv.org/html/2406.17519)
- CDA 的 r^p、r^c 公式符号方向（H − H̄ 还是 H̄ − H）需对照 PDF 原式；其 RS(α) 定义按抓取结果为 “α×Coverage + (1−α)×Accuracy，α 为作答率”，看起来不寻常，未独立核实。
- 未找到专门把 CAD 式 log-ratio 的**符号**直接用作 MCQ 弃答硬判决、并测量过度弃答的论文；这一组合似乎是空白（与 R1 新颖性相关，但“没找到”不等于不存在）。

---

## 问题 2：RAG 与 QA 中的弃答（Sufficient Context、Self-RAG、CRAG、综述、AbstentionBench、R-Tuning、Feng et al.、Kamath、P(True)/P(IK)、语义熵、SQuAD 2.0 no-answer 阈值）——哪些用了有/无证据对比？阈值怎么定？

### Takeaway
主流 RAG 弃答方法几乎都**不**用“有证据 vs 无证据”的概率对比做判决：要么用训练出的评估器 / 反思 token（Self-RAG、CRAG）或 LLM autorater（Sufficient Context），要么用置信度 / 一致性分数加阈值。阈值几乎都靠 **dev / held-out 集调参**（BERT 的 τ、Feng et al. 的 calibration 基线、Kamath、Whitehead）、**目标 coverage**（Sufficient Context）、**共形风险控制**（Yadkori），或**经验手设**（CRAG、Self-RAG 默认 0.2）。真正免阈值的只有“答案是否改变”这类离散一致性规则（Feng Compete、Counterfactual prompting、DBA、ClashEval TPC）。“有 vs 无检索”在这些论文里主要出现在**分析**中（如 Sufficient Context 的 closed-book vs RAG 弃答率），没有被用作判决。

### Cited Findings
#### Sufficient Context（Joren et al., ICLR 2025）[全文]
- 定义：存在答案 A′ 使其在 C 下是 Q 的合理答案时，context 即 sufficient；autorater 为 Gemini 1.5 Pro 1-shot，在 115 条人工标注上准确率 0.930、F1 0.935（FLAMe 0.878/0.892，TRUE-NLI 0.826/0.818） — [Sufficient Context](https://arxiv.org/html/2411.06037)
- 选择性生成：信号 = 自评置信（开源模型用 P(True)：采样 20 个回答、询问 5 次；闭源模型用 P(Correct)）+ FLAMe 的二值充分性标签（1600 token 切块，任一块充分即为充分）；组合器为 “simple logistic regression model to predict hallucinations”（100 次随机超参搜索）；分数低于阈值就弃答，阈值按目标 coverage 选 — [Sufficient Context](https://arxiv.org/html/2411.06037)
- 收益表述有不一致：摘要说回答时正确率提升 “2–10%”，引言说 “among total model responses”；Gemma 27B 在 HotpotQA 高准确区间提升 10% 以上，Gemini 1.5 Pro 在约 70% coverage 处提升 5% 以上 — [Sufficient Context](https://arxiv.org/html/2411.06037)
- **关键反例**：context 不充分时，模型仍有 35–62% 的情况答对（部分来自闭卷知识）——“证据没有提供支持”不等于“答错” — [Sufficient Context](https://arxiv.org/html/2411.06037)
- 有/无 RAG 的弃答率（分析而非判决）：Claude 3.5 Sonnet 无 RAG 84.1% → 有 RAG 52%；GPT-4o 34.4%→31.2%；Gemini 1.5 Pro 100%→18.6%。即加了检索后模型更少弃答、更多幻觉 — [Sufficient Context](https://arxiv.org/html/2411.06037)
- 人工子集上 context 不充分时（C/A/H%）：Gemini 1.5 Pro 9.6/50.0/40.4；GPT-4o 23.1/61.5/15.4；Claude 3.5 Sonnet 9.6/53.8/36.5 — [Sufficient Context](https://arxiv.org/html/2411.06037)

#### Self-RAG [全文·前段]
- 反思 token：Retrieve{yes,no,continue}、IsRel、IsSup{fully, partially, no support}、IsUse{1–5}；Retrieve=Yes 的归一化概率超过阈值就触发检索，大多数任务阈值 0.2，ALCE 为 0 — [Self-RAG](https://arxiv.org/html/2310.11511)
- 段落分数 f = p(y_t|x,d,y<t) + S(Critique)，S = Σ w^G·s^G，默认权重 IsRel 1.0、IsSup 1.0、IsUse 0.5，“can be adjusted at inference time”；critic 用 GPT-4 生成的标签蒸馏到 Llama 2-7B（需训练）；读到的部分没有显式弃答机制 — [Self-RAG](https://arxiv.org/html/2310.11511)

#### CRAG [全文·前段]
- 检索评估器：T5-large（0.77B）微调，正例为 PopQA 金标 wiki 标题，负例从检索结果随机采样；输出分数在 [−1, 1] — [CRAG](https://arxiv.org/html/2401.15884)
- 阈值 “set empirically”：PopQA (0.59, −0.99)，PubHealth 与 Arc-Challenge (0.5, −0.91)，Biography (0.95, −0.91)；至少一篇文档高于上阈值为 Correct，全部低于下阈值为 Incorrect，其余为 Ambiguous；自称 “plug-and-play” — [CRAG](https://arxiv.org/html/2401.15884)

#### “Don't Hallucinate, Abstain”（Feng et al., ACL 2024）[全文·前段]
- 基线分族：calibration（token 概率阈值 p*、温度缩放、verbalized 置信，都在 held-out 集 H 上定阈值）、training（hidden-layer probe、外部 verifier、指令微调）、prompting（Self-Reflect、MoreInfo、Gen+Match）、consistency（NOTA、SC 阈值，也在 H 上调）。原文：“a majority and often the stronger approaches would require a held-out set” — [Feng et al.](https://arxiv.org/html/2402.00367)
- 提出的方法：Cooperate（专家 LLM 给反馈，judge 汇总）；**Compete**（其他 LLM 生成带支撑段落的替代答案，前置后重答，多数情况下答案改变就弃答，无需 held-out） — [Feng et al.](https://arxiv.org/html/2402.00367)
- 有/无检索（MMLU，Mistral-7B，WikiSearch）：Ask Cali. 的 R-Acc 0.648→0.515（检索有害）；Compete 0.735→0.720；abstain-retrieve-abstain 两步流程 “cuts back the error rate by up to 56.6%” — [Feng et al.](https://arxiv.org/html/2402.00367)
- 最强基线之上，abstain accuracy 最多提升 19.3%；12 个设置中 9 个最优；指令微调式弃答跨数据集或跨 LLM 迁移时，abstain accuracy 降幅 “sometimes as large as 33.8%” — [Feng et al.](https://arxiv.org/html/2402.00367)

#### SQuAD 2.0 风格的 no-answer 阈值（BERT）[全文·相关节]
- 无答案视为 [CLS] span；s_null = S·C + E·C；只有当最优非空 span 分数 ŝ_ij > s_null + τ 时才作答，“the threshold τ is selected on the dev set to maximize F1”；BERT-LARGE 测试集 80.0 EM / 83.1 F1 — [BERT](https://ar5iv.labs.arxiv.org/html/1810.04805)

#### Selective QA（Kamath, Jia, Liang 2020）[全文]
- 测试集为 SQuAD 与未知 OOD 数据 50/50 混合；calibrator 为随机森林，7 个特征（passage 长度、答案长度、top-5 softmax 概率），用 1600 条 SQuAD + 1600 条已知 OOD 训练、各 400 条验证；指标为 risk–coverage AUC（越低越好）与 Cov@Acc=80%/90% — [Kamath et al.](https://ar5iv.labs.arxiv.org/html/2006.09462)
- 数字：MaxProb AUC 20.54、Cov@80% 48.23；Calibrator（SQuAD + 已知 OOD）AUC 18.47、Cov@80% 56.06；加 dropout 特征 AUC 17.31、Cov@80% 59.99 — [Kamath et al.](https://ar5iv.labs.arxiv.org/html/2006.09462)

#### P(True)/P(IK)、语义熵、R-Tuning
- Kadavath et al.：P(True) 是模型评估自己答案为真的概率；P(IK) 是不针对具体答案的“我知道”概率；P(IK) “increase appropriately in the presence of relevant source materials in the context” [摘要] —— 这是“有证据时置信应上升”的经验先例，但不是判决规则 — [Kadavath et al.](https://arxiv.org/abs/2207.05221)
- 语义熵：“unsupervised, uses only a single model, and requires no modifications to off-the-shelf language models” [摘要]；不涉及有/无证据对比 — [Kuhn et al.](https://arxiv.org/abs/2302.09664)
- R-Tuning：基于 knowledge intersection 构造 refusal-aware 数据做指令微调（需训练）；称拒答是可泛化的 “meta-skill” [摘要] — [R-Tuning](https://arxiv.org/abs/2311.09677)

#### 综述与基准
- Know Your Limits（Wen et al.）：从 query / model knowledge / human values 三个视角，refusal 函数 r(x,y) 由连接器 M 组合；推理阶段方法分 input-processing、in-processing、output-processing；我读到的部分**没有**描述“有 vs 无检索上下文对比”式的弃答方法 [全文·前段] — [Wen et al.](https://arxiv.org/html/2407.18418)
- AbstentionBench：20 个数据集、20 个前沿 LLM、6 种场景；主指标为 abstention recall（作者说模型的 precision 普遍高）；judge 为 Llama 3.1 8B Instruct，人工标注上一致率 88%；reasoning 微调使弃答平均下降 24%；规模几乎无影响 [全文·前段] — [AbstentionBench](https://arxiv.org/html/2506.09038)

#### 共形弃答（Yadkori et al. 2024）[全文·前段]
- 分数为 LLM 自评的采样回答匹配计数（m.c.），或其期望版本（e.m.c.）；阈值用 conformal risk control 定，保证 E[R(λ̂)] ≤ α；RCPS 变体给出高概率界；δ=0.05；Temporal Sequences α=0.1、n=3200 时，m.c. 测试 loss 0.068、弃答率 0.107，对数概率版 0.095 / 0.165 — [Conformal abstention](https://arxiv.org/html/2405.01563)

#### 其他免训练或免阈值的一致性规则
- Counterfactual prompting：两条反事实 prompt（“Assume that your answer is wrong because the quality of your referred passages is poor” / “…due to your improper use of the retrieved passages”）下重答，答案不变就保留、变了就丢弃，冲突时让 LLM 裁决；**不比较有/无上下文**；方法本身无置信阈值（0.6 只是校准基线用） [全文] — [CF prompting](https://arxiv.org/html/2409.16146)
- DBA（2026）：Direct / Assistive / Incremental 三种 prompting 的答案不一致就弃答；“training-free…requires no retrieval or fine-tuning”；称作为错误检测器优于标准不确定性基线（F1、AUROC） [摘要] — [DBA](https://arxiv.org/abs/2602.04853)
- SABER（“Trust or Abstain? A Self-Aware RAG Approach”，2026-05 预印本）：在约 69K 实例上比较参数答案（PK）与上下文条件答案（CK），两个轻量预测器给出可靠度信念，做 trust PK / trust CK / trust either / abstain 四格决策；backbone 不微调，预测器是否训练摘要未说；在 risk–coverage 上 “Pareto-dominates every prompt-based abstainer” [摘要] — [SABER](https://arxiv.org/abs/2605.18792)

### Inferences
- 在本次检索中，最接近 R1“有 vs 无证据”判决的弃答工作是 **SABER**（PK vs CK 四格决策，需要轻量预测器）和 **CDA**（参数 / 上下文两路的不确定性都高时才弃答）。两者都把“参数知识已经足够”单列为**不弃答**的一格，而 R1 在“证据没比问题本身更支持 o*”时一律弃答，等于把“参数知识已经正确”这一格也判成弃答——这正是误弃答的结构性来源。
- Sufficient Context 的 35–62%“context 不充分却答对”从实证上说明：只看“证据是否额外支持答案”会系统性误伤靠先验答对的样本。对应到 R1：视频问题若问题与选项本身就强烈暗示答案（语言先验），p'_Q(o*) 已经很高，证据很难再提升它。
- 常规做法是在 dev 集或用 coverage 目标定阈值；R1 的“阈值 0、无标签”在 NLP 里没有直接先例，比较像 PMI_DC 那种“无阈值的打分规则”的思路。如果要保持无标签，可以借鉴 AdaCAD（用无冲突数据的 JSD 分布定尺度）或 CDA（用 null prompt 校准）的无标签尺度设定。

### Gaps
- Self-RAG 后半部分（约 170k 字符）未读，不能确认其评测中是否有弃答率或过度弃答统计。
- R-Tuning 只读了摘要，具体的数据划分方式（模型预测与标签比对）和 AP 指标未核实。
- Kadavath 的 P(IK) 随 context 变化只核实到摘要，没有具体数字。
- 未找到 Wen 综述中关于 RAG / 检索对比式弃答的条目（只读了前约 100k 字符）。

---

## 问题 3：比较“证据条件答案 vs 先验答案”的免训练答案核验（PVI、上下文信息增益、反事实 prompting、上下文消融一致性、检索消融、证据敏感性）

### Takeaway
真正以“有证据 vs 无证据的答案概率”做**推理时判决**、且免训练的最接近先例是 **ClashEval 的 Token Probability Correction**：比较有上下文答案与无上下文答案的平均 token 概率，谁高用谁，无阈值。它的作者自己发现原始概率比较有偏（有上下文的概率分布右偏、先验分布近乎均匀），所以改为比较**百分位**，这直接对应 R1 的问题。其余工作把 PMI / CXMI / PVI / DIG（信息增益）用作**训练数据过滤、重排序标签或分析指标**，都需要金标答案；ContextCite 用 context 消融做归因而不是有/无判决。PVI 原文明确指出负 PVI 不代表标签错误。

### Cited Findings
- **ClashEval Token Probability Correction** [全文]：“If the probability is higher for the prior than the contextual response, then we use the model's generation without context as its final response”；校准版因为有上下文概率 “extremely right-tailed”、先验 “nearly uniform”，改为 “compare the percentiles rather than raw probability scores”（百分位怎么算论文未写）— [ClashEval](https://arxiv.org/html/2404.10198)
  - 结果（Accuracy / Context Bias / Prior Bias）：GPT-4o 基线 0.615/0.304/0.021 → TPC 0.693/0.194/0.043 → 校准 TPC 0.754/0.107/0.085；GPT-3.5 0.539→0.596→0.701（Prior Bias 0.028→0.056→0.147）；Llama-3 0.500→0.556→0.649（Prior Bias 0.021→0.046→0.188）— [ClashEval](https://arxiv.org/html/2404.10198)
  - 随机对照：在相同 8.5% Prior Bias 下，随机用先验答案替换的准确率为 57.5%，校准方法为 75.4% — [ClashEval](https://arxiv.org/html/2404.10198)
  - 局限：数据集刻意富集了上下文错误，偏差率不代表真实分布；方法要求能拿到概率；文中一处写 “mean token probabilities”、图 4 说明写 “average log probability”，不一致 — [ClashEval](https://arxiv.org/html/2404.10198)
- **PVI**（Ethayarajh et al. ICML 2022）[全文]：pvi(x→y) = −log₂ g[∅](y) + log₂ g′[x](y)，g′ 用真实输入微调，g 用空输入 ∅ 在同样标签上微调（**需训练**）；“The higher the pvi, the easier the instance”；“A negative pvi simply means that the model is better off predicting the majority class than considering X”，负 PVI 的样本仍可能被预测正确 — [PVI](https://ar5iv.labs.arxiv.org/html/2110.08420)
- **FilCo CXMI** [全文]：f_cxmi(t,e) = M_gen(o|t⊕q) / M_gen(o|q)，选分数最高且超过阈值的 span；用的是金标输出 o，属于 oracle 标签，测试时训练一个过滤模型来预测；FEVER 上 Flan-T5 95.8（Lexical 86.4，StrInc 80.9），NQ / TQA 上 StrInc 更好 — [FilCo](https://arxiv.org/html/2311.08377)
- **InfoGain-RAG**（EMNLP 2025）[检索摘要]：Document Information Gain = 加入文档前后 LLM 对**金标答案**生成置信度的差；用 DIG 训练轻量重排序器（正/负增益分类 + 排序）；NQ 上 EM 比 naive RAG、self-reflective RAG、ranking-based RAG 分别高 17.9%、4.5%、12.5% — [InfoGain-RAG](https://arxiv.org/html/2509.12765v1)
- **PMI as RAG performance gauge**（NAACL 2025）[全文·前段]：PMI(q,c) = log p(q|c)/p(q)（问题在 context 下的似然，**不需要答案**）；只多一次 LM 编码；NQ-Open（10 篇文档）上 LLaMA-3-8B 最高 PMI 排序 54.04% vs 最低 49.30%；用于文档排序；读到的部分没有用于弃答；假设 2.1：PMI 与正确答案的 log-odds 呈仿射关系 — [PMI gauge](https://arxiv.org/html/2411.07773v2)
- **ContextCite**（context 消融）[全文·前段]：f(v) = p_LM(R | Ablate(C,v), Q)，对 logit σ⁻¹(f) 拟合 Lasso 线性代理（alpha=0.01），32–256 次消融；指标 Top-k-drop = log p(R|C,Q) − log p(R|Ablate(C,v_top-k),Q)；**没有**全 context vs 无 context 的判决；核验应用是把 top-k 源回喂模型、取 “yes” 概率，AUC 高于用全 context — [ContextCite](https://arxiv.org/html/2409.00729)
- **证据敏感性 / 鲁棒性规则**：Feng Compete（加入冲突证据后答案改变就弃答）— [Feng et al.](https://arxiv.org/html/2402.00367)；Counterfactual prompting（反事实 prompt 下答案改变就丢弃）— [CF prompting](https://arxiv.org/html/2409.16146)；LUMINA（检索文档 vs 随机文档的分布 MMD；H1 检验显示有文档的分数显著高于无文档）— [LUMINA](https://arxiv.org/html/2509.21875v3)
- **Persuasion score**：分布级 KL(p(A|c,q) ‖ p(A|q))，可视为 R1 点估计的分布级版本 — [Du et al.](https://arxiv.org/html/2404.04633)

### Inferences
- R1 可以精确改写为 **Bayes factor 检验**：log p(o*|E,Q) − log p(o*|Q) = log p(E|o*,Q) − log p(E|Q)，即“证据 E 在 o* 假设下是否比边际更可能”。R1 的判据是 BF > 1。PVI 原文提醒“负信息量 ≠ 错误”，Sufficient Context 的 35–62% 从实证上印证了这一点。所以 BF ≤ 1 只说明“证据没有提供额外信息”，不能证明“答案无依据”。
- ClashEval 的“原始比较 → 百分位比较”给了一个**无标签、无阈值**的改进方向：把 Δ = log p'_E − log p'_Q 换成两路各自在（无标签的）样本总体上的百分位后再比较，或对 Δ 本身做总体标准化，只在 Δ 落在下尾时弃答。代价是需要一批无标签样本估计分布。ClashEval 的百分位校准在 GPT-4o 上让 Prior Bias 从 0.043 升到 0.085，说明这个改动会把决策更多推向先验一侧，需在 R1 场景实测。
- 把 R1 从“o* 的点概率比较”改成“完整选项分布的 KL / JSD”（Du、AdaCAD）后，可以区分“证据没改变分布（先验已正确，应保留答案）”和“证据把质量从 o* 移到别处（应怀疑）”。R1 现在两种情况都可能判弃答。这是推断，文献中未见在 MCQ 弃答上直接比较这两种做法。
- 用随机记忆作对照（LUMINA 式）和用问题本身作对照（R1 / PMI_DC 式）衡量的是不同的东西：前者衡量“答案是否依赖于**这份**证据”，后者衡量“证据相对先验的增益”。

### Gaps
- InfoGain-RAG 只见到检索摘要，DIG 的精确公式（概率差还是对数差、是否长度归一化）未核实。
- 没有找到用 PVI / CXMI 在推理时**无金标**地做弃答判决的论文。
- “retrieval-ablation test”“context ablation consistency” 作为弃答判据的专门论文，本次没有找到；最接近的是 ContextCite（归因）和 LUMINA（检测）。

---

## 问题 4：弃答代价怎么报——coverage–risk、AURC、effective reliability、abstain-F1、可答/不可答分开的准确率、过度拒答率；主指标是什么？

### Takeaway
成熟做法都把**可答样本上的损失**单独列出：Abstain-QA 的 AAC / UAC、CDA 的 F1_ans / F1_abs、Trust-Align 的 F1_GR（拒答 F1 与作答 F1 的平均）、Wen 综述的 ARSP（安全或可答 prompt 上的弃答率，即过度保守）、Feng 和 Whitehead 的 Effective Reliability（错答计 −c，弃答计 0）、Kamath / Whitehead 的 risk–coverage AUC 与 Cov@Acc / C@R。只报“弃答准确率”或“总准确率”是被批评的做法。AbstentionBench 以 recall 为主，但同时报 precision。

### Cited Findings
- **Wen 综述的指标目录**（混淆矩阵 N1–N5：N1 不该弃答时答对，N2 不该弃答时答错，N3 不该弃答时弃答，N4 该弃答时作答，N5 该弃答时弃答）[全文·前段] — [Wen et al.](https://arxiv.org/html/2407.18418)
  - Abstention ACC = (N1+N5)/total；Precision = N5/(N3+N5)；Recall = N5/(N2+N4+N5)；Coverage = (N1+N2+N4)/total
  - **ARSP（过度保守）= N3/(N1+N2+N3)**；Benign answering rate = (N1+N2)/(N1+N2+N3)
  - R-Acc = N1/(N1+N2+N4)；ER = (N1 − N2 − N4)/total；另有 Coverage@Acc、AURCC、AUACC、AUROC、Abstain ECE
  - 开放问题：“Over-abstention occurs when models abstain unnecessarily”，需要 “balance abstention with utility”
- **Feng et al.**：R-Acc = A/(A+C)；ER = (A−C)/(A+B+C+D)；A-Acc = (A+D)/all；Abstain P = D/(B+D)，R = D/(C+D)；A-F1 被称为 “a balanced metric between reliability and answer coverage” — [Feng et al.](https://arxiv.org/html/2402.00367)
- **Effective Reliability Φ_c（Whitehead et al. ECCV 2022，VQA）**：作答且（部分）正确得 Acc(x)，作答且全错得 −c，弃答得 0；阈值 γ “which optimizes Φc on a validation set”；MaxProb 在 1% 风险下 coverage 不到 7.5%（表 1 约 5.8–7.5%） [全文·前段] — [Whitehead et al.](https://ar5iv.labs.arxiv.org/html/2204.13631)
- **Abstain-QA（MCQ + IDK/NOTA 选项，与 R1 场景最像）**：2,900 道五选一，答案可答/不可答 50/50，不可答题通过把正确选项替换成似是而非的错误选项构造；TP / FP / FN（**可答题选了 IDK**）/ TN；AR = (FN+TN)/|D|，AAC = TP/|A|，UAC = TN/|U|，P = TP/(TP+FP) [全文] — [Abstain-QA](https://arxiv.org/html/2407.16221)
  - CoT 让 GPT-4 Turbo 在 CQA 上的 UAC 从 32.8 升到 84.8，但 AAC 下降，作者描述为 “a high rate of abstention in response to answerable questions”——过度弃答的典型案例 — [Abstain-QA](https://arxiv.org/html/2407.16221)
  - GPT-4 Turbo（Base，Standard）：Pop-QA 的 P/AAC/UAC 为 84.2/91.2/83.6，MMLU 为 44.8/77.0/26.0 — [Abstain-QA](https://arxiv.org/html/2407.16221)
- **CDA**：F1_ans（可答题）与 F1_abs（不可答题）分开报告，后者 “penalizes both missed abstentions and over-abstention”；另有 RS；Context-only 基线在 NQ 上 F1_abs 接近 2（几乎从不弃答）；Llama3-8B 上 RS：NQ 63.72 vs 最佳基线 59.64 — [CDA](https://arxiv.org/html/2412.12527)
- **Trust-Align / Trust-Score**：可答 = 检索文档能蕴含至少一条金标 claim；F1_GR = (F1_ref + F1_ans)/2；Trust-Score = (F1_GR + F1_AC + F1_GC)/3；没有单独的过度拒答率，用 Answered Ratio（AR%）间接反映；LLaMA-2-7b ICL 在 ASQA 上 AR=0.00%（全拒），FRONT 在 QAMPARI 上 AR=100%（全答）；“Excessive Refusal” 被列为五类幻觉之一 [全文·前段] — [Trust-Align](https://arxiv.org/html/2409.11242)
- **Counterfactual prompting 的风险指标**：Risk = |UK|/(|AK|+|UK|)；Carefulness = |UD|/(|UK|+|UD|)；Alignment = (|AK|+|UD|)/total；Coverage = (|AK|+|UK|)/total；Mistral RC-TQ 上风险 19.00 vs 21.95，但 coverage 71.14 vs 86.38（用 coverage 换风险） — [CF prompting](https://arxiv.org/html/2409.16146)
- **Kamath**：risk–coverage AUC 与 Cov@Acc=80%/90% — [Kamath et al.](https://ar5iv.labs.arxiv.org/html/2006.09462)；**Sufficient Context**：selective accuracy（# correct / # answered）对 coverage 作曲线，另报 %Correct / %Abstain / %Hallucinate — [Sufficient Context](https://arxiv.org/html/2411.06037)
- **Conformal abstention**：测试风险与弃答率成对报告 — [Conformal](https://arxiv.org/html/2405.01563)；**LUMINA / ReDeEP**：只报 AUROC / AUPRC / PCC，阈值 F1 用 oracle 最优阈值，不是可部署的工作点 — [LUMINA](https://arxiv.org/html/2509.21875v3)、[ReDeEP](https://arxiv.org/html/2410.11414)
- **AbstentionBench**：主指标为 abstention recall，同时测 precision（用于衡量过度弃答）和 F1，非弃答回答的准确率另算 — [AbstentionBench](https://arxiv.org/html/2506.09038)

### Inferences
- 用户说 OVO-Bench 中“只要有弃答选项，它就一定是正确答案”，这意味着弃答类题目上只能测到 UAC（或 recall），**测不到 AAC 和 FN**。按 Abstain-QA、CDA、Trust-Align 的惯例，审稿人会期待一个“带弃答选项但答案可答”的对照集，例如把弃答选项加到原本可答的题目上，或者用 Abstain-QA 式的方法构造 50/50 划分，以便报告 AAC / UAC、F1_ans / F1_abs，或 ARSP（R1 估计 16–20% 的误弃答就是这个量）。
- 建议的主表组合（推断）：(i) 可答与不可答分开的准确率（AAC / UAC）；(ii) Effective Reliability 或 Φ_c（对错答加罚，比 accuracy 更能体现弃答的价值）；(iii) 若有连续分数，加 risk–coverage 曲线或 AURC，以及 Cov@Acc；(iv) 单独一行“ARSP / 过度弃答率”。仅报总体 accuracy 会被 Wen 综述和 Feng et al. 定义的这些指标框架质疑。

### Gaps
- 没有找到 OVO-Bench 类“弃答选项总是正确”数据集的 NLP 对应讨论（数据集偏置使弃答规则虚高）；Abstain-QA 用 50/50 设计规避了这个问题，但没有专门讨论它。
- Whitehead 的 Φ_c 中 c 的具体取值（如 c=1、10、100）在读到的部分未确认。

---

## 问题 5：模块怎么包装（核心贡献 / 插件组件 / 推理技巧），给了什么理论（PMI、Bayes factor、因果效应、信息增益）？

### Takeaway
推理时的有/无上下文对比方法通常包装成“**simple, training-free, plug-and-play 的推理时方法**”（CAD、CRAG、CDA、LUMINA），而且几乎都以**解决前作在某类样本上的失败**为叙事主线（AdaCAD / COIECD 修 CAD 在无冲突样本上的退化；CDA 补上“两路知识都不可用”时的弃答）。理论以 **PMI / 互信息 / KL / JSD** 为主（CAD、PMI_DC、CPMI、Du、AdaCAD），辅以典型集或熵假设（COIECD）、V-information（PVI）、共形风险保证（Yadkori）、仿射 log-odds 假设（PMI gauge）、对信号方向的统计假设检验（LUMINA）。我没有找到用 Bayes factor 或因果效应作为主理论框架的弃答论文。

### Cited Findings
- CAD：定位为 “a simple inference-time method” 和 “a contrastive ensemble”，类比 classifier-free guidance；PMI 解释；免训练、适用于任何 LM — [CAD](https://ar5iv.labs.arxiv.org/html/2305.14739)
- AdaCAD：把问题定为上下文与参数知识的冲突，“PMI raised to a per-token power set by JSD”，修正 CAD 的固定 α — [AdaCAD](https://arxiv.org/html/2409.07394)
- COIECD：理论依托 Stable Entropy Hypothesis（Arora et al. 2023）与 Locally Typical Set（Meister et al. 2023）；假设 C.1：与参数知识一致的 context 可以视为模型的自然生成；常数 γ、ε 未指定，界不可直接使用 — [COIECD](https://arxiv.org/html/2402.11893)
- CDA：自称首个把弃答加入上下文对比解码的工作；贡献 = 四场景测试床 + CDA + 动量变体；只有直觉没有定理；局限：只覆盖单 context、短答案，弃答只是一个短语 — [CDA](https://arxiv.org/html/2412.12527)
- ClashEval：贡献 = 基准（上下文 vs 先验冲突）+ 分析（先验置信度与是否采纳上下文的斜率）+ 简单的 token 概率校正方法 — [ClashEval](https://arxiv.org/html/2404.10198)
- Sufficient Context：包装为分析 RAG 的 “new lens”，选择性生成是下游应用 — [Sufficient Context](https://arxiv.org/html/2411.06037)
- Counterfactual prompting：“risk control” 框架 + 新基准 RC-TQ / RC-NQ + 后处理技术；明说用 coverage 换风险 — [CF prompting](https://arxiv.org/html/2409.16146)
- Feng et al.：方法（Cooperate / Compete）+ 一整套弃答指标 + 卖点“不需要 held-out 集” — [Feng et al.](https://arxiv.org/html/2402.00367)
- CRAG：“plug-and-play and can be seamlessly coupled with various RAG-based approaches” — [CRAG](https://arxiv.org/html/2401.15884)
- 理论框架：Du et al. 把 persuasion 定义为 half-pointwise MI = KL，susceptibility = 条件互信息 I(C;A|q) — [Du et al.](https://arxiv.org/html/2404.04633)；PVI 基于 V-usable information — [PVI](https://ar5iv.labs.arxiv.org/html/2110.08420)；PMI gauge 提出假设 2.1（PMI 与正确答案 log-odds 仿射相关）和命题 2.1 — [PMI gauge](https://arxiv.org/html/2411.07773v2)；共形弃答给出分布无关的风险上界 E[R(λ̂)] ≤ α — [Conformal](https://arxiv.org/html/2405.01563)
- LUMINA 的“理论”是对信号方向做四个单侧假设检验（如 H1：有文档的外部上下文分数 > 无文档），t 统计量多在 6–110，p < 0.001；分析指出这只说明信号方向正确，不说明它是好的检测器 — [LUMINA](https://arxiv.org/html/2509.21875v3)
- ReDeEP 的机制叙事：“Knowledge FFNs…overemphasize parametric knowledge”，“Copying Heads fail to effectively retain or integrate external knowledge”，并有因果干预实验支持 — [ReDeEP](https://arxiv.org/html/2410.11414)

### Inferences
- 对 R1 最自然的理论包装：**PMI / Bayes factor**。log p(o*|E,Q) − log p(o*|Q) = log [p(E|o*,Q)/p(E|Q)]，即证据对 o* 的 log Bayes factor，并与 CAD 的 PMI 解释、PMI_DC 的 domain-conditional PMI 一脉相承。但文献也表明，这类量在“先验已正确、证据冗余”时接近 0 或为负（PVI 的负值说明、AdaCAD 的无冲突退化），所以理论部分应同时说明“BF ≤ 1 ≠ 错误”，并给出门控或校准。
- 包装建议（推断）：参照 CDA 与 AdaCAD 的叙事——“固定的有/无证据对比在证据冗余时会误伤 → 我们用无标签的分歧 / 不确定性门控修正”——比把 R1 包装成独立的核心贡献更容易站住。也可以像 LUMINA 那样先做信号方向的统计检验（例如有记忆时的 Δ 在可答题上显著大于 0），作为“信号有效”的证据。
- 作为插件组件时，审稿人会把 CAD、AdaCAD、CDA、ClashEval TPC 视为直接可比的基线（都是一次额外前向、免训练）；ClashEval 的“随机替换对照”（同样的先验使用率下随机替换）值得照搬，用来证明 R1 的弃答不是随机的。

### Gaps
- 没有找到以**因果效应**（如 do(evidence)）或 **Bayes factor** 明确命名的 RAG 弃答方法；以上 Bayes factor 改写是本笔记的推断。
- 没有核实 CDA 附录 D 中的计算成本细节（只读到前约 100k 字符）。
