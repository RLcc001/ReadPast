# 答案端弃答模块（R1）的评测与如实呈现：文献先例

> 调研日期 2026-11-03。阅读深度标记：**[全文]** = 读了 arXiv HTML 或 PDF 正文（经 WebFetch 或本地 PDF 抽取）；**[摘要]** = 只看到 abstract 页；**[搜索摘要]** = 只来自搜索引擎摘要或二手站点，未核对原文；**[OR 评审原文]** = 通过 OpenReview API 搜索端点拿到的 review 或 meta-review 原文。所有数字均来自所列来源，没有自行推算的数字都另行标注为"推断"。
>
> **对任务简报的一处更正**：SimpleStream **没有**把 HLD 排除在 OVO 主平均之外。它的 Avg 包含 HLD，只是在 §5.3 论证"HLD is not a memory task"，并另定义了不含 HLD 的记忆指标 ER=(EPM+ASI)/2。真正在 headline 中"excluding HLD"的是 StreamOPD（见 Q3）。

---

## Q1. 哪些基准或协议把可答与不可答版本配对，或给可答题加 "None of the above / I don't know" 选项？它们用什么 headline 指标，为什么？

### Takeaway
现有先例几乎都在**同一协议里同时测"该答的题"和"该弃答的题"**：用配对（MM-UPD 的 Dual accuracy、VideoHallucer 的 pair accuracy）或混合集（Abstain-QA、UVQA、AbstentionBench）来测，并且**把弃答选项同样加到可答题上**以保证公平（MM-UPD 的 Option/Instruction setting；Tam et al. 的 NA-as-distractor；Ling et al. 的 "Unknown" 注入）。只在"必然应弃答"的子集上报告弃答收益（RGB 的 negative rejection、OVO 的 HLD）属于少数做法，而且都没有测过度弃答。

### Cited Findings

**MM-UPD（Miyai et al.，arXiv v4 2025-06；ACL 2025；ICLR 2025 被拒稿）[全文]**
- 三种不可解设定都由 MMBench 构造：AAD 遮掉正确选项（820 题），IASD 打乱答案集（919 题），IVQD 打乱图文对（356 题），共 2,095 题。— [arXiv 2403.20331](https://arxiv.org/html/2403.20331)
- 指标分三种：Standard accuracy（可解题）、UPD accuracy（不可解题），以及 **Dual accuracy**，后者 "we count success only if the model is correct on both the standard and UPD questions"。论文明确说 Dual accuracy 是主指标（"Our evaluation thus uses this as the primary metric"），并把 "Original Standard"（原始 MMBench prompt 下的标准准确率）称为 "the upper bound of Dual accuracy"。— [arXiv 2403.20331](https://arxiv.org/html/2403.20331)
- **反事实先例**：Option setting 加入 "None of the above"（IVQD 用 "The image and question are irrelevant."），Instruction setting 加入 "If all the options are incorrect, answer F. None of the above"。这些额外选项和指令**同样加到标准（可解）题上**，原文理由是 "to make a fair comparison"。— [arXiv 2403.20331](https://arxiv.org/html/2403.20331)
- 附录 Table E 给出了加选项后标准准确率下降的例子：GPT-4o 在 MMMU-AAD 上的 Standard accuracy 从 42.9（Base）降到 24.4（Option）。这是 MMMU 而不是 MM-UPD 主集，数字取自 WebFetch 抽取，建议引用前核对 PDF。— [arXiv 2403.20331](https://arxiv.org/html/2403.20331)
- 论文没有设 "always abstain" 基线。从 Dual accuracy 的定义可以推出，总选 NOTA 的模型在标准半边全错，因此 Dual accuracy 为 0；但这一点原文没有写明。— [arXiv 2403.20331](https://arxiv.org/html/2403.20331)

**VideoHallucer（Wang et al. 2024）[全文]**
- 每个样本是一对问题，一个 basic、一个 hallucinated："we introduce an adversarial evaluation with paired questions—one basic and one intentionally hallucinated"。yes 和 no 的答案数量做了平衡（"we balance 'yes' and 'no' responses to reduce language biases"）。共 5 个设定，每个 400 对，合计 1,800 题。— [arXiv 2406.16338](https://arxiv.org/html/2406.16338)
- Overall accuracy 按对计分："marking it as a hit only if both questions are answered correctly"。原文理由是 "enhancing a model's ability to recognize and counter hallucinations should not compromise its performance on fundamental tasks"。— [arXiv 2406.16338](https://arxiv.org/html/2406.16338)
- 该文同时报告两种偏置指标。Yes Percentage Difference d_y 是预测 "yes" 数与真值 "yes" 数之差；FP Ratio 是错答中 "yes" 的占比，越接近 50% 偏置越小。数值例子：VideoChatGPT 的 basic 为 92.8、hallucinated 为 10.4，FP Ratio 0.89；GPT-4o 的 basic、hallucinated、overall 分别为 75.1、74.2、53.3。— [arXiv 2406.16338](https://arxiv.org/html/2406.16338)

**Abstain-QA（Madhusudhan et al.，arXiv 2407.16221v2）[全文]**
- 全部是五选一 MCQ，可答与不可答各占 50%，**每题都带 IDK/NOTA 选项**。不可答题的构造方式是 "substituting the correct option with a plausible yet incorrect alternative"。— [arXiv 2407.16221](https://arxiv.org/html/2407.16221)
- 指标是分开报告的：Answerable Accuracy AAC=TP/|A|，Unanswerable Accuracy UAC=TN/|U|，Abstention Rate AR，以及 Precision。论文观察到加 CoT 后 "AAC often decreased, indicating a high rate of abstention in response to answerable questions"，即可答题上出现过度弃答。— [arXiv 2407.16221](https://arxiv.org/html/2407.16221)

**Tam et al. 2025，"None of the Above, Less of the Right"（Findings ACL 2025）[全文]**
- 在 28 个 LLM、MMLU 上做了两种设定：NA-as-answer（用 NA 替换正确答案）和 **NA-as-distractor（NA 作为干扰项加入可答题）**。平均准确率为 Baseline 0.738、NA-as-answer 0.350、NA-as-distractor 0.674。NA 作为正确答案时，原文结论是 "a consistent 30–50% drop"。— [arXiv 2503.01550](https://arxiv.org/html/2503.01550v1)
- 原文在 NA 是"加入"还是"替换一个干扰项"这一点上前后不一致（§3.2 与 §5.1 说法不同）。论文建议 NA 结果单独报告。— [arXiv 2503.01550](https://arxiv.org/html/2503.01550v1)

**Ling et al. 2026，"LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty"（EMNLP 2026 main，pp. 12612–12633）[全文，本地 PDF 抽取]**
- 这是最直接的"给可答题加 Unknown 选项"反事实研究。S1 是不加选项的基线，S2 加入 "Unknown"，S4 把 "Unknown" 换成随机词（如 "Triangular"）。指标只有 Acc 和 Abs Rate（输出 "Unknown" 的比例）。— [ACL Anthology 2026.emnlp-main.549](https://aclanthology.org/2026.emnlp-main.549.pdf)
- True/False 题上，"accuracy drops by 19.8 percentage points on average, with Abs Rate rising to 34.5%"；四选一 MCQ 上则 "only small fluctuations"。Table 1 里 ARC、MedQA、MMLU、LogiQA 的准确率变化在 +0.4 到 −4.2 之间，Abs Rate 在 0.2% 到 7.4% 之间。换成随机词也会触发弃答（"'Triangular' triggers 30%+ abstention on FLD"）。— [ACL Anthology 2026.emnlp-main.549](https://aclanthology.org/2026.emnlp-main.549.pdf)
- 原文建议 "Add verification steps to benchmark construction. Detailed measurements should be performed when adding uncertainty categories"，并称这一现象 "is overlooked by existing abstention benchmarks"。— [ACL Anthology 2026.emnlp-main.549](https://aclanthology.org/2026.emnlp-main.549.pdf)

**UVQA（Yoon et al.，ICLR 2025 poster；arXiv 2507.04976）[全文]**
- 评测集由 300 道不可答题（UVQA）和 300 道可答题（ActivityNet-QA）组成，两者不成对，但放在一起评测。— [arXiv 2507.04976](https://arxiv.org/html/2507.04976v1)
- 提出三个分项指标。**Excessive Refusal** S_ex-ref = N3/(N1+N2+N3)，衡量原本答对的可答题现在被拒答的比例。Permissiveness 衡量原本拒答的可答题现在被回答的比例。Discretion 衡量不可答题被正确拒答的比例。三者合成 S_align = ⅓((1−S_ex-ref)+S_permis+S_disc)。N1–N12 的精确映射在 Figure 2 中，抽取文本里没有，需对照 PDF。— [arXiv 2507.04976](https://arxiv.org/html/2507.04976v1)

**AbstentionBench（Kirichenko et al.，NeurIPS 2025 D&B poster）[全文]**
- 共 20 个数据集、约 35k 例。多数数据集包含可答对照，例如 "we duplicate the original answerable questions before removing all context"。— [arXiv 2506.09038](https://arxiv.org/html/2506.09038)
- Headline 是 abstention recall，理由是 "as we find that models generally exhibit high abstention precision, we focus on abstention recall"。precision 和 F1 也报告，用于反映过度弃答。是否弃答由 Llama-3.1-8B judge 判定，在人工标注样本上准确率 88%。— [arXiv 2506.09038](https://arxiv.org/html/2506.09038)

**RGB（Chen et al.，AAAI 2024）[全文]**
- Negative Rejection 测试床只含噪声文档（"all the external documents are sampled from negative documents"），指标是拒答率，分精确匹配 Rej 和 ChatGPT 判定 Rej* 两种。**这个协议不测"该答时拒答"**。最高 Rej* 为 45.00（英文，ChatGPT）。— [arXiv 2309.01431](https://arxiv.org/html/2309.01431)

**OVO-Bench（Li et al.，CVPR 2025）[全文]**
- HLD 的定义只有一句："Ask questions irrelevant to existing video inputs."（§3.1.1）。正文没有给出 HLD 题数、构造方法，也没有说明正确答案总是 "Unable to answer"。— [arXiv 2501.05510](https://arxiv.org/html/2501.05510)
- 按 Table 1 的数值可以复算出：Backward Avg = (EPM+ASI+HLD)/3，Overall = 三大类平均的均值（以 Gemini 1.5 Pro 复算吻合，属推断）。**Table 1 中纯文本的 Blind GPT-4-turbo 在 HLD 上得 70.05，高于表中所有视频模型**；Dispider 只有 4.3，Flash-VStream 5.91，LLaVA-Video 7.53。论文没有评论这一现象。— [arXiv 2501.05510](https://arxiv.org/html/2501.05510)

**SQuAD 2.0（2018，早于时间窗，仅作背景）[搜索摘要]**
- 官方评测脚本除总分外还分别输出 HasAns_f1 和 NoAns_f1。对不可答题，gold 为空串，"F1 is 1 if they agree, 0 otherwise"。— [HF evaluate squad_v2 compute_score.py](https://huggingface.co/spaces/evaluate-metric/squad_v2/blob/8c584fcc4b6e58b83d24c3d10b8089e24f7f1031/compute_score.py)

### Inferences
- OVO HLD 属于 RGB 那类"只含应弃答样本"的协议。在这种协议里，任何提高弃答倾向的规则都只会加分，与 Ling et al. 所说的 "Abstention Inflation" 盲区相同。主流基准（MM-UPD、VideoHallucer、Abstain-QA、UVQA）都通过配对、混合或给可答题同样加选项来堵住这个盲区。
- Blind GPT-4-turbo 在 HLD 上拿到 70.05，强烈提示 HLD 分数主要反映"是否倾向于选弃答项"这一先验，而不是视觉证据利用能力。这可以作为在论文中把 HLD 单列的正当理由。
- 我们离线做的"给可答题加 'Unable to answer' 选项"实验，与 MM-UPD 的 Option setting、Tam 的 NA-as-distractor、Ling 的 S2 是同一类设计，可以直接引用这三者作为方法学先例。
- 量级参照（推断，机制不同，不宜直接比较）：Ling et al. 中 prompt 级 "Unknown" 注入在四选一 MCQ 上只造成 0.2%–7.4% 的弃答率；Tam et al. 的 NA-as-distractor 平均让准确率下降约 6.4 点（0.738→0.674）。R1 在记忆题上翻转 16%–20% 的正确答案，**明显高于**这些 prompt 级 MCQ 先例。

### Gaps
- TUBench [摘要]：只看到 headline 为不可答判别的 "average accuracy"（Gemini-1.5-Pro 69.2%），配对方式与 P/R/F1 是否报告未核实。— [arXiv 2410.04107](https://arxiv.org/abs/2410.04107)
- UNK-VQA [摘要，TPAMI]：只确认了 "deliberate perturbations on either the image or question"，指标和原题是否同评未核实。— [arXiv 2310.10942](https://arxiv.org/abs/2310.10942)
- 未核实：EventHallusion、VidHal、HallusionBench 的 pair 指标；FreshQA 的 false-premise strict/relaxed 评测；VizWiz 可答性任务的 AP 指标；Kadavath et al. 2022 关于用 NOTA 替换选项的实验（abstract 页没有该内容）。
- OVO-Bench HLD 的 186 题和"金标总是 Unable to answer"两点，在论文正文中都找不到，只能依据我们自己对数据的检查。

---

## Q2. 指标：coverage–risk 和 AURC、Effective Reliability Φ_c、abstain P/R/F1、dual accuracy、over-refusal、Brier 和 ECE，2024–2026 年 MLLM 论文常用哪些？

### Takeaway
两个圈子的做法不同。**MLLM 和视频幻觉或可答性基准（2024–2026）**几乎都用**按子集拆开的准确率**：Standard/UPD/Dual、basic/hallucinated/pair、AAC/UAC/AR、Excessive-Refusal/Discretion，外加一个偏置指标（yes-ratio 或 FP ratio）。**选择性预测或 Reliable-VQA 一脉**用 coverage@risk、AUC 和 Φ_c，阈值在 held-out 集上选定。LLM 弃答综述的立场是"没有单一指标"，并建议同时报告错误回答率与错误弃答率。

### Cited Findings

**Effective Reliability Φ_c（Whitehead et al.，ECCV 2022）[全文，本地 PDF 抽取]**
- 逐样本定义：答了且 Acc>0 时得 Acc(x)；答了且 Acc=0 时得 −c；弃答 g(x)=0 时得 0。总分取所有样本的均值。论文报告 c=1、10、100 三档。弃答记 0 分（而非负分）是为了让 "Lemma 1. The Effective Reliability score is equal to the VQA Accuracy … if a model abstains iff it is incorrect" 成立，从而有清晰的上界。— [arXiv 2204.13631](https://arxiv.org/pdf/2204.13631)
- 阈值选取："we choose a threshold γ which optimizes Φc on a validation set"。论文还强调 "Φc can be evaluated for any model, even those which do not incorporate the option to abstain"。— [arXiv 2204.13631](https://arxiv.org/pdf/2204.13631)
- 表格中同时列出**不弃答行**（"—"）和**Best Possible 行**（只保留正确预测的 oracle）。例如 Pythia 不弃答时 Φ1=36.97、Φ10=−211.96、Φ100=−2701.25（覆盖率 100%），Best Possible 时 Φ=64.63。论文还报告 coverage@risk（C@1%、5%、10%、20%）和 risk–coverage 曲线下的 AUC。— [arXiv 2204.13631](https://arxiv.org/pdf/2204.13631)

**Feng et al. 2024，"Don't Hallucinate, Abstain"（ACL 2024）[全文]**
- 用四格表定义指标：A 为答且对，B 为弃答但本可答对，C 为答且错，D 为弃答且本会答错。Reliable Accuracy = A/(A+C)；Effective Reliability = (A−C)/N；**Abstain Accuracy = (A+D)/N**；Abstain Precision = D/(B+D)；Abstain Recall = D/(C+D)；另有 A-F1 和 Abstain ECE。— [arXiv 2402.00367](https://arxiv.org/html/2402.00367)
- 关键协议：在**全部可答**的数据集上（如 MMLU），"abstain 是否正确"由模型自身的正误决定："LLMs should abstain when it would provide an incorrect answer and vice versa"。基线中包括 NOTA，做法是 "adding an extra nota option to every question"。— [arXiv 2402.00367](https://arxiv.org/html/2402.00367)

**Wen et al.，"Know Your Limits" 弃答综述（arXiv v3 2025-02）[全文]**
- 综述用统一四格记号列出了 Abstention Accuracy、Precision、Recall、F1、Coverage、Abstention Rate、Reliable Accuracy、Effective Reliability、Abstain ECE、Coverage@Acc、AURCC、AUACC、AUROC 等指标。其中 **ARSP（Abstained Responses on Safe Prompts，"over-conservativeness score"）= N3/(N1+N2+N3)** 用来衡量在应答题上的弃答率。— [arXiv 2407.18418](https://arxiv.org/html/2407.18418)
- 原文立场："No single metric captures all aspects of performance"。错误率方面，综述建议同时看 URUP（错误回答）和 ARSP（错误弃答）。— [arXiv 2407.18418](https://arxiv.org/html/2407.18418)

**过度弃答专门指标**
- UVQA 的 **Excessive Refusal** 衡量"原本答对、现在被拒"的比例 [全文]。— [arXiv 2507.04976](https://arxiv.org/html/2507.04976v1)
- Srinivasan et al. 2024，"Selective 'Selective Prediction'"（arXiv 2402.15610）[全文] 定义了 **Selective Prediction Recall**，即所有正确答案中实际被给出的比例。该文报告，在 90% reliability 目标下，BLIP2 在 A-OKVQA 上只回答约 4% 的题，并对约 94% 的正确预测弃答。这些数字取自 WebFetch 对原文的转述。阈值 "using a calibration set" 选定。— [arXiv 2402.15610](https://arxiv.org/html/2402.15610v1)

**偏置指标**
- VideoHallucer 报告 Pct. Diff 和 FP Ratio [全文]。— [arXiv 2406.16338](https://arxiv.org/html/2406.16338)
- POPE 报告 Yes 比例，并采用 1:1 的平衡设计 [搜索摘要]。— [arXiv 2305.10355](https://arxiv.org/pdf/2305.10355)

**2023–2024 年的选择性 VQA 后续工作 [搜索摘要]**
- Dancette et al. 2023 使用 C@1%。— [arXiv 2306.08751](https://arxiv.org/pdf/2306.08751)
- Eisenschlos et al. 2024（"Selectively Answering Visual Questions"）在 VizWiz 和 UNK-VQA 上用 coverage@accuracy。— [themoonlight 评述](https://www.themoonlight.io/review/selectively-answering-visual-questions)
- Ortiz 2025（arXiv 2601.00138）在 NExT-QA 上研究视频 QA 的置信弃答。只看到搜索摘要。— [arXiv 2601.00138](https://arxiv.org/abs/2601.00138v1)

### Inferences
- R1 是一个阈值固定为 0 的二元规则（比较两个 likelihood 的差）。只有一个工作点时，**分项准确率 + Φ_c + abstain P/R** 比 AURC 更合适。如果想画 risk–coverage 曲线，可以把 log-likelihood 差当作连续分数扫阈值，再按 Whitehead 和 Srinivasan 的做法在 held-out 集上选阈值。
- 我们离线测得的"翻转 16%–20% 正确记忆答案"，**等价于 UVQA 的 S_ex-ref，也等价于 1 − Srinivasan 的 Selective Prediction Recall**；"捕获 26%–35% 幻觉 HLD 答案"等价于 Feng 意义下的 abstain recall（以 base 答错为应弃答）。这两个名称都有先例可引用，不需要自造指标。
- 量级参照（推断，设置不同）：UVQA 中 prompt 级弃答基线的 S_ex-ref 为 0.08 和 0.12，DPO 对齐后的 Video-LLaVA 为 0.14，SFT 对齐为 0.50–0.66。R1 的 0.16–0.20 处于 prompt 基线与 DPO 对齐之上、SFT 对齐之下。
- 2024–2026 年 MLLM 论文里很少见到 Brier 或 ECE 作为弃答的主指标，ECE 主要出现在 LLM 校准类工作中（Feng 的 Abstain ECE）。这只是本轮阅读范围内的观察，不是系统统计。

### Gaps
- 没有对 2024–2026 年 MLLM 论文做系统统计，无法给出各指标的使用频次。
- AURC 原始文献（Geifman & El-Yaniv 2017）早于时间窗，未读。
- ADP（ICLR 2027 投稿，"Training-Free and Generation-Free Answerability Estimation"）是同期的免训练可答性估计工作，只读到 OR 摘要，其指标未知。— [OpenReview 9qhFrlcwIi](https://openreview.net/forum?id=9qhFrlcwIi)

---

## Q3. 写作先例：收益由单一子集驱动时如何披露，审稿人如何质疑，以及是否设置 always-abstain 或先验基线

### Takeaway
最贴近我们的先例是 **StreamOPD**：摘要的 headline 写 "OVO-Bench excluding its hallucination-detection subtask (HLD)"，HLD 另行单报，并把它明确当作"弃答"诊断。SimpleStream 的做法是保留 HLD，但用专节说明 "HLD is not a memory task"。审稿方面，有人批评 Dual accuracy 把通用能力与拒答能力耦合，建议改用条件化指标；有人要求补 prompt 弃答基线和对其他任务的副作用；还有人质疑同分布评测会高估收益。所查到的公开 review 中**没有**直接要求 "always abstain" 基线，但 Whitehead 和 Feng 都在表中放了"不弃答"行或 oracle 行，或者把退化为几乎总弃答的方法标记为无效。

### Cited Findings

**HLD 的处理方式**
- **StreamOPD**（arXiv 2608.16320，2026-08）[摘要] 在 headline 中写 "improves OVO-Bench excluding its hallucination-detection subtask (HLD) by 9.1 points under unchanged inference" 和 "71.9% on OVO-Bench (excluding HLD)"，另报自蒸馏变体 "lifts HLD to 57.0% … so abstention loss is not intrinsic to the recipe"。— [arXiv 2608.16320](https://arxiv.org/abs/2608.16320)
- **SimpleStream**（arXiv 2604.02317）[全文] 的 Table 1 注明 "'Avg.' is the mean of Real-Time and Backward category averages"，可见 Avg 包含 HLD。§5.3 写道 "Placing HLD under Backward Tracing therefore conflates two distinct abilities: memory recall and hallucination robustness" 和 "We therefore caution against interpreting HLD as a direct measure of long-range memory"，记忆指标另定义为 ER=(EPM+ASI)/2。— [arXiv 2604.02317](https://arxiv.org/html/2604.02317)

**表格如何呈现弃答方法**
- Whitehead et al. 在 Φ_c 表中并列 "—"（不弃答）、MaxProb、Calibration、Selector 和 "Best Possible" 五行。— [arXiv 2204.13631](https://arxiv.org/pdf/2204.13631)
- Feng et al. 的 Table 1 注释写道："'-' indicates that this approach fails to produce meaningful abstain decisions: almost always abstain, didn't follow instructions, etc."。正文还指出 "Instruction Tuning collapses and produces 100% abstention due to imbalances in the held-out set in K-Crosswords"。也就是说，几乎总弃答的结果被视为无效，不计分。— [arXiv 2402.00367](https://arxiv.org/html/2402.00367)
- UVQA 在附录 A.2 承认 "A key limitation of our approach is the increase in the excessive refusal score"。这条局限写在**附录**而不是正文。— [arXiv 2507.04976](https://arxiv.org/html/2507.04976v1)

**OpenReview 审稿意见 [OR 评审原文]**
- **MM-UPD（ICLR 2025 投稿 #51，被拒）**：审稿人 Jv8R4AYgFQ（rating 6）质疑 Dual accuracy："a model might correctly refuse to answer absurd questions even if it is not capable of answering the standard version. This coupling of general VQA ability and the capacity to reject unanswerable questions could skew comparisons"。他建议 "compare success rates on questions where the model correctly answers the standard version … (Mathematically, it's similar to divide Dual Accuracy by Standard Accuracy)"。— [OpenReview K4YMFdx2Z2](https://openreview.net/forum?id=K4YMFdx2Z2)
- MM-UPD 的 meta-review 给出的拒稿主因是只覆盖 MCQ（"limited to multiple-choice problems"），以及基准多样性不足。另一位审稿人（O9mJErb7Zb，rating 5）批评以 "low standard accuracy" 为由排除 MMMU 和 MathVista "may not be a sufficiently robust justification"。— [OpenReview K4YMFdx2Z2](https://openreview.net/forum?id=K4YMFdx2Z2)
- **UVQA（ICLR 2025 投稿 #10560，录用）**：一位审稿人要求补 prompt 基线："important to show if the problem can be migrated by explicit prompting, e.g., add to the prompt 'Say can't answer if you are not sure'"，并指出 "the models might drop performance on other tasks"。另一位审稿人提出 "The evaluation set is curated from the same distribution as the training set, which may overestimate the benefits"。作者随后在附录 A.9 补了 prompt 基线。— [OpenReview P9VdRQOyqu](https://openreview.net/forum?id=P9VdRQOyqu)；[arXiv 2507.04976](https://arxiv.org/html/2507.04976v1)
- **AbstentionBench（NeurIPS 2025 D&B，Accept poster）**：一位审稿人批评不可答 split 的构造方式 "simply copy each problem and then strip all pre-question context, making the task impossible by construction … gains on these splits could fail to transfer"。另一位审稿人肯定了 "a clear focus on abstention recall as the primary metric"。— [OpenReview OkHC30LLpO](https://openreview.net/forum?id=OkHC30LLpO)

**呼吁把弃答当作独立轴单列**
- Ling et al. 写道 "LLM abstention should be treated as a double-edged sword, rather than an always-positive signal"。— [ACL Anthology 2026.emnlp-main.549](https://aclanthology.org/2026.emnlp-main.549.pdf)
- Tam et al. 建议 NA 结果单独报告。— [arXiv 2503.01550](https://arxiv.org/html/2503.01550v1)

### Inferences
- 对我们最直接的写法是沿用 StreamOPD：**主表 headline 用 OVO（excl. HLD）**，并另列 HLD 和 Overall（incl. HLD）。这样 R1 的 +13–15 HLD 不会混进记忆收益里。SimpleStream 的 §5.3 可以作为"HLD 不是记忆任务"的佐证。
- 那位 MM-UPD 审稿人提出的"条件化"指标（在标准题答对的前提下看是否正确拒答），正好对应我们的"在 base 答对的题上 R1 保留了多少"，即 1 − S_ex-ref。用这个指标呈现 R1 的代价，有审稿先例可循。
- 按 UVQA 审稿人的思路，预计审稿人会要求三类对照：(i) 只用 prompt 弃答指令；(ii) 在其他答案端任务上的副作用；(iii) 不同分布上的验证。R1 免训练，因此 (iii) 主要指在 OVO 以外的基准上验证。
- 公开 review 中没有找到"always abstain"的直接批评。但从 Whitehead 的 Φ_c 定义可以推出 always-abstain 的 Φ_c=0（定义上成立），从 MM-UPD 的定义可以推出 always-abstain 的 Dual=0（推断）。加上这一行成本很低，可以抢先回应"HLD 金标恒为弃答"的质疑。

### Gaps
- OVO-Bench 发表于 CVPR，没有公开 review，找不到针对 HLD 设计的审稿意见。
- 没有找到"收益几乎全部来自幻觉子集"且被审稿人点名的视频 QA 论文实例。
- OpenReview 的 notes 接口被 challenge 拦截，以上 review 是经 search 端点拿到的。各篇作者 rebuttal 的全文没有逐条核对，例如 MM-UPD 对 Dual accuracy 质疑的回应未读到。

---

## Q4. 文献对"免训练弃答规则"最小公平评测的建议，以及可直接套用的协议

### Takeaway
综合先例，最小公平评测有四个要素：(1) 同一协议里**同时报告**应答题准确率和应弃答题检测率（MM-UPD、Abstain-QA、UVQA、AbstentionBench）；(2) 把**弃答选项同样加到可答题上**，测量过度弃答（MM-UPD Option setting、Tam、Ling、Feng 的 NOTA）；(3) 给出不弃答、总弃答（或 prompt-only）、oracle 三类参照行，单指标用 Φ_c 或 Abstain Accuracy；(4) 阈值或规则需在 held-out 集上确定，或者本身就不调参，并说明这一点（Whitehead、Srinivasan）。文献中没有一篇专门为"免训练弃答规则"写的评测指南，以下多为综合推断。

### Cited Findings
- 保证公平的核心操作是把额外选项也加到标准题上（"to make a fair comparison"），然后报告 Standard、UPD、Dual 三种准确率。— [MM-UPD](https://arxiv.org/html/2403.20331)
- 在全部可答的数据上，以"模型自身会答错"作为应弃答的标签，计算 A-Acc、ER、A-P/R/F1；退化为几乎总弃答的方法记为 "-"。— [Feng et al. 2024](https://arxiv.org/html/2402.00367)
- Φ_c 取 c=1/10/100，阈值在验证集上选；表中放不弃答行和 Best Possible 行。— [Whitehead et al. 2022](https://arxiv.org/pdf/2204.13631)
- 用 Excessive Refusal 和 Discretion 分别报告代价和收益；补 prompt 弃答基线（附录 A.9，原文称模型 "fail to benefit from the explicit answerability prompt"）。— [UVQA](https://arxiv.org/html/2507.04976v1)
- "Detailed measurements should be performed when adding uncertainty categories"；报告 Acc 与 Abs Rate 在加选项前后的变化。— [Ling et al. 2026](https://aclanthology.org/2026.emnlp-main.549.pdf)
- 报告弃答 recall，同时报告 precision 和 F1 以反映过度弃答。— [AbstentionBench](https://arxiv.org/html/2506.09038)
- 报告偏置指标（yes-ratio 或 FP ratio），采用平衡设计或配对计分。— [VideoHallucer](https://arxiv.org/html/2406.16338)；[POPE](https://arxiv.org/pdf/2305.10355)
- 同时报告 URUP（错误回答）和 ARSP（错误弃答），因为 "No single metric captures all aspects"。— [Wen et al. 综述](https://arxiv.org/html/2407.18418)

### Inferences（R1 评测与论文放置的具体建议）

**1. 主表**
- OVO 列出 "Overall (excl. HLD)" 作为 headline（StreamOPD 先例），并另列 "HLD" 和 "Overall (incl. HLD)"。
- 记忆方法（ReadPast 主体）的收益只在 excl. HLD 列上论证。R1 单独成行，例如 "+R1 (answer-side)"，并在表注中写明 "HLD gold is always 'Unable to answer'; R1 can only abstain, hence cannot lose HLD points"。

**2. R1 的独立评测小节（建议正文给精简版，附录给全版）**
- **(a) 反事实注入协议**：对 OVO 中所有可答任务（Real-Time 与 EPM、ASI）都加入 "Unable to answer" 选项，这是对标 MM-UPD Option、Tam 的 NA-as-distractor 和 Ling 的 S2 的做法。报告四项：加选项后无 R1 的准确率；加选项后有 R1 的准确率；**Excessive Refusal**，即 base 答对却被 R1 翻成弃答的比例（UVQA 定义；我们已有 16%–20% 与 3.5%–6% 两组数字）；Abs Rate。
- **(b) 应弃答侧**：在 HLD 上报告 R1 的 abstain recall，按 Feng 的定义只计 base 答错（幻觉）的题，即我们已有的 26%–35%；同时报告 HLD 原始分的增量。
- **(c) 单指标汇总**：把 HLD 并入 (a) 的集合，计算 Abstain Accuracy (A+D)/N、Φ_1 和 Φ_10。对照行包括 base（不弃答）、R1、always-abstain（Φ=0）、prompt-only 弃答指令，以及 oracle（只在 base 答错时弃答）。
- **(d) 先验对照**：报告一个 question-only 或 blind 的选项先验。OVO 论文中 Blind GPT-4-turbo 的 HLD 为 70.05，可以引用来说明 HLD 可被语言先验解决。

**3. 措辞**
- 不宜把 R1 写成通用的幻觉缓解方法。更稳妥的定位是"在带弃答选项的 MCQ 上，以选择性预测换取幻觉检测；其代价在加弃答选项的可答题上为 X% 的过度弃答"。
- 局限一节需要明说两点：OVO 中只有 HLD 带弃答选项；R1 的 +2.2–2.6 overall 与记忆无关，因为它在 SimpleStream 上增益相当。UVQA 只在附录承认 excessive refusal，而我们的代价数字较大，建议在正文局限段至少给出一句带数字的说明。

**4. 放置**
- 鉴于 R1 与记忆无关，且在可答题上的过度弃答比例（16%–20%）高于 prompt 级 MCQ 先例（Ling ≤7.4% 的 Abs Rate；Tam 平均 −6.4 点），建议把 R1 降级为"可选的答案端插件或分析"，放在独立小节或附录。主张应基于 excl.-HLD 数字，避免审稿人用 "always abstain gets HLD" 一类的论点否定整体收益。
- 如果要保留 R1 进正文，必须同时给出 (a) 到 (c) 的双侧数字。MM-UPD 的审稿意见和 UVQA 的审稿意见都表明，审稿人会追问可答侧的代价和 prompt 基线。

### Gaps
- 没有找到专门针对"免训练、基于 likelihood 比较的弃答规则"的评测规范或审稿先例。上述协议是对多篇论文做法的组合推断。
- "以 question-only likelihood 作对照"一类 PMI 或对比式判据，在弃答文献中的评测先例本轮未检索（按要求不研究我们自己的方法）。
- Ling et al. 的结论来自 LLM 的文本 TFQ 和 MCQ，在视频 MLLM 和流式场景下是否同样成立没有证据。
