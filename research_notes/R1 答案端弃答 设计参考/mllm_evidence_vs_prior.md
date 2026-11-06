# MLLM / VQA 中"有证据 vs 无证据(语言先验)"比较方法目录 —— 供 R1 重设计参考

> 阅读深度标记：**[FT]** = 读了全文(arXiv HTML / ar5iv / PDF 全文抽取后 grep 或逐段读);**[FT-partial]** = 读了全文前 ~100k 字符,附录未读;**[ABS]** = 只读了摘要/元数据;**[SNIP]** = 只看到搜索摘要或二手转述,数字未核实。
> 调研日期 2026-11-03。所有数字均来自对应链接,没有自己推算的数字(除非标"推断")。

---

## Q1. 对比/先验相减类方法(VCD, M3ID, ICD, IBD, PAI, AGLA, SID, VDD, CMI-VLD, CF-VQA, RUBi, LMH 等):减的是什么、token 级还是答案级、α/β 怎么定、用于弃答还是只重排

### Takeaway
几乎所有 MLLM 对比解码方法都是 **token 级、解码时重排**,形式统一为 `softmax[(1+α)·logit(y|证据) − α·logit(y|弱化/去除证据)]` + 自适应可信约束 `p(y|证据) ≥ β·max p`,超参基本沿用 VCD/对比解码惯例(α=1, β=0.1),**没有一个用于弃答**;而且 2025 年的复现研究(Mirage)表明这些方法在 POPE 上的增益主要来自"单向改变 yes/no 比例"和可信约束把采样变成近似贪心,而非真正去幻觉。VQA 去偏老文献(CF-VQA)是**答案级**做法,用因果 TIE = TE − NDE 减去"只有问题"的分支,并发现反事实分支用**均匀分布假设**(可学习常数 c、用 KL 匹配锐度)远好于用"真实先验"假设。

### Cited Findings

**A. MLLM token 级对比解码(训练无关)**

- **VCD (Leng et al., CVPR 2024) [FT]**:`p_vcd(y|v,v′,x) = softmax[(1+α)·logit_θ(y|v,x) − α·logit_θ(y|v′,x)]`,v′ = 对原图按前向扩散加 T 步高斯噪声;α=0 退化为普通解码 — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2311.16922)
  - 自适应可信约束:`V_head = {y: p_θ(y|v,x,y<t) ≥ β·max_w p_θ(w|v,x,y<t)}`,集合外 token 概率置 0 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922)
  - 默认 α=1, β=0.1;T=500(MME, LLaVA-Bench)、T=999(POPE) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922)
  - 消融(MME, LLaVA-1.5;Hallucination/Perception/Recognition 总分):α=0.25→583.89/1322.25/330.24,α=1.0→591.67/1340.89/323.45;β=0→577.22/1299.04/302.98,β=0.1→591.67/1340.89/323.45,**β=0.5→635.00/1474.02/331.43**(β=0.5 全面优于默认 0.1,论文仍用 0.1 未解释);T=200→586.67/…,T=999→557.78/… — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922)
  - 论证方式:经验分析而非理论——视觉不确定性(加噪)放大语言先验(香蕉颜色例子)和统计共现偏差;贡献包装为"分析 + 训练无关方法 + 实验" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922)
  - 用途:仅解码重排,不弃答 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922)
- **M3ID (Favero et al., CVPR 2024) [FT-partial]**:对比**有图条件 log-prob l_c 与无图(纯文本)log-prob l_u**;修正项 `(1−γ_t)/γ_t · (l_c − l_u)`,**仅当有图模型 top 概率低于阈值 α 时才施加**;γ_t = exp(−λt),随生成长度增大修正 — [arXiv HTML](https://arxiv.org/html/2403.14003)
  - PMI 解释:max_y log p(y|x,c)/p(y|x) = max (l_c − l_u),即最大化视觉输入与文本 token 的 PMI;理论包装为"fading memory"模型 l = γ·l* + (1−γ)·l_u 解出 l* — [arXiv HTML](https://arxiv.org/html/2403.14003)
  - 超参**用有标注的留出集网格搜索**:500 张不与评测重叠的 COCO val 图,以 CHAIR 为指标;α∈{0,0.01,0.1,0.3,0.5,0.8,1.0},λ∈{0.001,…,0.03};POPE 上 t₀=10(跳过模板 token) — [arXiv HTML](https://arxiv.org/html/2403.14003)
  - 过度补偿:LLaVA-7B 基线 CHAIRi 8.0/Cover 53.5;α=0.3,λ=0.02 → 5.8/55.0;λ=0.1 → 7.2/**Cover 跌到 45.5**("overcompensation") — [arXiv HTML](https://arxiv.org/html/2403.14003)
  - 附带指标 PDM(有图/无图下一 token 分布的 Hellinger 距离)只作分析,不用于检测/弃答 — [arXiv HTML](https://arxiv.org/html/2403.14003)
- **ICD (Wang et al., 2024) [FT]**:减去"扰动指令"(在原指令前加角色前缀,如 "You are a confused object detector…")条件下的分布;token 级;contrast 权重 λ=1、可信阈值(该文记为 α)=0.1,"in line with previous studies",**没有调参过程、没有 λ sweep**;不弃答 — [arXiv HTML](https://arxiv.org/html/2403.18715)
- **IBD (Zhu et al., 2024) [FT]**:反方向——用"图像偏置模型"(注意力 logit 对图像 token 加 ε)减原模型:`L_CD = logit_θ̂ − logit_θ`,采样 `softmax(logit_θ + α·I·L_CD)`;动态权重 I = min(I_sim, I_con),I_sim 为两分布 JSD(太相似视为噪声降权),I_con 为末层与早退层 JSD(内容词 vs 功能词) — [arXiv HTML](https://arxiv.org/html/2402.18476)
  - 默认 ε=2, α=1.5×10⁴, β=0.1, 早退层 24;完整方法含在 COCO caption 上微调的可学习 prompt(**完整版并非完全训练无关**);消融去掉 I_con 时 LLaVA-1.5 CHAIR_S 12.7→16.5 — [arXiv HTML](https://arxiv.org/html/2402.18476)
- **PAI (Liu et al., ECCV 2024) [FT]**:放大对图像 token 的注意力(α 步长)+ 减去**无图文本分支**:`p = γ·p(y|X_V,X_I,X_H) − (γ−1)·p(y|X_I,X_H)`,**γ=1.1**(很小的先验减除)且跨模型固定;无 β;论证概念 "text inertia"(去掉图像仍生成同样幻觉)— [arXiv HTML](https://arxiv.org/html/2407.21771)
  - γ sweep(LLaVA CHAIR_S/CHAIR_I/F1):1.1→24.6/6.8/75.7;1.5→25.0/9.6/74.6;2.0→24.2/7.1/74.0;MiniGPT-4 上 γ 增大 CHAIR 降但 F1 从 70.3 降到 64.3(γ=2.0)——**减先验越强,覆盖/召回越受损**;未单独报告 γ=1(只有注意力干预)的消融 — [arXiv HTML](https://arxiv.org/html/2407.21771)
- **AGLA (An et al., CVPR 2025) [FT]**:**加法而非减法**——`softmax[logit(y|v,t) + α·logit(y|v_aug,t)]`,v_aug 为 GradCAM 掩掉与 prompt 不相关区域的局部视图;α=2(多数基准),β=0.5(多数基准);"w/o assembly(只用增广图)"POPE-popular Acc 83.53 vs 完整 86.12 vs 常规 81.88 — [arXiv HTML](https://arxiv.org/html/2406.12718)
- **SID (Huo et al., ICLR 2025) [FT-partial]**:对比分支只保留早期层后**注意力最低的 10% 视觉 token**(CT2S),诱导"上下文相关"的幻觉再减掉;公式同 VCD,α=1,β 沿用 VCD/ICD — [arXiv HTML](https://arxiv.org/html/2408.02032)
  - 对 VCD/ICD 的批评:扰动与视觉/文本无关,"do not always amplify desired hallucinations",且推理成本翻倍 — [arXiv HTML](https://arxiv.org/html/2408.02032)
  - **去掉可信约束(Eq.3)后采样 POPE adversarial:VCD 76.2、ICD 75.4,低于常规解码;SID 82.2**;贪心 POPE random:Normal 88.8, VCD 87.8, ICD 87.9, SID 89.3 — [arXiv HTML](https://arxiv.org/html/2408.02032)
- **VDD / Post-Hoc Debias ("Debiasing Multimodal LLMs", Zhang et al. 2024) [FT]**:先验 = 无意义视觉输入下的输出,参考输入用 None(无图)/Unk(`<unk>` 占位)——带噪图/全零图等"无关图像"效果不稳定所以默认用无图 — [arXiv HTML](https://arxiv.org/html/2403.05262)
  - 分类/多选:**Post-Hoc Debias** `y = softmax(W·p + b)`,`W = diag(p′)⁻¹`, b=0(即 Zhao 式除以无图先验);开放生成:**VDD** `softmax[(1+α)l(y|x,v) − α l(y|x,v′)]`,α=1, β=0.1 全部固定 — [arXiv HTML](https://arxiv.org/html/2403.05262)
  - 失败模式:**在 MMMU 这类纯文本已有竞争力的任务上 Post-Hoc Debias "may prove detrimental"**(VDD 平均 34.3 vs Naive 35.7);"Both"(None+Unk 平均)无明显提升;强调整可能伤害高置信样本 — [arXiv HTML](https://arxiv.org/html/2403.05262)
  - MME(LLaVA-1.5-7B):Naive 891.19;Post-Hoc None 952.50;VCD 898.81;VDD-Both 966.07 — [arXiv HTML](https://arxiv.org/html/2403.05262)
  - 不用于弃答 — [arXiv HTML](https://arxiv.org/html/2403.05262)
- **CMI-VLD / C-PMI (NeurIPS 2025) [FT-partial]**:目标显式写成 PMI:每步 `log p(y_t|v,x,y<t) − log p(y_t|x,y<t)`,采样 `softmax[(1+λ)f(·|v,x) − λ f(·|x)]`,**λ=0.5**;另训练一个视觉 token 净化器(所以整体不是完全训练无关);声称 VCD 是其目标的特例;报告 VCD 在 LLaVA-1.5 POPE random 83.77% 低于默认 85.20% — [arXiv HTML](https://arxiv.org/html/2505.19678)
- **EventHallusion / TCD(视频)[SNIP]**:Temporal Contrastive Decoding 用下采样/打乱帧的视频作对比分支(模糊时序、保留空间),VCD 式减法,训练无关 — [arXiv](https://arxiv.org/abs/2409.16597)

**B. 文本域同构方法(上下文 vs 无上下文)**

- **CAD (Shi et al., NAACL 2024) [FT]**:`ỹ ∝ p(y|c,x)·(p(y|c,x)/p(y|x))^α` = product-of-experts(原分布 × α 次方 PMI),整理后 `softmax[(1+α)logit(y|c,x) − α logit(y|x)]`;类比 classifier-free guidance;**α=0.5 用于摘要、α=1 用于知识冲突**,"α=0.5 generally yielded good results" — [arXiv](https://arxiv.org/abs/2305.14739)

**C. VQA 去语言先验(带训练,但答案级)**

- **CF-VQA (Niu et al., CVPR 2021) [FT]**:因果图 Q→A、V→A、(V,Q)→K→A;`TE = Z_{q,v,k} − Z_{q*,v*,k*}`,`NDE = Z_{q,v*,k*} − Z_{q*,v*,k*}`(只有问题时的直接效应 = 语言偏置),**推理用 `TIE = TE − NDE = Z_{q,v,k} − Z_{q,v*,k*}` 取 argmax**,"totally different from … posterior probability P(a|v,q)" — [arXiv](https://arxiv.org/abs/2006.04315)
  - 无处理条件下的分数设为**可学习常数 c**(均匀分布假设),c 只通过 KL 损失学习,使反事实分布的"锐度"与 TE 相当,否则"TIE … dominated by TE or NDE";融合 HM `log(Z/(1+Z)), Z=σ(Zq)σ(Zv)σ(Zk)` 或 SUM `log σ(Zq+Zv+Zk)` — [arXiv](https://arxiv.org/abs/2006.04315)
  - **反事实分布假设消融(Table 4, VQA-CP v2, S-MRL)**:HM:random 31.27 / **prior 46.29** / **uniform 51.27**;SUM:random 27.52 / prior 38.06 / uniform 55.05;作者解释 "uniform distribution assumption guarantees a safe estimation of NDE" — [arXiv](https://arxiv.org/abs/2006.04315)
  - 收益集中在 Yes/No 题(~70%→~90%);VQA v2(无分布偏移)上 CF-VQA(SUM,S-MRL)60.94 vs S-MRL 63.10,有轻微过矫正 — [arXiv](https://arxiv.org/abs/2006.04315)
- **RUBi (Cadene et al., NeurIPS 2019) [FT]**:训练时用 question-only 分支输出 sigmoid mask 逐元素乘到主模型预测上 `f_QM = f(v,q) ⊙ σ(nn_q(e_q(q)))`,降低"不看图就能答对"样本的梯度;**测试时直接移除该分支** — [arXiv](https://arxiv.org/abs/1906.10169)
- **LMH (Clark et al., EMNLP 2019) [FT]**:Learned-Mixin `p̂ = softmax(log p + g(x) log b)`,g = softplus(w·h) 学习"何时信任 bias-only 模型";+H 加熵惩罚 `w·H(softmax(g(x) log b))` 防止 g→0 — [arXiv](https://arxiv.org/abs/1909.03683);CF-VQA 表中 LM+H 在 VQA-CP v2 52.01,但 VQA v2 掉到 56.35(LM 为 63.26),作者称 "over-correct language bias" 约 7 点 — [arXiv](https://arxiv.org/abs/2006.04315)

**D. 对上述对比解码的反证/复现**

- **Mirage (Yin et al., 2025) [FT]**:分析 VCD、ICD、SID;对比分支输入"heavily biased toward 'No'"(COCO Random 贪心 87.1 acc/39.2% Yes;VCD 对比输入 76.7/28.2%;SID 对比输入 79.0/23.6%),对比解码主要是抵消该 No 偏置,使 Yes 比例上升(VCD 88.6/46.4%Yes) — [arXiv HTML](https://arxiv.org/html/2504.10020)
  - **伪基线**:提示 "Answer Yes if possible"(PBA)、P(Yes)−P(No) 小于阈值就强制 Yes(OLM)——COCO Adversarial:VCD 81.9,PBA 83.7,OLM 83.6;"methods not designed for hallucination mitigation produce results similar to CD" — [arXiv HTML](https://arxiv.org/html/2504.10020)
  - 单独使用可信约束(采样)就有提升:LLaVA-1.5-7B Random 83.8→85.4,因为约束把弱候选删掉相当于贪心 — [arXiv HTML](https://arxiv.org/html/2504.10020)
  - 非 POPE 任务退化:MME-P 贪心 1475.1 vs VCD 1389.9 vs SID 1396.4;CHAIR_S 21.6 vs 25.4 vs 24.9 — [arXiv HTML](https://arxiv.org/html/2504.10020)
  - 提出评测准则:考虑解码策略影响;**避免单向修改**(只往一个方向推答案会制造虚假增益);**同时报告"纠正错误"与"破坏原本正确答案"** — [arXiv HTML](https://arxiv.org/html/2504.10020)

### Inferences
- 推断:R1 = "答案级、只在选中答案 o* 上做 PMI 符号检验、阈值 0、α→∞(纯 PMI)"。在 MLLM 文献里**纯 PMI 几乎没人用**:VCD/ICD/SID/VDD 用 α=1(即 2·logit_E − logit_prior,保留一份原分布),CAD 用 α=0.5、CMI-VLD λ=0.5、PAI 甚至只用 γ=1.1(减 0.1 份先验)。product-of-experts 形式 `p_E · (p_E/p_Q)^α` 的意义就是"原分布 + 部分 PMI",α 有限时先验强且证据也强的答案不会被否决。R1 的 16–20% 假弃答与此一致:它等价于 α=∞。
- 推断:可信约束 β 和 M3ID 的"只在 top 概率 < α 时修正"是两种**置信门控**。在 R1 中 o* 本来就是 argmax,β 约束不起作用;对应的可移植形式是"只有当 p′_E(o*) 不够大 / 对第二名的 margin 小时才允许 PMI 否决",或"否决要求 PMI ≤ −δ"(δ>0)。Mirage 显示约束本身就贡献了大量增益,这一点值得做消融。
- 推断:CF-VQA 的"uniform > prior"消融对 R1 很有启发:R1 的 p_Q 是"真实的仅问题前向"(对应 prior 假设),CF-VQA 发现用真实先验作反事实分支明显差于均匀假设 + 学一个温度 c 使锐度匹配。可移植的无标签版本:对 p′_Q 做温度缩放/向均匀分布收缩(如用熵匹配让 p′_Q 的锐度不超过 p′_E),再做比较。
- 推断:几乎所有"训练无关"方法的 α/β 都是**沿用惯例或在有标注开发集上调的**(M3ID 用 CHAIR 网格搜索;Reliable VQA 用验证集选阈值),"无阈值、无标签"并不是该领域的常态;R1 的无阈值可以作为卖点,但必须配上 Mirage 式的伪基线对照。

### Gaps
- 未读到 VCD 官方代码中 α/β 的实际默认值与论文是否一致(只读了论文)。
- ICD+VCD 组合的具体总分只在图中,未提取到数字。
- CMI-VLD 的 λ sweep 数值只在图里,未取到。
- EventHallusion/TCD 只看到搜索摘要,公式与权重未核实。

---

## Q2. 用"有图 vs 无图"差异来**检测幻觉或决定弃答/拒答**的方法;无标签时阈值怎么定;失败在哪里

### Takeaway
**没有找到**任何训练无关、在带弃答选项的多选题上用"有证据 vs 仅问题"的答案级 PMI 直接触发弃答的已发表方法——这正是 R1 的空白;最接近的是 2026 年的 BICR(用黑图视图做对比排序损失,但要训练 probe)、VSI(模糊相关区域后输出是否变化,检测 AUROC 0.85–0.87,仅摘要)、以及 PVI(V-information,"有输入 vs 空输入"的点式差)。关键的反面证据很一致:**"不依赖图像 ≠ 答错"**——PVI 原文明说负 PVI 的样本仍可能答对,BICR 发现图像不变样本"大多数是正确答案",Holtzman 的 PMI_DC 在"先验上就很可能"的 ARC-Easy 答案上失败;阈值方面,有监督的工作都是在验证集上选阈值(Reliable VQA 优化 Φ_c)。

### Cited Findings

**A. 用有/无输入差作为样本级信号**

- **PVI (Ethayarajh et al., ICML 2022) [FT]**:`PVI(x→y) = −log₂ g[∅](y) + log₂ g′[x](y)`,g[∅] 为空输入(空串)模型,g′ 为看到输入的模型;"PVI is to V-information what PMI is to Shannon information" — [arXiv](https://arxiv.org/abs/2110.08420)
  - **关键原话**:"A negative PVI simply means that the model is better off predicting the majority class than considering X … Examples with negative PVI can still be predicted correctly, as long as g′ places most of the probability mass on the correct label." — [arXiv](https://arxiv.org/abs/2110.08420)
  - 预测开始出错的 PVI 点跨数据集相似(**PVI ≈ 0.5 bits**,不是 0);SNLI/MultiNLI/CoLA 上答对与答错样本平均 PVI 差 3.03/2.87/2.45 bits;低 PVI 样本可用来找错标 — [arXiv](https://arxiv.org/abs/2110.08420)
  - 注意:PVI 的 g 与 g′ 是**分别微调**过的模型(带/不带输入),不是同一个模型去掉输入 — [arXiv](https://arxiv.org/abs/2110.08420)
- **BICR "Grounded or Guessing?" (2026) [FT-partial]**:同一冻结 LVLM 跑两次——原图与等尺寸纯黑图(问题不变),取末层最后 prompt token 隐状态;训练轻量 probe,排序损失 `Σ_i ReLU(γ − (p̂_i^base − p̂_i^blank))·y_i / (Σ y_i + ε)`,**只对正确样本施加"有图置信应高于黑图"约束**,因为错误答案没有明确方向 — [arXiv HTML](https://arxiv.org/html/2605.10893)
  - 结果(5 个 LVLM 平均):ECE 7.09、AUROC 78.63,vs 最强基线 P(I Know) ECE 9.31、AUROC 76.57;去掉排序损失 AUROC −3.32 — [arXiv HTML](https://arxiv.org/html/2605.10893)
  - "**Image-invariant does not mean wrong**"——图像不变子集大多是正确答案;POPE 上增益最小,因为幻觉检测"learnable largely from language priors" — [arXiv HTML](https://arxiv.org/html/2605.10893)
  - 我读到的部分没有"原始有/无图差值"作为训练无关基线的结果(VCD 对比在未读的附录 K) — [arXiv HTML](https://arxiv.org/html/2605.10893)
- **Visual Insensitivity Gap / VSI (Zhang, 2026-09) [ABS]**:模糊问题相关区域,看下一 token 分布是否变化;40%–97% 样本"不敏感";作检测信号时在有能力模型的多选推理上 AUROC 0.85–0.87,但**在事实类问题上不如 softmax 置信**,作者建议作为"conditional ensemble component";不敏感样本上视觉塔线性 probe 能以 0.72–0.79 区分扰动/干净图,但 argmax 只在 2%–11% 样本上改变 — [arXiv abs](https://arxiv.org/abs/2609.00868)
- **LPFV / Self-Validation (2026) [FT]**:不是无图对比,而是"去掉已生成上下文"的验证提示 `c_j = p_θ(o_j | v, x_e)`;物体幻觉检测 AUROC 0.85 vs 原 in-caption 概率 0.69(LLaVA-1.5-7B,COCO);阈值 α∈{0,0.005,0.01} 人工设,用于过滤句子而非弃答 — [arXiv HTML](https://arxiv.org/html/2601.22451)
- **PAS (CVPR 2026) [SNIP]**:以"图像与预测物体在 prelim tokens 条件下的互信息"为幻觉信号,用注意力近似,无额外前向 — [arXiv](https://arxiv.org/abs/2511.11502)
- **UniVRSE(VASE 的后续)[FT,但 ID 存疑]**:对原图与加噪图各采 M=10 个回答,语义聚类后 `P_dis = σ((1+λ)P(c|x_v,x_q) − λP(c|x_v′,x_q))`,λ=1,取熵为分数;阈值 τ 在验证集选;RAD-VQA AUC 76.25 vs 语义熵 71.87 — 页面 [arXiv HTML 2503.20504](https://arxiv.org/html/2503.20504)(注意:我请求的 ID 本应是 VASE,但返回内容是同组后续 UniVRSE,需人工核对 ID)
- **VIG, Visual Information Gain (Lee & Hwang, ICML 2026) [SNIP]**:有/无(模糊图代替无图)视觉条件下回答困惑度的对数比,可分解到 token;**用于训练数据/token 选择,不用于推理时检测** — [arXiv HTML](https://arxiv.org/html/2602.17186)
- **MMStar 的 Multi-modal Gain / Leakage (Chen et al., NeurIPS 2024) [SNIP,二手]**:MG = S_v − S_wv(有图分 − 无图分),ML = max(0, S_wv − S_t);动机例子 GeminiPro 在 MMMU 无图得 42.7%(另一来源写 42.9%,数值有出入)— [alphaXiv benchmark 页](https://www.alphaxiv.org/benchmarks/shanghai-ai-laboratory/mmstar);[arXiv](https://arxiv.org/abs/2403.20330)
- **"See or Guess?" (2026-06) [FT-partial]**:6 个开源 VLM 去图后准确率塌到 1–9%,有图 40–69%,"image's contribution gap" 16–45 点;**未测试任何有/无图对比式的弃答或 logit 相减** — [arXiv HTML](https://arxiv.org/html/2606.10400)

**B. 弃答 / 不可答基准(多选 + 弃答选项)**

- **MM-UPD (Miyai et al.) [FT-partial]**:三类不可解问题 AAD(正确答案不在选项中)、IASD(选项与问题无关)、IVQD(图文不匹配);三种设置 Base / Option(加 "None of the above" 或 "The image and question are irrelevant.")/ Instruction;主指标 **Dual accuracy**(标准题与其配对 UPD 题都对才算对) — [arXiv HTML](https://arxiv.org/html/2403.20331)
  - 加弃答选项不一定有用:GPT-4o AAD Dual 45.6(Base)→57.8(Option);**LLaVA-NeXT-34B 50.5→29.9(Option)**→55.1(Instruction);LLaVA-1.5-13B 的 AAD Base UPD 准确率接近 0,作者称部分模型"inherently lack the refusal ability" — [arXiv HTML](https://arxiv.org/html/2403.20331)
  - 训练无关基线:零样本 CoT、自反思(追问 True/False,选 False 则撤回答案):LLaVA-OV-7B AAD Dual 4.5→37.9(CoT)/27.6(自反思);**没有测试有/无图对比** — [arXiv HTML](https://arxiv.org/html/2403.20331)
  - 附录 LoRA 指令微调:提高 UPD 数据比例提升 UPD 但标准准确率下降 — [arXiv HTML](https://arxiv.org/html/2403.20331)
- **TUBench (He et al., 2024) [ABS]**:代码/常识/几何/表格四个视觉域,10 种策略构造不可答问题;28 个模型中 Gemini-1.5-Pro 平均判断可答性准确率最高 69.2%,GPT-4o 66.7%;仅基准 — [arXiv abs](https://arxiv.org/abs/2410.04107)
- **UNK-VQA (Guo et al., TPAMI) [ABS]**:对图像或问题做刻意扰动构造不可答,保持接近原分布;摘要称提出"a straightforward method"但未描述 — [arXiv abs](https://arxiv.org/abs/2310.10942)
- **TRAPSBench (2026) [SNIP]**:模型更容易察觉文本层面的不可能,而难以察觉视觉信息缺失;认为需要输出端干预 — [arXiv HTML](https://arxiv.org/html/2608.13167)

**C. 有监督选择性预测(阈值如何定)**

- **Reliable VQA (Whitehead et al., ECCV 2022) [FT]**:选择函数 g 对置信度设阈值 γ;直接对 softmax 设阈值时 SOTA 模型在 1% 风险下只能回答 <7.5% 问题;学习的 Selector(2 层 MLP,输入答案/问题/图像/多模态表示,**回归 VQA 准确率**)把 C@1% 从 6.8% 提到 15.6% — [arXiv](https://arxiv.org/abs/2204.13631)
  - **Effective Reliability** `Φ_c(x) = Acc(x)`(作答且部分正确)/ `−c`(作答且全错)/ `0`(弃答);报告 c=1,10,100;**阈值 γ 在验证集上选使 Φ_c 最大**,并验证阈值迁移到测试集的风险偏差小 — [arXiv](https://arxiv.org/abs/2204.13631)
  - Selector 特征消融(CLIP-ViL):**只用问题 q(单模态)C@1% = 0.02%、Φ100 = 0.94%;只用图像 0.00%**;多模态 r 13.26%;作者结论 "unimodal selection functions fail … reliable VQA is a truly multimodal problem" — [arXiv](https://arxiv.org/abs/2204.13631)

### Inferences
- 推断:R1 的失败模式有三条独立文献证据:(i) PVI 原文"负 PVI 仍可能预测正确,只要 g′ 把大部分概率放在正确标签上";(ii) BICR "image-invariant does not mean wrong",POPE 类题目大多可由先验学会;(iii) Holtzman ARC-Easy(见 Q3)。也就是说"证据没有比问题本身更支持 o*"在可答题上是常态而非异常,尤其当 p′_Q(o*) 本身已高时。PVI 实测的出错拐点在 ≈0.5 bits 而不是 0,也说明"0 阈值"在不同任务上并没有天然意义(推断)。
- 推断:BICR 的设计选择——**只在正确样本上约束有图 > 黑图,不对错误样本施加方向**——在训练无关场景的对应物是:PMI ≤ 0 不应单独成为弃答充分条件,可以把它与"p′_E(o*) 低"或"p′_Q 与 p′_E 都不尖锐"联合(AND),而不是 OR。
- 推断:VSI 摘要的结论(多选推理上有效、事实类不如 softmax 置信,作为条件集成组件)与 R1 的观察相符:有/无证据差更适合作**辅助门控**而不是单独决策器。
- 推断:Reliable VQA 的 Φ_c 与 MM-UPD 的 Dual accuracy 是现成的、审稿人熟悉的评价方式,可用来把 R1 的"假弃答 vs 正确弃答"权衡量化成一个数;"仅问题特征的 selector 几乎完全失败"也说明光靠 p_Q 做弃答判断信息不足。
- 推断:我没找到"训练无关 + 带弃答选项的 MCQ + 答案级 PMI → 弃答"的已有工作,这对 R1 的新颖性有利,但也意味着缺少可直接引用的阈值/归一化先例。

### Gaps
- BICR 附录 K(与 VCD 对比)和附录 L(图像不变子集)未读,原始有/无图差值作为训练无关基线的数值未知。
- VSI 公式只有摘要信息。
- MMStar 的 MG/ML 公式与 42.7% 来自二手页面(alphaXiv),未读原文。
- TUBench、UNK-VQA 的方法与弃答失败分析只有摘要。
- VASE 原文未读(ID 混淆)。
- 未找到在 streaming / 视频 QA 上用"有帧 vs 无帧"做弃答的工作。

---

## Q3. 与 MCQ 相关的校准文献(Zhao CC、Fei DC、Batch Calibration、PMI_DC、Prototypical);为何校准与"以上都不是/弃答"选项相互作用不好

### Takeaway
所有内容无关校准都在估计 `p(y | 无内容)` 再除掉(Zhao:N/A、[MASK]、空串三者平均;Fei DC:随机域内词采样 M=20 次;BC:用测试样本批均值,可流式更新;PMI_DC:用"领域前提"串),但都有已记录的失败:**"内容无关"输入会被模型当成有内容来解读**(BC 的 QNLI 例子:"N/A" 导致偏向 entailment),PMI_DC 在"先验上就很可能正确"的题(ARC-Easy 的 stock answer)上输给原始 LM 打分。NOTA/弃答方面,Kadavath 和 Tam 都发现模型**系统性地不愿选 NOTA**,当 NA 为正确答案时准确率掉 30–50%;没有找到专门研究"内容无关校准 × NOTA 选项"交互的论文——R1 观察到的"Zhao 式校准压垮弃答"可以由 BC 的失败机制解释(推断)。

### Cited Findings

- **Contextual Calibration (Zhao et al., ICML 2021) [FT]**:`q̂ = softmax(W·p̂ + b)`,`W = diag(p̂_cf)⁻¹`, b=0,p̂_cf 为内容无关输入下标签概率(标签集上重归一化);实验中平均 "N/A"、"[MASK]"、空串三种输入;例子:2-shot SST 提示下 "N/A" 被打成 61.8% Positive — [arXiv](https://arxiv.org/abs/2102.09690)
  - 脚注:替代方案 b = −p̂_cf、W = I 在生成任务上更好;作者明确这里的"calibration"不同于统计校准(置信与准确率对齐) — [arXiv](https://arxiv.org/abs/2102.09690)
  - 内容无关输入选择消融(1-shot SST-2 / 0-shot AGNews):未校准 66.5/48.5;N/A 74.2/64.5;"the" 69.1/59.0;"abc" 77.5/57.3;三者集成 79.0/66.5——"The choice of content-free input matters" — [arXiv](https://arxiv.org/abs/2102.09690)
- **Domain-context Calibration (Fei et al., ACL 2023) [FT]**:提出标签偏置分类(vanilla / context / **domain-label bias**);从无标注测试文本的词袋中按平均长度 L 采随机词作"内容无关"文本,重复 M=20 次取平均 `P̄(y|C)`,预测 `argmax P(y|x,C)/P̄(y|C)`;在 domain-label bias 大的任务上 Macro-F1 提升至多 37% — [arXiv](https://arxiv.org/abs/2305.19148)
- **Batch Calibration (Zhou et al., ICLR 2024) [FT]**:指出 CC 与 DC 在 QNLI 等多句任务上失败:模板 'Question: N/A, Sentence: N/A' 会被模型理解为"N/A 蕴含 N/A"而**偏向 entailment**——"relying on content-free tokens … may even introduce additional bias";DC 的随机词与示例可能产生虚假关联 — [arXiv](https://arxiv.org/abs/2309.17249)
  - BC:用一批 M 个**未标注测试样本**的平均输出估计上下文先验 `p(y|C)_j ≈ (1/M)Σ p(y=y_j|x_i,C)`,再除以/在对数空间平移;支持**在线滚动估计** `p^{n+1} = n/(n+1)·p^n + 1/(n+1)·p̂^{n+1}`;零样本、仅推理;也可接在 VLM(CLIP)输出上 — [arXiv](https://arxiv.org/abs/2309.17249)
- **PMI_DC (Holtzman et al., EMNLP 2021) [FT]**:"surface form competition";`PMI_DC = log P(y|x,domain) − log P(y|x_domain)`,x_domain 为短的领域前提(如 "because"、"answer:");多数数据集上优于 LM/AVG/CC — [arXiv](https://arxiv.org/abs/2104.08315)
  - **失败案例**:HellaSwag(AVG 最好)、**ARC Easy(原始 LM 概率最好:题目可由检索/共现基线解,"bias towards a priori likely phrases","stock answers")**、BoolQ(仅用领域前提的 UNC 最好) — [arXiv](https://arxiv.org/abs/2104.08315)
  - 把无条件基线 UNC(只看领域前提)作为"zero-shot 是否真的用了问题信息"的 sanity check — [arXiv](https://arxiv.org/abs/2104.08315)
- **Prototypical Calibration (Han et al., ICLR 2023) [ABS-level grep]**:用 GMM 在输出上估计各类原型簇、再把簇指派给标签,自适应学习决策边界 — [arXiv](https://arxiv.org/abs/2205.10183);BC 认为这类非线性边界易过拟合,线性边界(CC/DC/BC)更稳 — [arXiv](https://arxiv.org/abs/2309.17249)
- **NOTA 研究**
  - **Kadavath et al. 2022 [FT]**:把 MMLU 等多选的最后一个选项替换为 "none of the above" → "reduces accuracy and calibration significantly";"the model seems strongly biased against using this option";"even the 52B model is biased against using the 'none of the above' option and failed to use it with appropriate frequency",20-shot 也无改善;改成 True/False 单独判断时校准良好 — [arXiv](https://arxiv.org/abs/2207.05221)
  - **Tam et al., Findings ACL 2025 [FT]**:28 个 LLM,MMLU;**NA 为正确答案时一致下降 30–50%**(DeepSeek-V3 65.7% vs 基线 90.8%;Gemini 1.5 Pro 60.3% vs 90.1%);NA 作干扰项时与基线排名一致(r=0.98);学科差异大(大学数学下降小,商业伦理 48.1%);NA 为正确答案时多数学科置信下降 — [arXiv](https://arxiv.org/abs/2503.01550)

### Inferences
- 推断:R1 中 Zhao 式校准"压垮弃答"的机制,与 BC 记录的 QNLI 失败同构:把问题换成 "N/A" 之后,"Unable to answer" 对模型而言是**语义上正确的答案**(没有内容→无法回答),所以 p_cf(abstain) 很高,除以它就把弃答概率压到极低。内容无关输入对弃答选项不是"内容无关"的。
- 推断:"全选项(含弃答)PMI 也伤害"可由 Kadavath/Tam 的发现解释:模型对 NOTA/弃答选项的概率与内容选项**不可比**(系统性偏低且在无证据条件下偏高),把它与内容选项放在同一 PMI 归一化里会被两端的偏差放大。R1 目前"在非弃答选项上重归一化"的做法与 Kadavath 的观察(选项相对比较时校准好,绝对判断 NOTA 时差)方向一致,值得在论文里明确引用作为设计理由。
- 推断:BC 的"在线批均值先验"天然适合流式场景:可以用视频流中已出现问题的平均 p′_E 估计**选项字母/位置偏置**(与问题无关的部分),而 R1 的单题 p′_Q 估计**问题特异先验**;两者可分离——前者用 BC 方式去除,后者只作门控而不是硬否决。
- 推断:Holtzman 的 ARC-Easy 失败几乎就是 R1 的假弃答描述:正确答案本身就是"先验上就可能的",除以先验就惩罚了它。PMI_DC 的作者把这种情况作为"失败案例"写进论文,R1 也可这么诚实地处理并给出分析(例如按 p′_Q(o*) 分桶的假弃答率)。

### Gaps
- 没找到专门研究"内容无关/批校准 与 NOTA/弃答选项交互"的论文;上面关于"压垮弃答"的机制是推断。
- Prototypical calibration 只做了摘要级 grep,未读实验。
- 未读 Wang et al. 2024 / Xu et al. 2024 关于 NOTA 替换正确答案的结果(只在 Tam 和他人引用中出现)。

---

## Q4. 这些论文如何包装该想法(命名模块、理论依据、贡献点)以及展示哪些消融

### Takeaway
包装分三种理论口径:**(1) PMI / 互信息**(M3ID、CMI-VLD、CAD、PMI_DC、PVI)、**(2) 因果推断 TE/NDE/TIE**(CF-VQA;RUBi/LMH 被统一为 NIE 特例)、**(3) 经验性"先验放大"叙事**(VCD 的视觉不确定性、PAI 的 text inertia、ICD 的指令扰动、AGLA 的 attention deficiency);典型贡献三段式是"现象分析 → 训练无关模块 → 多基准实验"。标准消融是 α/β/γ 扫描、对比分支构造方式(噪声步数、无图 vs 噪声图、保留哪些 token)、反事实分布假设(random/prior/uniform)、去掉可信约束;而近年审稿口径(Mirage)要求加上**伪基线、纠正/破坏双向统计、解码策略控制**。

### Cited Findings

- **命名 + 理论口径**
  - M3ID:"Multi-Modal Mutual-Information Decoding",fading-memory 模型推出修正项,并声明等价于最大化视觉输入与文本 token 的 PMI;另有 PDM(prompt dependency measure)作为现象指标 — [arXiv HTML](https://arxiv.org/html/2403.14003)
  - CMI-VLD:把目标写成条件 PMI 并称 VCD 为其特例;双层优化(视觉 token 净化 + 文本解码) — [arXiv HTML](https://arxiv.org/html/2505.19678)
  - CAD:product-of-experts(原分布 × PMI^α),类比 classifier-free guidance — [arXiv](https://arxiv.org/abs/2305.14739)
  - PMI_DC:"surface form competition" 作为问题命名,"domain conditional PMI" 作为解法,利用 COPA 数据集结构去除表面形式竞争的实验作为机制证据 — [arXiv](https://arxiv.org/abs/2104.08315)
  - PVI:V-usable information 的点式版本,PMI 的类比;用于数据集难度与错标发现 — [arXiv](https://arxiv.org/abs/2110.08420)
  - CF-VQA:因果图 + 反事实记号,"cause-effect look";把 RUBi/Learned-Mixin 统一成用 NIE(而非 TIE)推理的特例 — [arXiv](https://arxiv.org/abs/2006.04315)
  - VCD:三条贡献 = 视觉不确定性如何影响幻觉的分析 + VCD 方法 + 不需训练/外部工具的实验 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922)
  - PAI:"text inertia" 现象命名;IBD:"image-biased decoding";AGLA:"attention deficiency";SID:"Context and Text-aware Token Selection (CT2S)" — [PAI](https://arxiv.org/html/2407.21771);[IBD](https://arxiv.org/html/2402.18476);[AGLA](https://arxiv.org/html/2406.12718);[SID](https://arxiv.org/html/2408.02032)
  - VDD 论文:贡献包括"MLLM 输出被 LLM 先验驱动"的分析、Post-Hoc Debias(MCQ)与 VDD(开放生成)两个模块、以及"解码配置对评测影响很大"的评测公平性论点 — [arXiv HTML](https://arxiv.org/html/2403.05262)
- **典型消融**
  - 对比强度/截断:VCD α∈{0.25,…,1}、β∈{0,…,0.5}、噪声步 T∈{200,…,999} — [ar5iv](https://ar5iv.labs.arxiv.org/html/2311.16922);PAI γ∈{1.1,…,2.0} — [arXiv HTML](https://arxiv.org/html/2407.21771);M3ID α×λ 网格与"去掉放大项"(CHAIRi 5.8→6.4) — [arXiv HTML](https://arxiv.org/html/2403.14003);AGLA α 0.5→3.0、β 0.01→1.0 — [arXiv HTML](https://arxiv.org/html/2406.12718)
  - 组件消融:IBD 去 DA/Prompt/APC/I_sim/I_con 全面变差 — [arXiv HTML](https://arxiv.org/html/2402.18476);SID 去掉可信约束后 CD 方法低于常规解码 — [arXiv HTML](https://arxiv.org/html/2408.02032)
  - 对比/先验分支构造:VDD 比较 None/Unk/Noise/全零/全一 — [arXiv HTML](https://arxiv.org/html/2403.05262);Zhao 比较 9 种内容无关串 — [arXiv](https://arxiv.org/abs/2102.09690);DC 的 M(采样次数)与无标注语料大小(50 条即可) — [arXiv](https://arxiv.org/abs/2305.19148);CF-VQA random/prior/uniform 假设 — [arXiv](https://arxiv.org/abs/2006.04315)
  - 过矫正检查:CF-VQA 与 LMH 同时报告 VQA-CP v2(偏移)与 VQA v2(无偏移)以检查是否"over-correct" — [arXiv](https://arxiv.org/abs/2006.04315)
  - 可答/不可答拆分:MM-UPD 报告 Standard / UPD / Dual 三个准确率,并跨 Base/Option/Instruction 设置 — [arXiv HTML](https://arxiv.org/html/2403.20331);Reliable VQA 报告 C@R、AUC、Φ_1/Φ_10/Φ_100 与验证阈值→测试的风险偏差 — [arXiv](https://arxiv.org/abs/2204.13631)
  - 伪基线与双向统计(Mirage 准则):PBA、OLM、单独可信约束;要求报告被破坏的原本正确答案 — [arXiv HTML](https://arxiv.org/html/2504.10020)

### Inferences
- 推断:R1 最自然的理论口径是 **PMI / 对数 Bayes 因子**:`log p′_E(o*) − log p′_Q(o*)` 就是"证据对 o* 的点互信息"(M3ID/CAD/PVI 的同一量);也可以用 CF-VQA 语言写成"TIE 的符号检验"(E 分支 = TE 侧,Q 分支 = NDE 侧),这能直接引用"uniform vs prior"消融来说明为何要对 Q 分支做收缩/温度。
- 推断:按领域惯例,R1 论文应至少包含:(a) 有/无 R1;(b) 门控/阈值 δ 或强度 α 的扫描(哪怕默认无阈值,也要展示 0 附近的稳健性);(c) 先验分支构造的替换(纯文本 Q vs 黑帧 vs 打乱帧 vs 内容无关 N/A——后者即已观察到的失败);(d) 可答/不可答拆分 + Dual accuracy 或 Φ_c;(e) Mirage 式伪基线:同弃答率的随机弃答、"不确定就选 Unable" 提示、p′_E(o*) 置信阈值弃答;(f) 纠正(错答→弃答)vs 破坏(对答→弃答)的计数,按 p′_Q(o*) 分桶。
- 推断:可写成贡献点的诚实叙述:"无阈值、无标签的答案级证据检验;并分析其失败模式——先验可解的问题(PVI 负但答对)",这与 PMI_DC、PVI、BICR 的写法一致,审稿人熟悉。

### Gaps
- 没有找到在 streaming video QA(OVO-Bench / StreamingBench)上用有/无帧对比做答案端弃答的论文,无法提供同场景的包装先例。
- 没有找到用"Bayes factor"字样包装 VLM 有/无图比较的论文;上面的 Bayes-factor 口径是推断。
- M3ID 页面最后约 2k 字符、SID 与 MM-UPD 的后 ~25% 页面、BICR 的后 ~60% 页面未读。
