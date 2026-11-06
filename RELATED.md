# 基于 / 对比 SimpleStream 的相关工作调研（2026-10-24）

> 来源：Semantic Scholar 上引用 SimpleStream（arXiv 2604.02317）的 36 篇论文（截至 2026-10-24），加网页检索。数字从各论文 arXiv HTML 版摘取，未逐一核对 PDF。
> 统一口径：OVO = (RT + BT) / 2，不含 Forward；论文自报的 "overall" 口径不一，下面的数是按 RT、BT 重新算的。
> 参照：SimpleStream r4 论文值 Qwen2.5-VL-7B 65.13（RT 78.4 / BT 51.9），Qwen3-VL-8B 67.70（81.4 / 54.0）；我们复现的 Qwen2.5 为 64.75，HLD 40.3。

## A. 免训练，在近帧窗口之外按需取历史（和我们最接近）

| 方法 | arXiv | 机制 | 骨干 | OVO (RT/BT) | 相对 SimpleStream | HLD |
|---|---|---|---|---|---|---|
| WRWS | 2609.37345 | SigLIP 检索 + 相似度熵决定检索还是近帧，K 帧预算 | Qwen2.5-7B / Qwen3-8B 等 | 63.86 / 65.37 | +0.04 / +0.32（同表重跑的 SimpleStream） | 未给可靠数 |
| ShallowStream | 2609.02780 | 浅层 KV 建索引；纯文本 few-shot 门（A/B logit 差，阈值在独立集上标定一次）；浅层注意力投票 + max-min 多样性选帧 | Qwen3-8B / LLaVA-OV-7B | 69.5 / 62.2 | **+3.1 / +1.9**（作者重跑的 SimpleStream 66.4 / 60.3） | 49.5（SS 44.1） |
| FOLIO | 2607.13298 | 写入端用 VLM 每 8 秒写实体中心的文本记忆 + 关键帧证据缓存 + 近帧缓冲；查询时混合检索 | Qwen2.5-7B / Qwen3-8B | 67.1（77.0/57.2）/ 75.6（82.0/69.1） | 未直接比较；按论文值约 +2.0 / +7.9 | 66.7 / 84.6 |
| StreamScout（免训练版） | 2609.00291 | 文本时间线 + 逐级升级：近 4 帧 → 均匀回看 8 帧 → 检索 12 帧，模型自己决定是否升级 | Qwen2.5-7B / Qwen3-8B / LLaVA-OV-2 | 62.7 / 68.4 | Qwen2.5 低于 SS；Qwen3 比自身只看近帧 +3.7 | 36.6 / 63.4 |
| FlashBack | 2610.01192 | 冻结 VLM 先判断要不要历史（Native / Recall），Recall 走临时 Side-KV | StreamingVLM-7B / Mage-VL-4B | 57.9 / 65.6 | 相对自身 −1.2 / +0.75 | 下降 |
| SAVEMem | 2605.07897 | 三级记忆 + 伪问题库定显著性 + 近帧锚点门控 | Qwen2.5-7B | 62.7 | 低于 SS（且门控阈值按 RT/BT 类别分别设，用了类别标签） | 37.6 |
| LatentStream | 2609.04131 | 分层记忆 + 测试时用熵奖励优化潜在记忆 token（不改权重） | Qwen2.5-7B | 64.3 | 略低于 SS | 68.3 |
| ReMem | 2610.05940 | 注意力压缩上下文 + 检索 1 个片段 | Qwen2.5-7B / Qwen3-8B | 51.4 / 60.0 | 远低于 SS | 29.6 / 43.4 |
| ObjectStream | 2607.28312 | 从 VLM 特征诱导潜在物体并跨帧追踪作为记忆锚点 | Qwen2.5-7B | 61.4 | 低于 SS | 未报 |
| DeltaStreamer | 2610.19849 | 智能体：记录帧间变化 + 规划 + 回放历史帧（SAM2/Whisper 工具） | Qwen3.6-27B | 70.0 | 未同骨干比较 | 69.9 |
| FrameBlend | 2610.29229 | 查询时选择性重算 KV（ReKV/HERMES/LiveVLM 上的插件） | LLaVA-OV-7B / Qwen2.5-7B | 58.0（HERMES+） | 未在 SS 上测 | 未报 |

## B. 需要训练

| 方法 | arXiv | 训练什么 | 骨干 | OVO (RT/BT) | 相对 SimpleStream | HLD |
|---|---|---|---|---|---|---|
| SelectStream | 2606.16353 | 冻结 VLM，训练潜在记忆图（写入/整合/检索模块） | Qwen2.5-7B / Qwen3-8B | 70.95 / 72.48 | +5.8 / +4.8 | 未报 |
| SPARE | 2610.14671 | 训练"近帧够不够"的门 + 检索 + LoRA 学会弃答 | Qwen3-8B | 77.78 | +10.2 | **90.3**（SS 52.7） |
| StreamScout-S / -R | 2609.00291 | LoRA 蒸馏升级策略 / 再 GRPO | Qwen2.5-7B / Qwen3-8B | 71.6 / 76.2（-R） | 大幅领先 | 89–98 |
| StreamTTT | 2608.13416 | 每层加测试时训练快权重分支存长程历史 | Qwen3-VL-4B | 69.0 | +2.9（SS-4B 66.06） | 60.8 |
| StreamOPD | 2608.16320 | 只做后训练（on-policy 蒸馏），推理仍是近 4 帧 | Qwen3.5-4B | 69.3（ST-CueGate） | 相对自身近帧 +9.6 | 45.7（主指标排除 HLD） |
| MaLoW | 2610.22070 | 超网络把历史写成 LoRA 权重 | Qwen3-VL-4B 等 | 69.3 | 同骨干表中优于近帧窗口 | 61.8 |
| OneStreamer | 2610.01762 | SFT 学生成带时间的字幕记忆 + 近 16 帧 | Qwen3-VL-4B | RT 81.4 | 优于自身近 16 帧 | 99.5 |
| SVMemAgent | 2609.18540 | RL 学与问题无关的在线选帧 | — | — | — | — |

## C. 效率方向（不以精度为主）

CodecSight（2604.06036，编解码信号裁剪 patch + KV 刷新）、FrameMogging（2605.03351，避免重复计算）、DeltaS（2609.27470，混合架构里用线性注意力状态变化做 KV 淘汰）、Mage-VL（2607.24904，codec-native 流式基础模型）。

## D. 新 benchmark（引用 SimpleStream 作基线）

OmniInteract、OVO-S-Bench、Streaming Interventions、StreamArena、APM-Bench、FastBench、EgoProbe、RoboChrono、VSI-Super-Wild。

## 对我们的启示

1. **免训练"检索原始帧 + 门控"这条路已经很挤，而且在 Qwen2.5-VL-7B 上收益接近零**：WRWS +0.04，我们 +0.5（交叉验证）。在 Qwen2.5 上，SAVEMem、LatentStream、StreamScout、ObjectStream、ReMem 都低于 SimpleStream。
2. **免训练里明确超过 SimpleStream 的只有两类做法**：
   - **ShallowStream**：用模型内部信号检索（浅层注意力），门控只看问题文本、阈值离线标定一次，只在 Qwen3-8B / LLaVA-OV 上报告。
   - **FOLIO / StreamScout**：历史以**文本记忆**（实体记录、字幕时间线）形式提供，视觉只保留近帧。文本记忆不稀释对当前画面的感知（OneStreamer 的消融也是这个结论）。FOLIO 是唯一在 Qwen2.5-7B 上明显高于 SimpleStream 的免训练方法（RT/BT 均值约 67.1），但没有和 SimpleStream 直接比较。
3. **"何时取历史"的门控是现在最热的点**：WRWS、ShallowStream、FlashBack、SAVEMem、StreamScout、SPARE 都在做。我们的创新点 2（置信度仲裁）和它们重叠很大。
4. **HLD 是公认的难点**：BT 涨得多的方法主要涨在 HLD，靠训练弃答（SPARE 90、OneStreamer 99.5、StreamScout-R 98），或靠结构化记忆判断"出现过没有"（FOLIO 66.7 / 84.6）。StreamOPD 的主指标直接排除 HLD，并称 "following SimpleStream"；但 SimpleStream 主表的 Avg 其实含 HLD，只在 §5.3 论证 HLD 不是记忆任务、另定义 ER=(EPM+ASI)/2（11-03 核实）。
5. **低分辨率历史帧**（我们的创新点 1）在这批工作里没人做，但单独效果小（4 帧时从 −2.9 改善到 −1.7）。

## 补充（2026-10-24）：和"增量字幕"最接近的工作

**DeltaStreamer（arXiv 2610.19849，Qwen3.6-27B，8×A100，智能体）**，读了论文 HTML 的方法和消融部分：
- 写入：每个采样帧（基础 1 fps，静止时可降到 0.25 fps），VLM 看上一帧、当前帧和此前的记忆文本，写出 O_t = (D_t 结构化场景描述, Δ_t 相对上一帧的变化描述)。
- 有界：记录先进工作记忆，用 BOCPD 切 episode，生成摘要和长期 pin；渲染时把较早的原文按文本预算裁掉，保留摘要和近期观察，被裁的可以召回。
- 工具：记忆图、核验与回放调度、历史帧回放、裁剪、SAM2.1 跟踪、Whisper。
- 消融（OVO 总分，含 Forward）：1F-NoDelta 58.87 → **1F-TextDelta 63.19** → 2F-NoDelta 61.12 → 2F-VisualDelta（完整）66.07。
- OVO：RT 72.23 / BT 67.72 / (RT+BT)/2 69.98。它表里的 SimpleBaseline 是 RT 79.90 / BT 67.40：重型记忆在 RT 上低了 7.7 分。
- **和我们的重叠**：
  - 它的 1F-TextDelta（看记忆文本，只写变化）基本就是我们的 delta16。
  - 它的"按文本预算裁剪 + episode 摘要"覆盖了我们设想的"长度有上限的分层记忆"。
- 差别只在规模和形态：我们是 7B、每 16 秒一段、无工具、插在 SimpleStream 上。这属于工程差异，不足以作为创新点。

**OneStreamer（2610.01762，4B，需要训练）**：Observe 记录（物体、动作、场景、状态变化）+ Summary 记录；PSTL 训练时挑选状态变化 token。ASI 63.5 → 71.6。

**更早的、非流式的类似思路**：
- LLoVi（2312.17235）：用 LLM 汇总字幕以去掉冗余。
- Video ReCap（2402.13250）：递归分层字幕，需要训练。

## 补充（2026-11-01）：校准路由有没有人做过

结论：**"只看问题文本、决定要不要取历史"的路由已有两篇直接相关的工作；"用空问题做上下文校准"来自 Zhao et al. 2021。两者的组合没有检索到，但新意很薄。**

**ShallowStream（2609.02780）的 Query-Logit Gate**，最接近我们：
- 输入只有问题文本。固定的 few-shot 路由提示词，两个单 token 选项：A = 检索历史，B = 只用近帧。
- 判定方式：一次前向，取 logit 差，与阈值比较：ℓ_ret − ℓ_recent ≥ τ_g。
- 阈值按骨干设定：在 **200 个由生成模型合成的问题（带合成路由标签）** 上标定一次，然后冻结。准则是"检索精度的 95% Wilson 下界 ≥ 90%"。Qwen3-VL-8B 上 τ_g = 11.0，LLaVA-OV-7B 上 τ_g = 0.875。
- OVO 上的触发率：BT 48.7%、RT 4.9%（我们 Qwen3 校准后是 63% / 6%）。路由到的是浅层 KV 投票检索出的历史帧。正文没有给出 gate 开/关的 RT/BT 数字。

**FlashBack（2610.01192）**：
- 冻结 VLM 只看问题文本，按"需要哪种时间证据"做结构化分类：一级 5 类（当前、历史、交互、聚合、预测），"当前"类再分 4 类。每类映射到 Native 或 Recall。
- 用多种类别顺序平均 log 概率，**不用阈值**。
- 消融：直接二分类路由（69.20 / 41.67）不如结构化路由（72.80 / 60.83）。

**其他**：
- StreamScout 免训练版是模型看过画面后自评是否升级，不是只看问题；
- WeaveTime（CVPR 2026）的 Past–Current Dynamic Focus Cache 需要训练；
- SAVEMem 的门控阈值按 RT/BT 类别分别设，用到了类别标签；
- SPARE 的门是训练出来的。
- 非视频方向，"何时检索"是成熟课题：Self-RAG、Adaptive-RAG、FLARE、RetrievalQA / TA-ARE。

**我们能主张的差别**（都不强）：
1. 校准不需要任何标注或合成问题集，只用 3 个空问题（N/A、""、[MASK]）。两个骨干的 p_cf 相差 4 个数量级，路由比例却几乎相同（65/7 vs 63/6），说明可以零成本迁移到新骨干。
2. 路由到的是文本时间线而不是原始帧。
3. 有 OVO / SB 两个基准上的诊断：SB 上"当下"题也能从记忆受益，路由放弃了这部分收益。

**必须补的对比**：
- 在我们已记录的 L0/L1 上，离线比较 ShallowStream 式 logit 差门（合成问题集 + Wilson 准则）、FlashBack 式结构化路由、固定阈值，以及我们的空问题校准；
- 2 个骨干 × 2 个基准。每种路由只需对问题文本跑一次前向，几分钟一个。

## 补充（2026-11-01）：同方向免训练论文的写法结构

读了 5 篇的 arXiv HTML，看章节结构和故事怎么讲（部分附录内容来自摘要，未逐字核对）。

| 论文 | 写法 | 方法前的实验 | 主卖点 | 相对最强基线 |
|---|---|---|---|---|
| SimpleStream 2604.02317 | 分析 / 基线型 | 主结果本身就是动机 | "复杂记忆没必要" + 评测建议 | 方法就是基线 |
| ShallowStream 2609.02780 | 观察 → 方法 | §3 Observations：一个逐层检索实验 | 效率（预填充最多 52×）+ 精度约 +1.7–1.9 | 对重跑的 SimpleStream +1.9（LLaVA） |
| WRWS 2609.37345 | 观察 → 方法 | §2 Empirical Observations：两个实验 | 效率（视觉处理时间为 SimpleStream 的 48%），精度写 "competitive" | OVO +0.25（Qwen2.5）/ +1.04（Qwen3），有平局 |
| FlashBack 2610.01192 | 系统型 | 无，只有示意图 | "免训练也有竞争力" | 只和自身骨干比：OVO +1.4 / +2.75 |
| FOLIO 2607.13298 | 系统型 | 无，引言用文字论证 | SOTA，大幅领先 | +4–12，但没和 SimpleStream 直接比 |

共同点：
- 规模都是 2 个骨干 × 2 个基准（OVO + SB-RT），WRWS 额外扫了多个尺度。
- 都没有显著性检验或多次运行。
- 独立 Limitations 节：SimpleStream、FlashBack 有；ShallowStream、FOLIO 没有（FOLIO 写在结论里，WRWS 放附录）。
- 附录都很重：提示词全文、逐任务完整表、失败案例、效率测量协议。

各写法要点：
- SimpleStream：方法只有一行公式，4 条贡献里只有 1 条是方法；正文主体是 §5 Analysis（窗口长度、模型尺度、Visual-RAG、感知-记忆权衡、基准缺陷）和 §6 "Why does a simple baseline win?"；结论提出 "recent-first, history-on-demand" 但没有实现。
- ShallowStream：一个实验（浅层就能检索，代价随深度上升）直接推出整个方法；门的阈值标定放附录 C。
- WRWS：预实验给出逐题 oracle +6–8 分，方法实得 +0.25–1，靠把效率当主卖点、精度写 "comparable" 来撑；方法后有 "Understanding Adaptive Evidence Allocation" 分析节，承认 79.9% 的题和只看近帧完全相同。
- FlashBack：没有预实验，消融在两个诊断子集上做；HLD 两个骨干都下降，在 Limitations 里承认。

## 补充（2026-11-01）：按"不含 HLD"口径重算 FOLIO

FOLIO 只给了 RT、BT 和 HLD，用 BT = (EPM + ASI + HLD) / 3 反推 EPM、ASI 的均值（论文值，未逐项核对；题目集合可能与我们的 1468 题不同）：

| 骨干 | 方法 | RT | EPM、ASI 均值 | HLD | 标准总分 | 不含 HLD |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | SimpleStream r4（复现） | 77.7 | 57.5 | 40.3 | 64.75 | 67.62 |
| Qwen2.5-7B | FOLIO | 77.0 | 52.5 | 66.7 | 67.10 | 64.73 |
| Qwen2.5-7B | 我们（校准路由） | 77.8 | 59.1 | 50.5 | 67.02 | 68.44 |
| Qwen3-8B | SimpleStream r4（复现） | 81.4 | 55.2 | 51.6 | 67.66 | 68.25 |
| Qwen3-8B | FOLIO | 82.0 | 61.4 | 84.6 | 75.55 | 71.67 |
| Qwen3-8B | 我们（校准路由） | 81.7 | 60.8 | 55.9 | 70.39 | 71.19 |

- FOLIO 在 Qwen2.5 上相对 SimpleStream 的领先全部来自 HLD；不含 HLD 时它比 SimpleStream 低 2.9，比我们低 3.7。
- Qwen3 上 FOLIO 的领先从 5.2 缩到 0.5。
- 待办：从 FOLIO 表 1 逐项核对 EPM、ASI；对其他给出逐任务分数的方法（ShallowStream、SelectStream、StreamScout、WRWS、SimpleStream 论文值）同样重算；确认各论文的 OVO 题目数（SimpleStream 论文 1640 题，我们 1468 题）。

## 补充（2026-11-01）：有没有人"不含 HLD"报告 / 做过内容对照

- **SimpleStream §5.3**：论证 HLD "不是记忆任务"，把记忆、幻觉鲁棒性混在 Backward Tracing 里不合理；分析里的记忆指标 ER = (EPM + ASI) / 2 不含 HLD。只是论证，没有对照实验。
- **StreamOPD（2608.16320，需要训练）**：主指标直接用 "OVO-Bench (excl. HLD)"，同时报告含 HLD 的总分和 HLD 原始分；理由是 HLD "scores abstention on unanswerable queries rather than streaming event recall"，并注明 "following SimpleStream"。它的 HLD 是下降（训练削弱了弃答），方向和我们相反。没有记忆内容对照。
- **FOLIO**：正文承认 HLD 的提升 "also benefit from conservative answer calibration"，但主表照常用含 HLD 的总分。
- **没有找到**（两次检索 + 读了 6 篇论文；StreamOPD 补充材料未读到）：
  - 用外来 / 打乱的记忆做对照，证明 HLD 的提升与记忆内容无关；
  - 按不含 HLD 的口径重算已发表方法，说明其领先集中在 HLD。
- 其他领域的类似对照：RAG 里加入随机无关文档反而提升精度（Cuconasu et al., The Power of Noise, SIGIR 2024）；VQA 的语言先验 / 盲基线（Goyal et al. 2017 VQA v2；Agrawal et al. 2018 VQA-CP）。

## 补充（2026-11-02）：ShallowStream 表 1/表 2 逐项数字与"不含 HLD"重算（arXiv 2609.02780 v2，逐字读表）

不含 HLD = (RT + (EPM + ASI) / 2) / 2。‡ = 作者重跑。ShallowStream 全文与参考文献都没有引用 FOLIO。

| OVO | RT | EPM | ASI | HLD | 标准总分 | 不含 HLD |
|---|---|---|---|---|---|---|
| Qwen3-8B SimpleStream‡（他们） | 80.2 | 53.2 | 60.1 | 44.1 | 66.4 | 68.43 |
| Qwen3-8B SimpleStream（我们复现，与官方逐题一致） | 81.36 | 52.19 | 58.11 | 51.61 | 67.66 | 68.25 |
| Qwen3-8B ReKV‡ | 70.8 | 58.6 | 71.6 | 22.6 | 60.8 | 67.95 |
| Qwen3-8B HERMES‡ | 69.1 | 50.5 | 65.5 | 19.9 | 57.2 | 63.55 |
| Qwen3-8B OASIS | 78.1 | 62.0 | 60.1 | 47.3 | 67.7 | 69.58 |
| Qwen3-8B ShallowStream | 80.9 | 52.5 | 72.3 | 49.5 | 69.5 | 71.65 |
| Qwen3-8B FOLIO（BT 反推） | 82.0 | 61.4（均值） | | 84.6 | 75.55 | 71.67 |
| Qwen3-8B 我们 门 (a) / 完整方法 | 81.65 | 51.85 | 69.59 | 55.91 / 70.97 | 70.39 / 72.89 | 71.19 |
| LLaVA-OV SimpleStream‡（他们） | 71.0 | 53.2 | 52.0 | 43.6 | 60.3 | 61.80 |
| LLaVA-OV SimpleStream（我们复现） | 74.73 | 55.56 | 54.73 | 20.43 | 59.15 | 64.94 |
| LLaVA-OV HERMES‡ | 66.3 | 62.0 | 60.8 | 26.9 | 58.1 | 63.85 |
| LLaVA-OV ShallowStream | 75.4 | 50.2 | 50.7 | 46.2 | 62.2 | 62.93 |

SB-RT：Qwen3-8B SimpleStream‡ 76.5、HERMES‡ 76.6、ShallowStream 78.2（我们复现的 SimpleStream 80.28，SimpleStream 论文 80.59，我们门 (a) 81.64）；LLaVA-OV SimpleStream‡ 73.5、CausalMem 74.3、ShallowStream 75.5。

- Qwen3 不含 HLD：FOLIO 71.67 ≈ ShallowStream 71.65 > 我们 71.19；相对各自的 SimpleStream：ShallowStream +3.23，我们 +2.94。
- ShallowStream 在 Qwen3 OVO 上的增益结构和我们几乎一样：ASI +12.2、HLD +5.4、RT +0.7、EPM −0.7。EPM 不涨是两种方法共有的。
- 基线复现差异大：他们的 SimpleStream 在 Qwen3 SB 上比论文低 4.1，在 LLaVA-OV OVO 上 HLD 43.6 对我们 20.4、RT 71.0 对 74.7。绝对分数不可直接比，应以"相对各自复现的 SimpleStream"为主。

## 补充（2026-11-02）：FOLIO 方法细节与逐项数字（arXiv 2607.13298 v1，正文逐字读；附录 F–I 只读到摘要）

**方法**
- 写入：与作答相同的 VLM（Qwen2.5-VL-7B / Qwen3-VL-8B，vLLM），每 8 秒一段；每段自适应取帧（边界帧、中间帧、帧差最大的探测帧，平均 3.83 帧）。输出 JSON：以实体为中心的记录（名字、别名、类别、属性、位置时段、动作链、关系三元组）+ 事件记录，按名字/别名/类别做规则合并，不是只追加。片段长度敏感性：2/4/8/16/32 秒 ≈ 69.9/70.3/71.0/69.5/65.0（图 3，按标签顺序对应）。
- 关键帧缓存：存关键帧及时间戳，与记录关联；只在记录得分都不够时取回（触发率 2.1%）。近帧缓冲：有，帧数未写。
- 读取：问题解析成 8 种题型（含 hallucination-detection），按字段匹配给实体打分取 top-K；用 VLM 做语义扩展（全系统 75.2% 的题触发，与附录 K.2"只在检索为空时调用"矛盾）。**没有"读不读记忆"的门，每题都读。**
- 作答提示（K.3）：严格事实模式、禁用词表（"most likely"、"probably" 等）、JSON 输出证据与理由；"conservative answer calibration" 只在 4.2 节出现一次，没有定义，也没有消融（最接近的是提示里"宁可答 Unable to answer 也不猜"的规则，这是我们的推断）。
- 开销（Qwen3-8B，RTX 6000 Ada）：写入 7.5 次/分钟、5.88 秒/段；每题 2.15 次 VLM 调用、首 token 6.58 秒（OVO）。
- 没有跑 SimpleStream（只在附录说它是重要的健全性基线）；代码未公开。
- OVO 用的正是 9 个子任务、1468 题、512 个视频，与我们的题目集合相同。

**OVO 表 1 逐项（RT 均值 / EPM / ASI / HLD）与对比**

| | RT | EPM | ASI | HLD | 标准总分 | 不含 HLD |
|---|---|---|---|---|---|---|
| Qwen2.5 FOLIO | 77.0 | 50.5 | 54.3 | 66.7 | 67.10 | 64.70 |
| Qwen2.5 我们 门 (a) / +R1 | 77.84 | 53.87 | 64.19 | 50.54 / 63.44 | 67.02 / 69.17 | 68.44 |
| Qwen2.5 SimpleStream（我们复现） | 77.73 | 53.54 | 61.49 | 40.32 | 64.75 | 67.62 |
| Qwen3 FOLIO | 82.0 | 59.0 | 63.6 | 84.6 | 75.55 | 71.65 |
| Qwen3 我们 门 (a) / +R1 | 81.65 | 51.85 | 69.59 | 55.91 / 70.97 | 70.39 / 72.89 | 71.19 |
| Qwen3 SimpleStream（我们复现） | 81.36 | 52.19 | 58.11 | 51.61 | 67.66 | 68.25 |

- FOLIO 的 RT 子任务分布和 SimpleStream 很不一样（OCR 95–97，FPD 62–63），RT 均值则相近。
- StreamingBench：FOLIO 报的是完整基准多轮设置（Realtime 75.3 / 81.2），和我们的 SB-RT 协议可能不同，不直接比。
- 上面 11-01 那张表里反推的 EPM、ASI 均值（52.5 / 61.4）与逐项数字一致。

## 补充（2026-11-02）：D-HSM（Dynamic Hub-and-Spoke Memory，arXiv 2608.30294 v1，EMNLP 2026 Findings）

**方法**（免训练，Qwen2.5-VL-7B / Qwen3-VL-8B 冻结）：
- 写入：同一 VLM 按固定模式给每个历史片段写结构化观察（Objects、People、Actions、OCR Text、Spatial、Event 六个字段，无证据写 NONE）。片段时长、每段帧数未写明。
- 组织：Objects/People 用 bge-small-en-v1.5 嵌入相似度（阈值 τ_link）链接成跨时间的实体 hub；动作、文字、空间、事件作为 spoke 挂在 hub 上；同片段共现边、每个 hub 的"下一个动作"链。
- 预算：最多 20 个历史片段，超过后在全部历史上**近似均匀**选 20 个；离开的片段按支持计数删除其条目。20 最好，25/30 变差。
- 读取：关键词门（now / currently / recent frame 就不检索）；问题与所有 hub/spoke 的嵌入余弦，阈值 θ，top-K ≤ 12，再按"最大相邻分差"动态截断；沿边扩展；问时间的题加时间线。提示 = 渲染后的文字记忆 + 近 4 帧 + 问题。
- 开销（A100）：每片段写观察 1.55 秒；每题作答 0.74 秒；检索 11 毫秒；每片段 1 次、每题 1 次 VLM 调用。
- 动机实验（图 1，无数字）：把历史片段作为视频 token 加入，不如把同样片段转成字幕；历史会干扰当前时刻的问题。**与我们的 O1 结论相同，已先发表。**
- 没有和 SimpleStream、FOLIO、ShallowStream 比较；没有讨论 HLD 的答案校准。

**结果**（OVO 表 2 逐项，4 帧；他们的 Overall 含 Forward，这里按 (RT+BT)/2 重算）：

| | RT | EPM | ASI | HLD | (RT+BT)/2 | 不含 HLD |
|---|---|---|---|---|---|---|
| Qwen2.5 D-HSM | 79.0 | 52.9 | 63.5 | 72.0 | 70.9 | 68.6 |
| Qwen2.5 我们 门 (a) | 77.84 | 53.87 | 64.19 | 50.54 | 67.02 | 68.44 |
| Qwen2.5 SimpleStream（我们复现） | 77.73 | 53.54 | 61.49 | 40.32 | 64.75 | 67.62 |
| Qwen3 D-HSM | 81.6 | 54.6 | 62.2 | 67.2 | 71.45 | 70.0 |
| Qwen3 我们 门 (a) | 81.65 | 51.85 | 69.59 | 55.91 | 70.39 | 71.19 |
| Qwen3 SimpleStream（我们复现） | 81.36 | 52.19 | 58.11 | 51.61 | 67.66 | 68.25 |

- SB 实时：Qwen2.5 82.5、Qwen3 83.1（4 帧）；我们复现的 SimpleStream 79.80 / 80.28（协议可能不同）。
- 拿我们复现的 SimpleStream 作参照（跨协议，只作粗看）：D-HSM 的 BT 增益 Qwen2.5 +11.0，其中 HLD 贡献 10.6；Qwen3 +7.3，其中 HLD 5.2。**BT 增益主要来自 HLD。**
- 消融（Qwen2.5，OVO，含 Forward 的 Overall / BT）：按时间排列的普通字幕 58.97 / 53.91；分字段观察 62.99 / 60.73；实体组织 63.51 / 60.94；完整 66.80 / 62.82。动态截断对固定 top-12：OVO 66.84 对 64.71，SB 82.45 对 79.47，检索 token 减少约 51%。消融没有逐任务数字，不知道"字段化"的收益在 HLD 还是 EPM/ASI。

## 补充（2026-11-02）：CAM（arXiv 2609.06504 v1，2026-09-06，CUHK / Knowin AI / 武汉大学；代码 github.com/Jake-Tian/CAM）

读了 HTML v1 正文的约 92%；附录 E.2 末尾与 E.3 只读到模型摘要，验证器没有给出提示词。

- **方法**：免训练的提示词流程，但所有角色都用 GPT-5.2（另有 Qwen3-VL-flash / Qwen3.5-flash 版本，规模未写）。
  - 写入与问题无关：每 30 秒一段，MLLM 写行为、对白、外观描述，LLM 转成 (主体, 关系, 客体, t) 三元组，构成人物、物体、对话异构图；边积累到阈值后异步写人物和关系摘要（带置信度）。
  - 记忆里不存帧或特征；提问时按 clip id 从原视频重新取帧。
  - 读取用 LangGraph 的规划器 → 执行器 → 验证器，7 个工具（general_search、时间窗检索、物体检索、频次统计、图遍历、watch_video_clip、listen_to_audio）。
  - **"验证器"只判断信息够不够（Complete / Incomplete / Stuck），用来控制循环，不核对证据**；回看的输出没有任何核验，也没有测幻觉。答题提示**禁止弃答**。
- **基准**：M3-Bench-robot / web（开放式问答，GPT-5.2 判分，平均 27–34 分钟）、HippoVlog（选择题）。**没有 OVO-Bench / StreamingBench**，也没有比较 SimpleStream、FOLIO、D-HSM、ShallowStream、ReKV。
- **数字**：CAM (GPT) robot 64.5 / web 69.3 / HippoVlog 76.4；"+23" = robot 64.5 对 M3-Agent 41.4（M3-Agent 被换成 GPT-5.2 底座）。CAM (Qwen) 51.3 / 55.6 / 73.8，web 上低于 WorldMM 58.1（正文说超过所有基线，与表矛盾）。
- **消融**（robot）：去掉图 44.8（−19.7）；只用相似度检索 56.5，加回看 57.8（+1.3，作者说 token 代价显著）；固定间隔摘要 60.3。没有单独去掉验证器或回看的消融。回看占工具调用 16.8%（robot）、3.7%（web）。每题 47.3 秒、11.3K token。
- **与我们的关系**：
  - 方法上不撞车：CAM 没有"由记忆冲突触发、只对选项核对"的证据核验，也没有防编造设计；我们计划的核验与它的验证器是两回事。
  - 支持我们的发现：用 GPT-5.2 回看原视频也只多 +1.3。
  - **对我们"结构不涨"结论的限定**：在 M3-Bench（长视频、以人物和关系为中心、多跳的开放式问题）上，实体图贡献 +19.7。我们和 D-HSM 的结论只在 OVO 记忆题、7–8B 写者上成立，论文里要写清这个范围。

## 补充（2026-11-03）：公开代码情况（用户决定：不对比没有代码的方法）

逐篇查了 arXiv abs、HTML 首页和项目页（很多仓库只在 HTML 页眉或项目页上给出，abs 里没有）。"无"指三处都没找到，或者只写了"代码将发布"。

| 方法 | 是否训练 | 代码 | OVO / SB 评测脚本 | 代码支持的骨干 |
|---|---|---|---|---|
| SimpleStream | 免训练 | 有 | 有 / 有 | Qwen2.5-VL、Qwen3-VL |
| D-HSM | 免训练 | github.com/Oshikaka/D-HSM | 有 / 有 | Qwen2.5-VL-7B、Qwen3-VL-8B |
| ShallowStream | 免训练 | github.com/CURRENTF/ShallowStream（完整方法） | 无 / 无（只有 smoke 推理） | Qwen3-VL-8B、LLaVA-OV |
| SAVEMem | 免训练 | github.com/wuhang03/savemem | 有 / 有 | Qwen2.5-VL |
| ObjectStream | 免训练 | github.com/DMK041218/ObjectStream | 未确认 | Qwen2.5-VL-7B、Qwen3.5-9B |
| DeltaStreamer | 免训练（智能体） | github.com/BBBBchan/DeltaStreamer | 有 / 有 | Qwen3.6-27B + 工具 |
| ReKV / HERMES / InfiniPot-V / LiveVLM | 免训练 | 本地 third_party 已有 | — | — |
| SelectStream | 训练 | github.com/Haonan-Ge/selectstream | 只有通用 eval | Qwen2.5-VL-7B、Qwen3-VL-8B |
| StreamTTT | 训练 | github.com/zeyun-zhong/StreamTTT | 有 / 有（据 README） | Qwen3-VL-4B |
| MaLoW | 训练 | github.com/dragonlzm/MaLoW（推理 + 权重；训练代码未放） | 未确认 | Qwen3-VL-4B、Qwen3.5 |
| StreamOPD | 训练 | github.com/UniX-AI-Lab/StreamOPD | 有 / 有 | Qwen3.5-4B |
| OneStreamer | 训练 | github.com/MCG-NJU/OneStreamer（权重未公开） | 有 / 有 | Qwen3-VL-4B |
| WRWS、FOLIO、StreamScout、FlashBack、LatentStream、ReMem、FrameBlend、SPARE | — | 无 | — | — |

## 补充（2026-11-05）：需要训练的方法具体怎么训（3 个子代理读 arXiv HTML 正文 + 附录；图里的数字读不到；原文抽取在 scratchpad/papers/）

| 方法 | 训练什么（冻结什么） | 写记忆时看问题吗 | 监督信号 | 数据 | 算力 | 训练部分的关键消融 |
|---|---|---|---|---|---|---|
| OneStreamer 2610.01762 | Qwen3-VL-4B 的 LLM + 投影层全量 SFT（视觉冻结）；模型自己写带时间戳的字幕记忆 + 近 16 帧 | 不看 | 模仿老师字幕（Gemini3.1Pro / Seed2.1Pro 写，"对照视频核验"），交叉熵；无 QA 损失、无 RL | OneStreamer-1M（字幕记忆 6 万条）；QA 由 Gemini/Seed 生成、Qwen3.5-397B 核验；**剔除评测集视频（含 OVO、SB）** | 32×H200，1 epoch 1174 步 | 表 2 同一权重去掉 / 保留字幕：ASI 63.5→71.6、EPM 62.0→62.6、RT 80.9→81.4；**表 17：不用字幕数据训练、用自己提示出的字幕作记忆 −0.1（OVO）/ +0.3（SB），用字幕 SFT +2.0 / +0.6** |
| M3-Agent 2508.09736 | 写记忆的 Qwen2.5-Omni-7B 做 SFT；控制器另做 RL | 不看 | 模仿 Gemini-1.5-Pro / GPT-4o 写的记忆（10,952 条） | — | — | 表 6：SFT 写者优于直接提示 Gemini 写 |
| SelectStream 2606.16353 | 骨干冻结；训练潜在记忆图的写入 / 检索 / 校准模块 | 写入不看，检索看 | 答案的自回归损失端到端 + 用证据时间戳的检索损失 + 稀疏损失 | Streamo-Instruct-465K；未提去重 | 8×A100，1 epoch | 表 5（Qwen2.5）：无历史 62.31 → 完整 65.71；随机记忆 token 62.02 |
| MaLoW 2610.22070 | 骨干冻结；超网络（198–321M）把历史写成 rank-8 LoRA | 不看 | QA 损失穿过生成的 LoRA 端到端 | 380,920 QA / 55,867 视频（Qwen3.5-122B 出题、397B 核验）；剔除 OVO 视频 | 4×H200，2800 步 | GitHub 文档（Qwen3.5-9B，Back / RT）：近 2 秒 51.2/75.5；+免训练文本摘要 59.4/78.1；MaLoW 66.4/76.2；两者都加 69.7/77.3 |
| StreamTTT 2608.13416 | 每层加测试时训练的快权重分支（训练哪些参数未写） | 不看 | 答案损失 | 23 万 QA（LLaVA-Video、Streamo 改造等）；未提去重 | 未写 | 表 2：相对冻结的近 4 帧，ER +8.25、RT −0.61 |
| SPARE 2610.14671 | 骨干冻结；门 = 2 层 MLP（1.06M）；LoRA r16 学弃答 | 记忆是固定的 4 秒片段嵌入 | 门：27B 模型判"近帧够不够"（两个提示一致才留）；LoRA：**173 条**样本（95 条弃答） | 1.5 万门标签；声明与 OVO/SB 视频不重合 | 1×H100 | HLD 52.69→90.32，总分增益约 62% 来自 HLD；门比"总是检索"只多 +0.59 |
| StreamScout-S/-R 2609.00291 | LoRA r16 只学"答 / 要更多上下文"的决策 token | 提问后 | -S：自蒸馏（最浅能答对的视图标为答）；-R：GRPO，奖励 = 1[对] − 1[错] − 0.03·代价 | 1.34 万 / 7.8K 题（Video-MME、OVBench、ODV-Bench） | -S 14.5 h 1×A100；-R 约 100 GPU-h | 普通 GRPO 塌缩成"总在第一视图作答"，去掉 std 归一化（Dr.GRPO）解决；对零样本的增益 53–91% 在 HLD |
| SVMemAgent 2609.18540 | 外部 195.6M 选帧策略（16 槽），VideoLLM 冻结 | 不看 | 冷启动伪标签 + GRPO，奖励 = 冻结 InternVL3.5-2B 对正确答案的平均 log-prob；DAAD 压掉"记忆差别大但奖励相同"的组 | Video-R1、LongVILA 子集；未提去重 | 未写 | 无 OVO/SB；Qwen3-VL-8B 平均 63.4，Reservoir 62.2，均匀 63.3；GRPO 比冷启动 +0.8 |
| StreamOPD 2608.16320 | 学生 Qwen3.5-4B（推理只看近 4 帧，无记忆） | — | 在线策略蒸馏：9B 老师（额外看 27B 写的时空提示）的反向 KL 作优势 | 25,118 题（短片段）；未提去重 | 4+4×H200 | RT 70.6→82.6、EPM 47.1→58.9；OPD 削弱弃答（HLD 47.9→38.7） |
| CapRL++ 2606.09393（离线） | Qwen3-VL-4B 视频描述器 RLVR | 不看（离线描述） | **奖励 = 不看画面的 LLM 只读描述答选择题的正确率** + 时间戳格式 + 长度项；题目只保留"看视频能答、不看不能答"的 | 约 2 万道 MCQ（Qwen3-VL-235B / Gemini-3-Flash 出题） | — | Prism 平均 47.5 vs 41.2 |

非视频先例（已核实）：RECOMP（Xu et al., ICLR 2024，摘要器从 GPT-3.5 摘要蒸馏，只保留能提升冻结 LM 下游结果的摘要，都不帮就用空目标）；SCST（Rennie et al., CVPR 2017，CIDEr 奖励）；MemAgent（Yu et al., ICLR 2026，RL 覆写记忆，但写记忆时看问题）；MEM1（Zhou et al., ICLR 2026，PPO，非问题无关）。VideoLLM-online（Chen et al., CVPR 2024）：Ego4D 旁白流用的是原始旁白，对话数据来自 COIN / Ego4D GoalStep 的离线标注。

**对"训练写入端"的启示**：
1. 写文字记忆的监督目前只有"模仿强老师"（OneStreamer、M3-Agent）；用冻结读者的问答结果当奖励训练描述器只有离线的 CapRL++，存帧的 SVMemAgent 用的是同类奖励。**流式 + 文字记忆 + 读者奖励**这一格在这次检索范围内没找到。
2. OneStreamer 表 17 与我们的结论一致：提示出来的字幕不涨，训练过的字幕才涨（+2.0，主要在 ASI，EPM 只 +0.6）。
3. 数据都是大模型生成 QA（10 万–100 万级）+ 核验；OneStreamer、MaLoW、SPARE 声明剔除了 OVO / SB 视频。
4. 两张卡能做的规模是 SPARE / StreamScout 级（1 卡，10–100 GPU-h）；OneStreamer / MaLoW 级（4–32 张 H200，几十万到百万样本）做不了。
5. 已知的坑：GRPO 塌缩（StreamScout，Dr.GRPO 解决）、奖励带语言偏置（SVMemAgent）、蒸馏削弱弃答（StreamOPD）。
6. 训练类方法在 OVO 上的总分增益大部分来自弃答训练（SPARE 173 条样本 HLD +38），比较时要看逐任务分数。
