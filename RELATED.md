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
4. **HLD 是公认的难点**：BT 涨得多的方法主要涨在 HLD，靠训练弃答（SPARE 90、OneStreamer 99.5、StreamScout-R 98），或靠结构化记忆判断"出现过没有"（FOLIO 66.7 / 84.6）。StreamOPD 的主指标直接排除 HLD，并说明 SimpleStream 也把 HLD 单独报告。
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
