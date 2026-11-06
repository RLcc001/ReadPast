# 视频 / 流式视频论文如何设计、论证和呈现答案端弃答（幻觉拒答）模块：R1 重设计参考

> 阅读状态标记：**[全文]** = 通过 arXiv HTML 读了正文（WebFetch 每次最多 100k 字符，部分论文的末尾附录没读到，已在条目中注明）；**[片段]** = 只看到搜索结果摘要或二手描述，没有读全文。
> 注意：WebFetch 返回的是经过小模型摘要后的内容。下文引号里的"原文"大体可靠，但写进论文前请回原文逐字核对。
> 调研日期：2026-11-03。

## Q1. 哪些流式/在线视频 LLM 论文在 OVO-Bench HLD 上大幅提升？由什么机制带来？

### Takeaway
2026 年 HLD 提升最大的几篇（OneStreamer 99.5、StreamScout-R 98.39、SPARE 90.32、FOLIO 84.6）几乎都靠两类机制：（a）**训练**模型去选弃答选项（LoRA/SFT 加 answerability 数据，或 RL 惩罚错误作答）；（b）**答案 prompt 里写死规则**，要求"有 Unable to answer 选项时宁可选它也不要猜"。这些论文都没有把弃答做成有名字、有消融、会衡量误拒代价的模块。OneStreamer、StreamScout、D-HSM 的正文甚至没有解释自己的 HLD 为什么高。OVO-Bench 原文里一个**只看文本的 GPT-4-turbo 在 HLD 上得 70.05，超过所有视频模型**，说明 HLD 很大程度上可以只靠问题文本解决。

### Cited Findings

**OVO-Bench 本身（基准定义）[全文]**
- HLD 的定义只有一句："Ask questions irrelevant to existing video inputs"，归在 Backward Tracing 下。正文没有描述 HLD 的选项怎么构造，也没提 "Unable to answer" 这类选项是否总是正确答案 — [OVO-Bench](https://arxiv.org/html/2501.05510)
- Table 1 的 HLD 列：Human 91.37；**Blind GPT-4-turbo（纯文本）70.05**；Gemini 1.5 Pro 52.64；GPT-4o 48.66；Qwen2-VL-72B 57.53；Qwen2-VL-7B 56.08；LLaVA-OV-7B 21.51；LLaVA-Video-7B 7.53；LongVU 4.84；Flash-VStream 5.91；VideoLLM-online 12.18；Dispider 4.30。论文没有讨论为什么 blind baseline 比所有视频模型都高 — [OVO-Bench](https://arxiv.org/html/2501.05510)
- 官方评测代码中 HLD 用通用 MCQ 模板 `BR_PROMPT_TEMPLATE`（"Question: {} Options: … Respond only with the letter corresponding to your chosen option"），没有专门针对 HLD 的弃答指令 — [OVO-Bench constant.py](https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/constant.py)
- FOLIO 附录 B.1 写 HLD 共 186 题，"deliberately irrelevant to the existing video inputs" — [FOLIO](https://arxiv.org/html/2607.13298)

**SimpleStream（2604.02317）[全文]：HLD 随帧数增加而下降**
- Qwen3-VL-8B 在 2f/4f/8f 下 HLD 为 54.8/52.1/50.5；Qwen2.5-VL-7B 为 42.5/40.3/36.6 — [SimpleStream](https://arxiv.org/html/2604.02317)
- Visual-RAG 消融（Table 4）：HLD 45.7 → 33.3（+V-RAG，Δ −12.4）— [SimpleStream](https://arxiv.org/html/2604.02317)

**FOLIO（2607.13298，USC/Intel，training-free 实体记忆）[全文，两段都读了]**
- HLD：Qwen2.5-VL-7B 44.6 → 66.7；Qwen3-VL-8B 43.6 → **84.6（+41.0）** — [FOLIO](https://arxiv.org/html/2607.13298)
- 正文只有一句说 HLD "benefits from conservative answer calibration"，没有给出定义 — [FOLIO](https://arxiv.org/html/2607.13298)
- 实际机制写在附录 K.3 的 Answer Prompt 里，是硬规则：
  - "Avoid speculation: prefer "Unable to answer" (when available) over guessing."
  - "If "Unable to answer", "cannot determine", "not visible", or similar is among the options …"
  - "Step 2: If memory says "TARGET NOT FOUND" -> pick "Unable to answer" if available."
  - "Mere presence of an object does NOT confirm its location, action, or relation."
  - 禁止 "most likely/probably" 等模糊措辞，用了就倾向选 Unable to answer
  - 另有 Mode B（CONCEPT QUESTION），允许间接证据（"bookshelf -> reader"）
  — [FOLIO 附录 K.3](https://arxiv.org/html/2607.13298)
- 记忆侧有一个"存在性检查"：附录 D.10 写 "Yes/no and hallucination-detection: presence evidence together with a coverage tag"，即在检索结果里标注目标是否出现在记忆中 — [FOLIO](https://arxiv.org/html/2607.13298)
- 消融（Table 5）只有 200 题 Backward 子集的 overall（S 54.5 / S+O 55.5 / FOLIO 56.7），没有 HLD 拆分，也没有去掉 calibration 的版本 — [FOLIO](https://arxiv.org/html/2607.13298)

**SPARE（2610.14671）[全文 0–100k，附录 B.3/C 未读]：训练弃答**
- 机制：先用一个轻量 gate 判断最近帧是否足够，不够就检索历史片段。之后由 LoRA 微调的 Qwen3-VL-8B 根据已有证据作答或弃答。分工的原话是 "The gate decides whether to seek history; the understanding model decides whether the supplied evidence supports an answer." — [SPARE](https://arxiv.org/html/2610.14671)
- 训练目标（Eq. 9）：要么作答，要么弃答。MCQ 的弃答目标是 "the existing abstention-option letter"；开放题的弃答目标是 "Unable to answer."。answerability 标签决定训练目标，但不放进输入 — [SPARE](https://arxiv.org/html/2610.14671)
- 数据：共 173 条（78 MCQ + 95 开放题），其中包括 hard negatives，即"与原问题相似但证据不足以回答"。answerability 由两次 Qwen3.8-27B 判断，两次一致才保留。训练用纯 answer-token CE（Eq. 10），没有 reward — [SPARE](https://arxiv.org/html/2610.14671)
- Gate：阈值 0.5，r_t = I[p_t ≥ 0.5]·I[|M_t|>0]，用 BCE 训练 — [SPARE](https://arxiv.org/html/2610.14671)
- HLD：无 LoRA 时 42.47 → 有 LoRA 时 **90.32**（Real-time 81.91→81.93 基本不变）；SimpleStream（4 帧）52.69；论文原话 "Hallucination detection shows the largest gain, from 52.69% to 90.32%." — [SPARE](https://arxiv.org/html/2610.14671)
- 弃答没有单独命名，写在贡献 1 里："LoRA fine-tuning trains the understanding model to use available evidence and abstain when it remains insufficient." 没有报告误拒率 — [SPARE](https://arxiv.org/html/2610.14671)

**OneStreamer（2610.01762，NJU/PJLab 等，Qwen3-VL-4B）[全文 0–100k]**
- HLD：Qwen3-VL-4B†（作者重测）41.9 → **OneStreamer 99.5**；AURA-8B 58.6；VideoChat3-4B 47.3；Mage-VL-4B 38.7；JoyAI-VL-Interaction 23.1（Table 8）— [OneStreamer](https://arxiv.org/html/2610.01762)
- 正文没有弃答选项、拒答框架、弃答相关的 loss 或 reward，也没有解释为什么 HLD 能到 99.5。训练数据（Table 6）里有一类 "Answerability judgment"：EgoQA+YouTube-Collected 3,523 条，VST-Training-Data 2,992 条，构造方式没写。没有误拒测量 — [OneStreamer](https://arxiv.org/html/2610.01762)

**StreamScout（2609.00291）[全文]：S = LoRA 蒸馏，R = RL**
- HLD（Table 1）：
  - Qwen2.5-VL-7B：44.62 / zero-shot cascade 36.56 / -S 88.17 / -R 89.25
  - Qwen3-VL-8B：43.55 / 63.44 / 84.41 / **98.39**
  - LLaVA-OV-2-8B：20.97 / 73.66 / 84.41 / 90.81
  — [StreamScout](https://arxiv.org/html/2609.00291)
- RL reward（Eq. 8）：正确性项 − λ·(k̂−1)/(K−1)，其中 η=1，λ=0.03。原话 "η penalizes a wrong commitment so that guessing carries negative expected value"。没有对弃答本身的奖励 — [StreamScout](https://arxiv.org/html/2609.00291)
- 没有任何视角能答对的题被丢弃（"Questions with k⋆=∞ admit no valid stop label and are discarded"）— [StreamScout](https://arxiv.org/html/2609.00291)
- 开放题 prompt 明确防止过度拒答："Do NOT refuse or say "cannot be determined" unless the question is truly unanswerable." — [StreamScout 附录 B.2](https://arxiv.org/html/2609.00291)
- 正文没有单独讨论 HLD，也没有 HLD 消融（Table 4 只报 Perception/Backward/Overall）。zero-shot cascade 让 Qwen2.5-VL 的 HLD 下降，论文没有讨论 — [StreamScout](https://arxiv.org/html/2609.00291)

**D-HSM（2608.30294）[全文]**
- HLD（Table 2）：Qwen2.5-VL+D-HSM 4f 72.0 / 8f 62.4；Qwen3-VL+D-HSM 4f 67.2 / 8f 62.4 — [D-HSM](https://arxiv.org/html/2608.30294)
- **Table C2（Qwen2.5-VL-7B）：只用 HSM 文本记忆 91.40；只用最近 4 帧 51.61；D-HSM 4f 72.04** — [D-HSM](https://arxiv.org/html/2608.30294)
- 正文**没有**提到答案端的弃答指令或 prompt 模板。唯一相关的句子 "The VLM is instructed to output NONE when a field has no visible evidence" 针对的是写记忆时的 schema，不是作答 — [D-HSM](https://arxiv.org/html/2608.30294)

**StreamOPD（2608.16320，Wu et al.）[全文 0–100k + abs]**
- 摘要里明确写 OVO-Bench 提升 "excluding its hallucination-detection subtask"，并单独报告 HLD 升到 57.0% — [StreamOPD abs](https://arxiv.org/abs/2608.16320)
- HLD：Qwen3.5-4B 学生 47.9；9B 教师 47.3；+OPD **38.7（下降）**；+ST-CueGate(OPD) 45.7；+ST-CueGate(OPSD，用学生自身冻结副本当教师) 57.0 — [StreamOPD](https://arxiv.org/html/2608.16320)
- 原话："dense teacher supervision strengthens answerable-task competence but can weaken abstention"；"HLD, which scores abstention on unanswerable queries rather than streaming event recall" — [StreamOPD](https://arxiv.org/html/2608.16320)

**其他 2026 流式记忆论文：HLD 提升小，或没有弃答机制 [全文]**
- OASIS（2604.17052）：Qwen2.5-VL 44.62→48.92；Qwen3-VL 43.55→47.31。没有弃答选项，最接近的是 coarse-reasoning gate（"the model is instructed not to guess"），但信息不足时它触发的是检索，不是弃答 — [OASIS](https://arxiv.org/html/2604.17052v1)
- StreamEMS（2608.27881）：TimeChat-Online 11.8 → 21.5，没有弃答机制，正文没有讨论 HLD — [StreamEMS](https://arxiv.org/html/2608.27881)

**同一 baseline 的 HLD 在不同论文里差异极大（协议敏感）**
- Dispider：OVO/SimpleStream/D-HSM/SPARE 报 4.30；OASIS/FOLIO 表里报 45.43 — [OVO-Bench](https://arxiv.org/html/2501.05510)、[OASIS](https://arxiv.org/html/2604.17052v1)
- LLaVA-Video-7B：7.53（OVO）vs 51.53（OASIS）— [OVO-Bench](https://arxiv.org/html/2501.05510)、[OASIS](https://arxiv.org/html/2604.17052v1)
- Qwen3-VL-8B：SPARE 在 2 fps 下测得 11.83；StreamScout/OASIS 复现 43.55；SimpleStream 4f 52.1 — [SPARE](https://arxiv.org/html/2610.14671)、[StreamScout](https://arxiv.org/html/2609.00291)、[SimpleStream](https://arxiv.org/html/2604.02317)
- TimeChat-Online：9.7（D-HSM/SimpleStream/StreamOPD）、6.99（SPARE）、11.8（StreamEMS）— [D-HSM](https://arxiv.org/html/2608.30294)、[SPARE](https://arxiv.org/html/2610.14671)、[StreamEMS](https://arxiv.org/html/2608.27881)

### Inferences
- 目前 HLD 的"大提升"来源可以归成四类：
  - **prompt 规则**：FOLIO K.3；按我们自己的代码检查，D-HSM 也属于这一类，但其论文正文没写
  - **监督训练弃答**：SPARE，以及很可能包括 OneStreamer 的 Answerability judgment 数据
  - **RL 惩罚错误作答**：StreamScout-R
  - **减少视觉上下文**：SimpleStream 帧数越少 HLD 越高；D-HSM 只用文本记忆时 91.4；V-RAG 让 HLD −12.4

  R1 是第五种：**推理期用 with/without-evidence 的对比来校准**。在流式 HLD 文献里没有找到直接先例，这是 R1 可以主张的差异点。
- 帧数越多、视觉 RAG 越多 → HLD 越低，同时 blind 纯文本模型 HLD = 70，两者合起来提示：HLD 主要测的是"模型在看到不相关画面时会不会编造"，而不是记忆能力。这和 R1 的设计逻辑一致：证据并没有比问题本身更支持 o*，就弃答。但这也意味着审稿人会说 R1 只是把模型推向 blind-baseline 的行为。
- baseline HLD 在不同协议下可以从 ~4 变到 ~52，而且多数论文没有统一协议。所以只报"HLD +13–15"说服力弱；同时报"不含 HLD 的 overall"和误拒代价会更稳。SimpleStream、StreamOPD 已经确立了"HLD 单独报"的惯例。

### Gaps
- 没能从一手数据核实 OVO-Bench HLD 的选项格式，以及弃答选项是否恒为正确答案：HF datasets-server 返回 501，README 和 constant.py 里都没有写。"HLD 正确答案恒为弃答选项"目前只依据我们自己的内部分析。
- D-HSM 的弃答指令论文正文没写，本次没有检查其代码仓库。
- OneStreamer 的 "Answerability judgment" 数据构造方式、HLD 99.5 的成因，在已读的前 100k 字符里都没有说明，附录 E.4 之后没读到。
- Flash-VStream、Dispider、StreamBridge、StreamForest、TimeChat-Online、LiveVLM、StreamMem、InfiniPot-V、HERMES、ReKV 没有逐篇读全文。它们的 HLD 数字只来自他人表格（例如 HERMES 37.6、StreamForest 32.3，见 [SimpleStream](https://arxiv.org/html/2604.02317)）。没有证据表明它们讨论过弃答。
- AURA（HLD 58.6）、StreamTTT-4B（60.75，见 SPARE 表）的机制没有查。

## Q2. 用 with/without 视觉证据做对比的 training-free 视频幻觉缓解方法：对比怎么定义、怎么归一化、阈值怎么选、怎么决定弃答？

### Takeaway
视频领域的 training-free 方法几乎都是**扰动视频**（时间下采样、倒放、时间均匀化、加噪），而不是去掉视频，然后在**token logits 层面做 contrastive decoding**，即 (1+α)·z_orig − α·z_neg，用来改变生成，**都不输出弃答**。与 R1 形式最接近的是 **RED**（2610.02976，音视频）：显式计算 log p(w|evidence,q) − log p(w|q) 这个 PMI，question-only 参考由同一个冻结模型给出。另一个是 **StreamOPD 的 ST-CueGate**：Δ = log π(y|q,c) − log π(y|q)，但它是训练权重。两者都没有用这个量来做弃答判定。所以"把 question-only PMI 用作弃答判据"在视频文献里没有找到先例。

### Cited Findings

**EventHallusion / TCD（2409.16597）[全文]**
- 对比输入：把原视频按时间顺序下采样（"chronologically downsample the original video"），保留画面、削弱时间线索。加高斯噪声的版本试过，没有采用 — [EventHallusion](https://arxiv.org/html/2409.16597)
- 公式：z_t = (1+α)·z_t^ori − α·z_t^con；plausibility 约束 τ_t = β·max(z_t^ori)，低于 τ_t 的 token 置为 −∞ — [EventHallusion](https://arxiv.org/html/2409.16597)
- 超参：α 在 0.25–0.5 有效，α=1.0 明显变差；β 在 0.2–0.5 稳定。帧数（原/扰动）：LLaVA-NeXT-Video 32/8，VILA 12/8，VideoChat2 16/4 — [EventHallusion](https://arxiv.org/html/2409.16597)
- 结果（binary yes/no，overall）：LLaVA-NeXT-Video 64.79→67.48，VideoChat2 32.76→34.72，VILA 63.33→66.50，但 VILA 的 "Entire" 子项 53.51→50.88 下降。没有弃答选项，也没有测非幻觉题 — [EventHallusion](https://arxiv.org/html/2409.16597)

**SEASON（2512.04643，NTU）[全文 0–100k]**
- 负样本有两种：空间负样本 vS = 视频表征加高斯噪声；时间负样本 vT = 逐层把帧特征向均值混合，h = (1−β)h' + β·d_l，保留每帧的空间结构 — [SEASON](https://arxiv.org/html/2512.04643v1)
- 公式：p = softmax[(1+α)·logits(vO) − α·(w_S·logits(vS) + w_T·logits(vT))]。逐 token 权重 w_S = D_S/(D_S+D_T)，其中 D 是原视频与负样本在帧注意力分布上的 JSD（层 20–23）。α、β 用 grid search 选，没有阈值 — [SEASON](https://arxiv.org/html/2512.04643v1)
- **副作用检查**：LLaVA-Video-7B 上 TempCompass 69.7→71.1、TVBench 45.2→46.4、VideoMME 53.0→53.4、MVBench 58.4→57.9（下降）— [SEASON](https://arxiv.org/html/2512.04643v1)
- 与 TCD、DINO-HEAL 对比（VidHalluc / VideoHallucer / EventHallusion，LLaVA-OV-7B）：base 70.32/46.40/60.15；TCD 71.35/48.10/68.46；DINO-HEAL 70.21/46.90/60.15；SEASON 75.36/48.50/69.19 — [SEASON](https://arxiv.org/html/2512.04643v1)

**VidHalluc / DINO-HEAL（2412.03735）[全文 0–100k]**
- DINO-HEAL 不做 logit 对比：用 DINOv2 [CLS] 注意力生成显著性图，对视觉编码器特征逐元素重加权，F^reweighted = F ⊙ S^normalized（Eq.12），training-free。平均提升 3.02%，没有测非幻觉题。有 yes-bias 现象：Video-LLaVA、VILA1.5 对所有题都答 "Yes" — [VidHalluc](https://arxiv.org/html/2412.03735)

**MotionCD（MHBench，AAAI 2025）[片段]**
- 用原视频和倒放视频做 "bidirectional motion elimination"，构造一个去掉运动信息的 amateur 模型来做对比；报告最多 +15.1% — [MHBench AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32463)
- 后续论文把它当 baseline，测得延迟 8.05s，backbone 为 3.89s — [2602.00559](https://arxiv.org/pdf/2602.00559)

**Vista-LLaMA（CVPR 2024）[片段]**
- 不是 training-free：EDVT 注意力去掉视觉–文本之间的 RoPE，让每个视觉 token 与文本 token 等距，目标是避免生成变长后视觉影响衰减 — [Vista-LLaMA](https://arxiv.org/html/2312.08870v2)

**VideoHallucer / Self-PEP（2406.16338）[全文 0–100k]**
- Self-PEP 是自我验证式方案，分两步：self-improvement（参考 Woodpecker 抽取视觉知识）+ self-explanation（预测 → 解释 → 根据解释修正预测）。平均 +5.38%；VideoChat2 overall 7.8→23.5，但 PLLaVA 38.1→31.7 下降 — [VideoHallucer](https://arxiv.org/html/2406.16338)

**RED: Relevant Evidence Decoding（2610.02976，音视频）[全文 0–100k]：与 R1 形式最接近**
- 公式：Δ_t^S(w) = log p_t^S(w) − log p_t^Q(w)，其中 S ∈ {AV, A, V}，Q = question-only（z_Q=∅，"All conditions use the same question and generated prefix"），同一个冻结模型。背景式（Eq.2）是条件 PMI 的近似 log p(w|a,v,c_t) − log p(w|c_t) — [RED](https://arxiv.org/html/2610.02976v1)
- 归一化：在整个词表的 next-token 分布上计算（不是只在选项上）。另外用 100 道**无标签**题估计中心化偏移 η，ℓ̃ = ℓ − η；没有强度系数（权重 = 1）— [RED](https://arxiv.org/html/2610.02976v1)
- 组合：用 question-only selector 选出证据类型 ∈ {∅, A, V, A+V, A↔V}，然后 s^RED = ℓ̃^AV + ρ — [RED](https://arxiv.org/html/2610.02976v1)
- 不处理弃答。TPR/FPR 分析：TPR 78.8→84.7，FPR 23.6→19.3。在一般 QA 上也测了副作用：MUSIC-AVQA 64.5→67.1，TTFT 1.6× — [RED](https://arxiv.org/html/2610.02976v1)
- 消融（CMM/AVHBench）：两个组件都不用 84.8/75.2；只用 Q-guided 86.8/76.2；只用 PMI 84.2/75.8；两者都用 87.0/78.6 — [RED](https://arxiv.org/html/2610.02976v1)
- 图像领域的先例：M3ID 惩罚 "log probabilities unconditioned on the image"，即语言先验 — [M3ID（片段）](https://arxiv.org/pdf/2403.14003)

**StreamOPD 的 ST-CueGate（2608.16320）[全文]：with/without cue 的似然比，用作训练权重**
- τ⁺ = log π_τ(y_t|y_<t,q,c)，τ⁻ = log π_τ(y_t|y_<t,q)，Δ_t = τ⁺ − τ⁻。只移除 cue，帧、问题、前缀保持不变 — [StreamOPD](https://arxiv.org/html/2608.16320)
- 后续处理：
  - 长度归一化 g_i = (1/T)ΣΔ_t
  - 组内标准化 g̃ = (g−μ_G)/(σ_G+ε)
  - 权重 w = sg[clip(1+α_g·g̃, 0, 2)]，α_g=0.5（搜索范围 {0.25, 0.5, 1.0}）

  论文没有把 HLD 的提升归因于这个 gate — [StreamOPD](https://arxiv.org/html/2608.16320)

**其他（只见片段）**
- MACD（2602.01740）用 model-aware mask 构造反事实帧，training-free — [MACD](https://arxiv.org/html/2602.01740)
- SSCD（2601.22574）构造负视频特征后做对比解码 — [SSCD](https://arxiv.org/html/2601.22574v1)

### Inferences
- R1 的 log p'_E(o*) − log p'_Q(o*) ≤ 0 本质上就是**在选项集上重新归一化后的条件 PMI**（RED 的 Δ^S 限制到 MCQ 选项上），可以直接把 RED 的 Eq.2/3 和 M3ID 当方法谱系来引用。与 TCD/SEASON/MotionCD 的差别要讲清楚：（1）R1 是**移除**证据（question-only），而不是扰动视频；（2）R1 用它做**离散决策**（answer → abstain），而不是重加权 logits。
- RED 用 100 道无标签题估计中心化偏移，这给了 R1 一个现成且可辩护的阈值方案：把 "≤0" 换成 "≤ η"，η 在**可回答**的无标签题上标定，使误弃答率不超过 x%。这样阈值有原则，也能回应"阈值 0 是任意选的"这一质疑。这是推断，没有论文这样用过。
- ST-CueGate 的组内标准化和截断 clip(1+α g̃, 0, 2)，可以作为"按问题类型分组标准化 PMI"的参考，用来缓解不同题型 PMI 尺度不一的问题。

### Gaps
- MotionCD 原文（公式、α 值）没有读全文，只有二手描述。
- 没有找到任何视频论文把 question-only / without-video 对比**用于弃答判定**。
- ELV-Halluc、HAVEN、ARGUS 的缓解方法都是训练式或纯评测（见 Q3），不属于 training-free 对比。
- "Video-of-Thought" 式的自我验证与弃答之间的关系没有找到专门文献。

## Q3. 这些论文怎么把模块包装成贡献？做了哪些消融（开/关、阈值扫描、对可回答题的影响、误拒）？

### Takeaway
包装方式分三档：（1）**有命名的方法 + 贡献条目**：TCD、MotionCD、DINO-HEAL、SEASON、RED、ST-CueGate、Self-PEP，以及 UVQA 的 "alignment for answerability"。（2）**埋在训练贡献里**：SPARE 把弃答写进贡献 1，但不单独命名。（3）**一句话或干脆不提**：FOLIO 的 "conservative answer calibration"、OASIS 的 "instructed not to guess"，OneStreamer、StreamScout、D-HSM 则不解释 HLD。流式论文几乎没有 HLD 专门消融；training-free 解码类论文常做 α/β 扫描和通用基准副作用检查，但都不测误拒。

### Cited Findings
- FOLIO：HLD 仅作为结果观察出现，calibration 只有一句话，没有命名，也没有消融。另有原话 "HLD is reported separately and is not the sole evidence for memory quality" — [FOLIO](https://arxiv.org/html/2607.13298)
- SPARE：贡献 1 包含弃答训练。消融只能看到 Table 1 中 no-LoRA vs LoRA 两行（HLD 42.47 vs 90.32），无法把弃答和 LoRA 的其他效应分开。Table 2/3 没有 HLD 拆分，也没有误拒率 — [SPARE](https://arxiv.org/html/2610.14671)
- D-HSM：唯一的 HLD 消融是 Table C2（HSM only 91.40 / recent-4f 51.61 / 完整方法 72.04），弃答没有被当作组件 — [D-HSM](https://arxiv.org/html/2608.30294)
- StreamScout：HLD 不单独讨论，Table 4 消融里没有 HLD。prompt 中有防止过度拒答的措辞 — [StreamScout](https://arxiv.org/html/2609.00291)
- OneStreamer：Tables 2/3/16–18 的消融都没有 HLD，也没有弃答数据的消融 — [OneStreamer](https://arxiv.org/html/2610.01762)
- StreamOPD：主表单独给 HLD，消融表（Tables 2–5）报的是 OVO excl. HLD。HLD 的变化只作为观察来讨论（"dense teacher supervision … can weaken abstention"）— [StreamOPD](https://arxiv.org/html/2608.16320)
- EventHallusion/TCD：作为 "Temporal Contrastive Decoding" 命名，有 α（0.25/0.5/1.0）和 β（0.2–0.5）消融，还对比了"加噪"这种扰动方式 — [EventHallusion](https://arxiv.org/html/2409.16597)
- SEASON：命名方法，有 TCD/DINO-HEAL 对比，有 4 个通用视频基准的保持性检查 — [SEASON](https://arxiv.org/html/2512.04643v1)
- RED：3 条贡献（分析 evidence following + 方法 + 实验），有组件消融（Q-guided / PMI），还有 selector 的 matched/mismatched/random/heuristic/oracle 对照 — [RED](https://arxiv.org/html/2610.02976v1)
- UVQA（ICLR 2025）的贡献表述：（1）形式化 alignment for answerability；（2）同时惩罚漏拒和过度拒答的指标；（3）UVQA 数据管线。把 POPE 式存在性提问作为更贵、更弱的替代方案。在 Limitations 里明确承认 "undesired refusals" — [UVQA](https://arxiv.org/html/2507.04976v1)
- 训练式视频幻觉缓解的包装：
  - HAVEN：SRFT + thinking-based DPO（TDPO），+7.65%，bias score −4.5%；TACL 2026 版改成 HGRPO — [HAVEN](https://arxiv.org/abs/2503.19622v1)、[TACL](https://aclanthology.org/2026.tacl-1.116/)
  - ELV-Halluc：位置编码 + DPO，SAH −27.7% — [ELV-Halluc](https://arxiv.org/abs/2508.21496) [片段]
  - ARGUS：只做评测（hallucination + omission），不提出缓解方法 — [ARGUS](https://openaccess.thecvf.com/content/ICCV2025/papers/Rawal_ARGUS_Hallucination_and_Omission_Evaluation_in_Video-LLMs_ICCV_2025_paper.pdf) [片段]

### Inferences
- 如果 R1 要成为正式贡献，推荐的写法是参照 RED/TCD：**命名方法 + 一行公式 + 三类消融**。三类消融是：（a）开/关；（b）阈值 η 扫描，画 HLD 增益 vs 可回答题误弃答率的曲线；（c）对照"固定弃答 prompt"（D-HSM/FOLIO 式）和"只用 blind/question-only 预测"两个 baseline。所有已读的流式论文都没做（b）和误拒测量，这一点本身就是差异化。
- 如果只把 R1 写成一句话的实现细节（FOLIO 式），在 HLD 贡献了 overall 大部分提升的情况下，容易被认为是在隐藏增益来源。StreamOPD/SimpleStream 已经把"单独报 HLD"变成惯例，审稿人会期待看到 excl.-HLD 的数字。

### Gaps
- 没有找到任何流式论文对弃答阈值做扫描。
- SPARE 附录 B.3/C 没读到，可能有额外的 HLD 分析。

## Q4. 有没有视频论文测量弃答对可回答题的代价（过度拒答）？

### Takeaway
**流式视频论文里一篇都没有。** 能直接借用的测量协议主要来自非流式工作：UVQA（ICLR 2025）定义了 **Excessive Refusal = 可回答题中被拒的比例**，配套 Permissiveness、Discretion、Alignment score，并做了 Pareto 分析；VideoHallucer 的 basic/hallucinated **成对准确率**会惩罚对正常题的错误否定；图像领域的 MM-UPD 用 **Dual accuracy**，并且**把弃答选项和指令同样加到可回答题上**，这和我们想做的"给可回答 MCQ 加 Unable to answer 选项"是同一个设计。

### Cited Findings
- UVQA 指标（[UVQA](https://arxiv.org/html/2507.04976v1)）：
  - S_ex-ref = N3/(N1+N2+N3)，即可回答题中拒答的比例，越低越好
  - S_permis = (N7+N8)/(N7+N8+N9)
  - S_disc = (N11+N12)/(N10+N11+N12)
  - S_align = ⅓((1−S_ex-ref) + S_permis + S_disc)
  - 另有 Answerability F1 和 GPT-4 打分

  评测集为 300 道不可回答题（UVQA）+ 300 道可回答题（ActivityNet-QA）。N1..N18 的具体映射在 Figure 2，本次没有读到。
- UVQA 的代价数字（Table 1）：Excessive refusal 在 SFT 下为 Video-LLaVA 0.50、VideoChat2 0.25、VLM-RLAIF 0.10；DPO 下为 0.14、0.09、0.08；未对齐模型为 0。同时 Accuracy 从 ~0.22–0.25 升到 ~0.47–0.54 — [UVQA](https://arxiv.org/html/2507.04976v1)
- UVQA 的 Pareto 分析（附录 A.10，VLM-RLAIF）：横纵轴是可回答 QA 准确率 vs 不可回答准确率；SFT 以可回答题的小幅代价换取不可回答题提升，DPO 更接近 Pareto 前沿 — [UVQA](https://arxiv.org/html/2507.04976v1)
- UVQA 的 prompt-only baseline 用的指令是 "If the question cannot be answered using the video content, state that it is unanswerable and provide a reason."。在开放式生成中几乎无效：Answerability F1 0.00→0.04–0.12，ex-ref 0–0.02，VLM-RLAIF 的 LLM score 2.36→1.66 — [UVQA](https://arxiv.org/html/2507.04976v1)
- VideoHallucer：每道 basic question 与一道 hallucinated question 配对，"marking it as a hit only if both questions are answered correctly"。另有 Yes Pct. Diff（理想 0）和 FP Ratio（理想 0.5）衡量偏置，发现多数模型偏 Yes，VideoChat2 偏 No — [VideoHallucer](https://arxiv.org/html/2406.16338)
- MM-UPD（图像，2403.20331）[全文 0–100k]：
  - 三类问题：AAD（去掉正确选项）、IASD（选项与问题无关）、IVQD（图像与问题无关）
  - 指标：Standard / UPD / **Dual accuracy**（同一对题都对才算对），Dual 是主指标
  - Option 设置加 "None of the above" 或 "The image and question are irrelevant."；Instruction 设置加 "If all the options are incorrect, answer F. None of the above."，**这些选项和指令也加到 standard 题上**以保证公平
  - 提高 UPD 训练数据比例 → UPD 提升、standard 下降

  — [MM-UPD](https://arxiv.org/html/2403.20331)
- RED 报了 TPR/FPR（78.8→84.7 / 23.6→19.3）和一般 QA（MUSIC-AVQA）副作用 — [RED](https://arxiv.org/html/2610.02976v1)
- SEASON 在 4 个通用视频基准上检查了保持性 — [SEASON](https://arxiv.org/html/2512.04643v1)
- SPARE、OneStreamer、StreamScout、FOLIO、D-HSM 均**没有**报告误拒率 — [SPARE](https://arxiv.org/html/2610.14671)、[OneStreamer](https://arxiv.org/html/2610.01762)、[StreamScout](https://arxiv.org/html/2609.00291)
- OVO-S-Bench（EMNLP 2026）在构造时由独立小组交叉审核 answerability 和证据充分性，但没有看到显式的弃答选项 — [OVO-S-Bench](https://arxiv.org/html/2606.03890) [片段]

### Inferences
- 给 R1 的可执行评测方案（推断）：
  - 在 OVO 的 EPM/ASI（以及 StreamingBench 的记忆类 MCQ）上**追加 "Unable to answer" 选项**，参照 MM-UPD 的 Option 设置，然后报 **Excessive Refusal**（UVQA 的 S_ex-ref：可回答题被切到弃答的比例）以及弃答前后的准确率变化
  - 把 HLD 题与可回答题配对，报 **Dual accuracy** 或 S_align
  - 扫描阈值 η，画 HLD 召回 vs 误弃答率的曲线（coverage–risk / Pareto）

  我们估计的 16–20% 误弃答率处在 UVQA 的 SFT 区间（0.10–0.50）内，高于 DPO（0.08–0.14）。照实报告反而能和 SFT/DPO 训练式方法直接对比。
- UVQA 的 prompt-only 在开放式生成中几乎无效，而 D-HSM/FOLIO 式 prompt 在 OVO 的 MCQ 上很有效。推测原因是 MCQ 里有现成弃答选项时，弃答的"成本"是一个字母，而且 HLD 的弃答选项恒为正确答案。这个推断还需要验证。

### Gaps
- 没有找到任何流式基准或流式论文给可回答题加弃答选项来测代价，也没有 coverage–risk 曲线。
- UVQA N1..N18 的完整映射（Figure 2）没有读到。

## Q5. 有没有审稿人可见的讨论（OpenReview 审稿意见/rebuttal）批评 HLD/弃答增益是基准伪影？

### Takeaway
**没有找到 OpenReview 上针对 HLD 增益的审稿意见或 rebuttal**：检索了 openreview.net 和 UVQA 的 OpenReview，都没有拿到审稿文本。但**论文正文里的批评已经形成共识**：SimpleStream 明确写 "HLD is not a memory task"，StreamOPD 跟进单独报告，FOLIO 自己加了免责声明，OVO-Bench 自己的 blind 纯文本 baseline 在 HLD 上得 70.05、超过所有视频模型。审稿人很容易拿这些证据质疑"HLD 增益 = 弃答偏置"。

### Cited Findings
- SimpleStream 第 5.3 节 "HLD is not a memory task."，原话 "We argue that HLD is conceptually misaligned with long-term memory evaluation"、"Placing HLD under Backward Tracing therefore conflates two distinct abilities: memory recall and hallucination robustness"、"We therefore caution against interpreting HLD as a direct measure of long-range memory." 它的 Memory Gain 指标只用 EPM 和 ASI — [SimpleStream](https://arxiv.org/html/2604.02317)
- StreamOPD："we therefore report it separately following SimpleStream"；"HLD tests hallucination robustness rather than episodic event recall"；"it does not remove HLD from the deployment analysis." — [StreamOPD](https://arxiv.org/html/2608.16320)
- FOLIO：HLD 单独报告，"not the sole evidence for memory quality" — [FOLIO](https://arxiv.org/html/2607.13298)
- OVO-Bench：Blind GPT-4-turbo HLD 70.05 > Gemini 1.5 Pro 52.64 > GPT-4o 48.66，论文没有讨论 — [OVO-Bench](https://arxiv.org/html/2501.05510)
- 检索 openreview.net 时，"OVO-Bench hallucination detection" 和 UVQA 审稿意见都没有返回 OpenReview 页面 — [搜索结果参见 OVO-Bench CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Niu_OVO-Bench_How_Far_is_Your_Video-LLMs_from_Real-World_Online_Video_CVPR_2025_paper.pdf)、[UVQA ICLR poster](https://iclr.cc/virtual/2025/poster/29769)

### Inferences
- 预期的审稿攻击点有三个：（1）HLD 的正确答案恒为弃答，所以任何"更爱弃答"的规则都会涨分；（2）blind baseline 已经有 70，R1 用 question-only 分支，等于往 blind 行为靠；（3）同一 backbone 的 HLD 在不同协议下从 ~4 到 ~52，波动很大。
- 对应的防御：
  - 报 excl.-HLD 的 overall（已成惯例）
  - 报误弃答率和 Dual accuracy（Q4）
  - 和"固定弃答 prompt"、"永远选弃答"、"blind 预测"三个下界/对照比较
  - 证明 R1 在可回答题上的翻转是有选择性的，比如被翻转的题原本的正确率明显低于平均

### Gaps
- 没有拿到任何 ICLR/NeurIPS/CVPR 2025–2026 的 OpenReview 审稿文本，包括 UVQA、StreamForest、TimeChat-Online 等，无法确认审稿人是否直接批评过 HLD 增益。需要手动在 OpenReview 按标题检索。
