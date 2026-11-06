# 免训练的查询时视觉搜索 / zoom-in / 抗幻觉验证（图像域 + 可迁移到长视频/文档），以及对 StreamKV 重看过去帧的启示

> 调研日期 2026-11-02。范围是免训练方法，需要训练的方法单独标出。场次信息以抓取到的 arXiv/会议页面为准；只有页面写明的才算“已确认”。凡是只来自搜索摘要、没读到原文的数字，都标了“(未核实全文)”。

---

## Q1. 哪些免训练方法在 7B 级模型上报告了增益（基准、幅度）？哪些需要训练？

### Takeaway
图像域的免训练 zoom/搜索方法在“大图里找小目标”这类基准（V*Bench、HR-Bench、TextVQA）上，对弱 7B 模型提升很大：LLaVA-1.5-7B 在 V* 上 +10 到 +35 点。对强模型提升小得多：Qwen2.5-VL-7B / LLaVA-OV-7B 约 +6 到 +15 点。反幻觉解码（VCD/OPERA/M3ID）在 POPE 上只有约 +2 到 +4 点，而且 NeurIPS 2025 的一篇工作指出，这些增益基本是输出分布偏置造成的假象。V*/SEAL、Chain-of-Spot、InSight-o3 都需要训练。

### Cited Findings

**A. 定位 / zoom / 视觉搜索（图像域）**

- **V\* / SEAL，需要训练。** Wu & Xie，arXiv 2023-12（常被引为 CVPR 2024，抓到的 arXiv 页面没写场次）。
  - 两个组件都要训练。VQA LLM（Vicuna-7B）用约 387k 样本做两阶段训练；搜索模型（LLaVA-7B + OWL-ViT）在检测/分割和 VQA 数据上训练 100K 步。
  - V*Bench 共 191 题（属性 115 题、空间 76 题）。总分：SEAL 75.39，GPT-4V 54.97，LLaVA-1.5-7B 48.68，随机 35.99。
  - 平均搜索长度 4.65 步，单个目标约 6.0 s（A100）。
  - 来源：[V* arXiv HTML](https://arxiv.org/html/2312.14135)
- **ZoomEye，免训练、与模型无关的树搜索，EMNLP 2025 Main。**
  - 机制：把图像当成树，子节点是放大后的子区域。用 Yes/No 下一个 token 的概率构造三种置信度：existing（“这块里有 {o} 吗”）、latent（“继续放大能找到吗”）、answering（“当前视野能回答吗”）。answering 置信度 ≥ 0.8（只用局部图）或 ≥ 0.6（全局+局部）时停止搜索。[ZoomEye HTML](https://arxiv.org/html/2411.16044)
  - 增益（基线 → ZoomEye，同一来源）：

    | 模型 | V* | HR-Bench 4K | HR-Bench 8K |
    |---|---|---|---|
    | LLaVA-v1.5-7B | 48.68 → 83.25 | 36.13 → 53.25 | 32.13 → 50.75 |
    | LLaVA-OV-7B | 75.39 → 90.58 | 63.00 → 69.63 | 59.75 → 69.25 |
    | InternVL2.5-8B | 69.11 → 84.82 | 66.00 → 75.13 | 57.38 → 73.63 |
    | Qwen2.5-VL-3B | 76.96 → 89.01 | — | — |

  - 主表没有 Qwen2.5-VL-7B。论文里有算术不一致：LLaVA-OV-7B 的 V* 增益写作 +14.19，实际差值是 15.19。
  - 代价和副作用：
    - 平均搜索步数 5.0–8.2（LLaVA-OV-7B，随子区域数变化）。没有报告前向次数和延迟。每个节点都要多次 Yes/No 询问。
    - MME-RealWorld 上 LLaVA-OV-7B 的 Position 子项 −12.95，作者归因为全图与子图之间的位置关系难以对应。
    - 作者承认该方法不太适用于文档理解。[ZoomEye HTML](https://arxiv.org/html/2411.16044)
- **ViCrop（“MLLMs Know Where to Look”），免训练。** arXiv 2025-02，常被引为 ICLR 2025，抓到的 v1 页面没写场次。
  - 三种定位信号：
    - rel-att：答案对图像的注意力，除以通用指令下的注意力；需要第二次前向。
    - grad-att：注意力乘梯度。
    - pure-grad：输入图像梯度。
  - 选出的裁剪图与原图 token 拼在一起作为输入。
  - LLaVA-1.5-7B 结果：

    | 基准 | 原图 | ViCrop |
    |---|---|---|
    | TextVQA | 47.80 | 56.06（grad-att） |
    | V* | 42.41 | 62.30（rel-att） |
    | POPE | 85.27 | 87.25 |
    | DocVQA | 15.97 | 19.84 |
    | GQA | 60.48 | 60.98 |

  - LLaVA-NeXT-Mistral-7B 用 rel-att：TextVQA 65.17 → 68.65，V* 58.11 → 61.78。
  - 每次回答额外开销约 1–2 s（GPU）。LLaVA-1.5 上原始 0.17 s/token，rel-att 1.16 s，grad-att 0.89 s。
  - 核心发现：模型答错时，注意力也常常落在正确区域，所以失败在感知而不在定位。
  - 来源：[ViCrop HTML](https://arxiv.org/html/2502.17422)
  - 注意：这里 LLaVA-1.5 的 V* 基线是 42.41，SEAL/ZoomEye 报的是 48.68，评测协议不同。
- **DyFo，免训练，CVPR 2025 Highlight。**
  - 用 MCTS 协调 LMM 与视觉专家 Lang-Segment-Anything（检测框 + mask）。
  - Qwen2-VL-7B：
    - V* 72.25 → 81.15（+8.90），空间子项 68.42 → 82.89。
    - POPE-GQA 对抗子集 acc 75.90 → 81.97。
    - POPE-COCO 随机子集 90.07 → 92.13。
  - LLaVA-1.5-7B：V* 48.68 → 59.16。
  - 平均搜索 3.20 步。未报告推理成本。
  - 来源：[DyFo HTML](https://arxiv.org/html/2504.14920)，[DyFo abs](https://arxiv.org/abs/2504.14920)
- **HiDe，免训练，ICML 2026。**
  - 摘要确认 Qwen2.5-VL-7B / InternVL3-8B 在 V*Bench 上达到 92.1 / 91.6，内存比此前的免训练方法少 75%。[HiDe abs](https://arxiv.org/abs/2510.00054)
  - 表格数字（未核实全文，来自 PDF 搜索摘要）：Qwen2.5-VL-7B 基线 V* 79.1、HR4K 71.8、HR8K 67.9 → HiDe 92.1 / 77.5 / 75.4。[HiDe PDF](https://arxiv.org/pdf/2510.00054)
  - 关键论点：裁剪之所以有用，主要是去掉了背景干扰，而不在于放大小物体。方法是用 token 级注意力对齐目标区域，再做保持布局的解耦。[HiDe abs](https://arxiv.org/abs/2510.00054)
- **RAP（Retrieval-Augmented Perception），免训练，arXiv 2025。**
  - 检索相关裁剪块并按空间布局融合；RE-Search 根据模型置信度和检索分数决定保留多少块。
  - LLaVA-v1.5-13B：V* +43%，HR-Bench +19%。摘要没说明是相对还是绝对值。
  - 来源：[RAP abs](https://arxiv.org/abs/2503.01222)
- **VEA（“Seeing but Not Believing”），免训练，ICLR 2026。**
  - 先用约 100 个 TextVQA 样本一次性挑出与证据框 AUROC 最高的 10% 层，再用这些层的注意力生成高亮掩码，把低注意力区域调暗后重新输入模型。
  - 结果：8 个模型平均 EM +5.67（最多 +11.1）；Qwen2.5-VL-7B EM 73.4 → 78.4（TextVQA/DocVQA/SROIE/InfoVQA）。
  - 抽取 LLaVA-NeXT 和 InternVL3.5 的注意力时显存溢出，作者改用 Qwen2.5-VL-7B 为它们生成增强图。
  - 来源：[VEA HTML](https://arxiv.org/html/2510.17771v1)，[ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010230)
- **Chain-of-Spot，需要微调。** 在 LLaVA-1.5 的 665K SFT 数据上训练 1 epoch，更新全部 LLM 参数。7B 上增益很小：VQAv2 +2.2，TextVQA +2.7，POPE +0.5，MMBench +0.1，MME −9.6。[CoS HTML](https://arxiv.org/html/2403.12966)
- **InSight-o3，需要训练。** vSearcher 用 RL 训练，arXiv 2025-12。[InSight-o3 abs](https://arxiv.org/abs/2512.18745)
  - 搜索摘要显示它的表中 Qwen3-VL-8B 在 V* 上基线 86.4、HR-4K 78.9（未核实全文）。[InSight-o3 PDF](https://arxiv.org/pdf/2512.18745)
- **ZoomClick（GUI grounding），免训练，arXiv 2025-12。** Qwen2.5-VL-7B 总体 44.0，此前的免训练方法 RegionFocus 为 32.1（未核实全文）。[ZoomClick](https://arxiv.org/pdf/2512.05941)
- **训练出来的 zoom 工具到底学到了什么（MED，ICML 2026 camera-ready）。**
  - crop-and-zoom 工具 RL 带来的增益主要来自模型内在能力的提升，而不是更会用工具。
  - 用工具修正模型内在错误的进展有限。
  - 工具调用本身会把原本答对的题改错（call-induced errors）。
  - 来源：[MED abs](https://arxiv.org/abs/2602.01334)

**B. 推理时抗幻觉解码**

- **VCD，免训练，arXiv 2023-11，常被引为 CVPR 2024。** 把原图和加噪图的输出分布做对比。
  - POPE-COCO acc（常规 → VCD）：
    - LLaVA-1.5：随机 83.29 → 87.73，热门 81.88 → 85.38，对抗 78.96 → 80.88。
    - Qwen-VL：随机 84.73 → 88.63，对抗 82.26 → 84.26。
  - 每个 token 要算两套 logits，约 2 倍前向。这是由公式推出的，不是论文报告的数字。
  - 来源：[VCD HTML](https://arxiv.org/html/2311.16922)
- **OPERA，免训练，CVPR 2024。** 在 beam search 中对“过度信任少数摘要 token”的模式加惩罚，并带回滚重分配。作者称“几乎免费”，但抓到的页面上没有量化数字。[OPERA abs](https://arxiv.org/abs/2311.17911)
- **M3ID，免训练，CVPR 2024。** 放大与图像互信息高的 token。在 LLaVA-13B 上，描述中的幻觉物体减少 25%，POPE +21%（未说明是相对还是绝对）。M3ID+DPO 变体需要训练。[M3ID abs](https://arxiv.org/abs/2403.14003)
- **MemVR（“Look Twice Before You Answer”），ICML 2025。** 在模型不确定时，于中间层把视觉 token 作为 FFN 的 key-value 记忆再注入一次，作者称不增加时间开销。抓到的页面没有数字。[MemVR abs](https://arxiv.org/abs/2410.03577)
- **反面证据：“The Mirage of Performance Gains”（NeurIPS 2025）。**
  - 对比解码在 POPE 上的增益来自两点：单向、粗暴地调整输出分布，以及 adaptive plausibility constraint 把采样退化成贪心。
  - 作者构造了与幻觉无关的“伪改进方法”，得到相当甚至更高的增益，结论是这些增益“与其声称的目标完全无关”。[NeurIPS 2025 abs](https://papers.nips.cc/paper_files/paper/2025/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)
  - 第三方评述给的例子：在 prompt 里加一句 “Whenever possible, please select Yes” 就能拿到类似增益。[themoonlight 评述](https://www.themoonlight.io/en/review/the-mirage-of-performance-gains-why-contrastive-decoding-fails-to-address-multimodal-hallucination)
  - 另一篇 “Delve into VCD” 认为，整体提升只是因为“修正为对”的样本多于“改错”的样本（搜索摘要）。[2412.06775](https://arxiv.org/pdf/2412.06775)

**C. 长视频中明确以“搜索”为框架的关键帧方法（简述）**

- **VideoAgent。** arXiv 2024-03，抓到的页面没写场次。用 LLM 当 agent 迭代收集信息，零样本结果 EgoSchema 54.1（平均 8.4 帧）、NExT-QA 71.3（平均 8.2 帧）。中枢是大型 LLM，不是 7B 端到端。[VideoAgent abs](https://arxiv.org/abs/2403.10517)
- **T\*，CVPR 2025。**
  - 把时间搜索当作空间搜索来做，带时空自适应 zoom。
  - LV-Haystack 数据集：480 小时视频、15,092 个标注实例，含训练集。
  - 现有 SOTA 时间搜索在 LongVideoBench 子集上的时间 F1 只有 2.1%。
  - 32 帧预算下 LongVideoBench XL：GPT-4o 50.5 → 53.1，LLaVA-OV-72B 56.5 → 62.4。
  - 抓到的页面没写是否免训练。
  - 来源：[T* abs](https://arxiv.org/abs/2504.02259)
- **DAFS，免训练，arXiv 2026-07。**
  - 用 MLLM 选定层的跨模态注意力给“查询–帧”相关性打分，不需要自回归生成；2B 级的选择器就够用。
  - 32 帧预算下，Video-MME 上比均匀采样最多 +6.4，并超过同预算下需要训练的选择器。
  - 来源：[DAFS abs](https://arxiv.org/abs/2607.15689)
- **F2C，免训练。** Qwen2.5-VL-7B 选关键片段并自适应分辨率。相对均匀采样：Video-MME 最多 +8.1，LongVideoBench +5.6，MLVU +10.3。[F2C HTML](https://arxiv.org/html/2510.02262v2)
- **FOCUS，ICLR 2026，免训练。** 把选帧建模成多臂老虎机，只处理不到 2% 的帧。[FOCUS](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a010577f93a8301dada61f0b97cda923-Abstract-Conference.html)
- **TCS，免训练。** 多查询推理加 CLIP 检索，在 Qwen2-VL-7B / MiMo-VL-7B 上最多 +6.9（未核实全文）。[TCS](https://arxiv.org/html/2601.11359v1)

### Inferences
- **基线越强，增益越小。** LLaVA-1.5-7B 在 V* 上 +35（ZoomEye）。LLaVA-OV-7B 只有 +15。Qwen2.5-VL-7B 在 HiDe 下约 +13（基线 79.1，未核实全文）。Qwen3-VL-8B 的基线若确为 86.4，剩余空间更小。
- **我们的瓶颈不是分辨率。** 我们缺的是哪个时刻发生了什么（时间检索），以及动作/顺序信息。V*/HR-Bench 的前提是“目标确定在这张大图里，只是太小”，和我们的设定只部分重合。
- **抗幻觉解码不应作为主手段。** 证据主要来自 POPE 和 LLaVA-1.5 时代的模型，且已被 Mirage 一文系统质疑。
- **方法要求的内部接口不同。** 免训练方法按需求分两类：
  - 需要注意力图：ViCrop、HiDe、VEA、DAFS。这要求 eager attention，不能用 FlashAttention，显存更高；VEA 在两个模型族上就遇到了显存溢出。
  - 只需要 Yes/No 概率：ZoomEye。对接 vLLM 等推理框架最简单。

### Gaps
- 没找到 ZoomEye / ViCrop / DyFo 在 Qwen2.5-VL-7B 或 Qwen3-VL-8B 上的官方数字。HiDe 的 Qwen2.5-VL-7B 基线数字只来自搜索摘要。
- OPERA、MemVR 的具体数字和延迟没有抓到全文。
- T* 是否免训练，摘要页没说清（数据集带训练集）。
- 没找到任何方法在“流式 / 第一人称记忆型”问答（OVO-Bench EPM 一类）上做“重看过去帧 + zoom”的报告。

---

## Q2. 让验证器只对给定 MCQ 选项打分（似然或逐项 yes/no），是否比自由描述更少编造？

### Takeaway
没有找到直接对照实验，在同一模型、同一证据图上比较“逐选项打分”和“自由描述再回答”的编造率。但有三类间接证据：答案空间越小越准；Yes/No 概率在 7B 模型上可用作搜索和对齐信号；带预设前提的问句会强烈诱导编造。三者都支持这样的方向：约束输出空间，用中性陈述，并按选项做相对比较。同时，yes 偏置和 yes/no 指标容易被钻空子，所以必须做空图校准，并且只在边际足够大时才覆盖原答案。

### Cited Findings
- **答案空间越大越不准（ConBench，NeurIPS 2024）。** 判别式问题的解空间越大，答案准确率越低。判别准确率与模型和自身 caption 的一致性强正相关。[ConBench abs](https://arxiv.org/abs/2405.14156)
- **Yes/No 概率是可用的视觉证据信号。**
  - VQAScore（arXiv 2024-04）用 “Does this figure show '{text}'?” 中 “Yes” 的概率衡量图文对齐，在 8 个对齐基准上达到 SOTA。作者指出 CLIPScore 在复杂文本上不可靠，像 bag-of-words。[VQAScore abs](https://arxiv.org/abs/2404.01291)
  - ZoomEye 用同样的 Yes/No 概率在 7B/8B 模型上驱动搜索和停止。[ZoomEye HTML](https://arxiv.org/html/2411.16044)
- **yes 偏置是真实风险（POPE）。** LLaVA、mPLUG-Owl、MultiModal-GPT 在 POPE 上回答 “Yes” 的比例为 95–100%；InstructBLIP 和 MiniGPT-4 为 55–69%。作者称这些模型“极易答 Yes”。[POPE HTML](https://arxiv.org/html/2305.10355)
- **yes/no 类指标的提升可能是假象。** 对比解码在 POPE 上的提升可由单向分布偏移复现。[Mirage NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)
- **带错误前提的问句诱导编造（MAD-Bench）。**
  - 题目包含不存在的物体、错误计数、错误空间关系，共 1,000 题。
  - GPT-4o 82.82%，其他模型只有 9–50%。
  - 在 prompt 里加一段“回答前再想一想”可让准确率最多翻倍，但绝对值仍低。
  - 来源：[MAD-Bench abs](https://arxiv.org/abs/2402.13220)
  - 这与我们看到的 “you took a knife from the rack” 一类编造在机制上一致：问句里的预设被当成了事实。
- **视觉谄媚 / 信念分裂（“To See or To Please”，arXiv 2026-03 预印本）。**
  - 9 个 VLM、9,000 个模型–样本对中，72.9% 的样本出现“内部证据保留着，但解码出幻觉答案”的模式。
  - 在 Qwen-VL 家族内，扩大模型规模会减少语言捷径，但放大视觉谄媚。
  - 用诊断分数做免训练的选择性预测，在 50% 覆盖率下准确率最多 +9.5 点。
  - 来源：[2603.18373 abs](https://arxiv.org/abs/2603.18373)
- **注意力看对了，答案却错了（VEA，ICLR 2026）。** 深层注意力在答错时也稀疏但可靠地落在证据区域上。[VEA HTML](https://arxiv.org/html/2510.17771v1)
- **文本 LLM 的类比（Chain-of-Verification，arXiv 2023-09）。** 把验证拆成独立回答的短问题，可以减少列表型问答、闭卷 MultiSpanQA 和长文本生成中的幻觉。[CoVe abs](https://arxiv.org/abs/2309.11495)

### Inferences
- 我们的失败 (i) 是“问题条件化的自由描述”在编造。这恰好是 MAD-Bench 预设诱导、加上自由生成长链条（M3ID 指出视觉依赖会随生成长度衰减）的组合。逐选项打分去掉了自由生成，只取单个 token 的概率，理论上去掉了“长生成越写越偏”的通道，但去不掉预设诱导。
- **可行的打分形式：**
  1. 把每个选项改写成中性陈述，例如 “In this image, the person is holding a knife.”，不用第二人称、不用 “you took…”。
  2. 计算 Δ_k = logit(Yes | 裁剪图, 陈述_k) − logit(Yes | 空图或其他片段的帧, 陈述_k)，扣掉语言先验和 yes 偏置。
  3. 在选项之间比较 Δ_k 的相对大小，不看绝对的 yes 比例。
  4. 最大的 Δ 和第二大的 Δ 之差（下称 top1−top2 边际）超过阈值时才覆盖原答案。

  这是由 POPE/Mirage/VQAScore 推出的设计，没有文献直接验证。
- 选择性预测能带来增益（To See or To Please 的 +9.5 点 @50% 覆盖率），说明“只在高置信时覆盖”这一机制本身是有依据的。

### Gaps
- 没找到在视频或第一人称记忆题上，直接比较“自由描述 vs 逐选项似然 vs 逐选项 yes/no”编造率的研究。
- 没找到 Qwen2.5-VL-7B / Qwen3-VL-8B 的 yes 偏置量化数据。POPE 的数字来自 2023 年的旧模型。
- 一篇 2025 年论文据称报告了“同一模型在生成式和判别式下约 10% 的判断冲突”，但抓到的摘要页没有这一内容，不予采信。[2502.16842](https://arxiv.org/abs/2502.16842)

---

## Q3. 注意力 / 检测器式的定位，和 CLIP 整图检索相比，找小细节的效果如何？

### Takeaway
唯一的同表对照来自 ViCrop（TextVQA）：模型内部注意力或梯度做定位明显好于 SAM、YOLO、CLIP 外部定位。三种外部工具都只有小幅增益，而且 SAM 最慢。视频选帧也有同样趋势：DAFS 用 MLLM 注意力选帧，超过均匀采样和需要训练的选择器；T* 指出现有时间搜索的时间 F1 只有 2.1%。这和我们“CLIP top-k 帧检索不准”的观察一致。检测器方案（DyFo 用 Lang-SAM）在对象类问题上有效，但依赖外部专家，对动作/事件类问题帮助不明。

### Cited Findings
- **ViCrop 的同表对照（TextVQA，Table 4）：**

  | 模型 | 原图 | SAM | YOLO | CLIP | 内部注意力/梯度 |
  |---|---|---|---|---|---|
  | LLaVA-1.5 | 47.80 | 49.42 | 48.84 | 48.55 | 51.67–56.06 |
  | InstructBLIP | 33.48 | 39.23 | 36.49 | 39.61 | 42.23–45.71 |

  - SAM 最慢，LLaVA-1.5 上需 3.33 s。CLIP 基线是迭代 20 步、向最高分区域逐步裁剪。
  - 来源：[ViCrop HTML](https://arxiv.org/html/2502.17422)
- **ViCrop 对 POPE 和大物体任务影响很小**（LLaVA-1.5 POPE 85.27 → 87.25，GQA 60.48 → 60.98），说明它不会明显伤害整体感知。[ViCrop HTML](https://arxiv.org/html/2502.17422)
- **HiDe：失败主要来自背景干扰，而不是物体太小。** 它用 token 级注意力定位并去除背景。[HiDe abs](https://arxiv.org/abs/2510.00054)
- **DyFo：检测器 + MCTS。** Lang-SAM 加 MCTS 在 Qwen2-VL-7B 上 V* +8.9，空间子项 +14.5。[DyFo HTML](https://arxiv.org/html/2504.14920)
- **RAP：检索裁剪块。** 用检索器挑出相关裁剪块，LLaVA-v1.5-13B 在 V* 上 +43%。摘要没写检索器类型。[RAP abs](https://arxiv.org/abs/2503.01222)
- **CLIP 对复合文本的局限。** CLIPScore 在复杂文本上不可靠，表现像 bag-of-words；生成式 VLM 的 P(Yes) 更好。[VQAScore abs](https://arxiv.org/abs/2404.01291)
- **视频选帧：**
  - DAFS 用 MLLM 注意力打分，Video-MME +6.4（相对均匀采样），并超过需要训练的选择器。[DAFS abs](https://arxiv.org/abs/2607.15689)
  - T* 指出 SOTA 时间搜索的时间 F1 只有 2.1%。[T* abs](https://arxiv.org/abs/2504.02259)
  - TCS 显示，CLIP 检索配合多查询改写仍有最多 +6.9 的增益（未核实全文）。[TCS](https://arxiv.org/html/2601.11359v1)

### Inferences
- **CLIP 不准的原因。** 对我们来说，CLIP 用问题文本检索整帧不准，原因可能有三：
  1. 问题文本是复合句或第一人称动作，CLIP 按 bag-of-words 匹配；
  2. 目标只占画面一小块，整图嵌入被背景淹没（HiDe 的论点）；
  3. 事件持续 16 s，单帧未必包含答案。

  前两点可以用 MLLM 注意力或 P(Yes) 打分改善，第三点没有方法能解决。
- **推荐的分层做法：**
  - 粗选用文本：在 caption 时间线上选 1–2 个片段。这一步便宜，也绕开了 CLIP 的弱点。
  - 细选用 MLLM 注意力：在片段内选帧和区域。
  - CLIP 只作为片段内帧排序的兜底。
- 检测器方案（Lang-SAM）适合 “what object / what color” 类问题，对 “what did you do after X” 类问题基本无效。

### Gaps
- 没找到多帧输入下（例如一次输入 8 帧）Qwen2.5-VL 或 Qwen3-VL 注意力定位的质量评估。ViCrop、HiDe、VEA 都只在单图上验证过。
- DAFS 与 CLIP 类选择器（AKS、BOLT、Q-Frame）的具体对比数字没抓到全文。
- 没找到在第一人称视频中，用注意力定位“手中物体”的评估。

---

## Q4. 针对我们设定的具体免训练方案、计算成本和风险（结合已有的负面结果）

### Takeaway
值得做的最小版本是一个“门控 + 校准的逐选项 yes/no 验证”管线：caption 时间线选片段，MLLM 注意力在片段内定位并裁剪，在裁剪图上对每个选项打分并扣掉空图基线，只在边际足够大时覆盖原答案。每道触发题的额外成本估计约 10 次短前向，延迟约 1–3 s（推断）。但成功前提很脆弱：44% 的题记忆里没有信息，44% 有歧义，这意味着片段选择本身常常会错；动作类问题未必能从单帧裁剪看出来；文献中也没有“逐选项打分可消除编造”的直接证据。建议先做“oracle 片段”上限实验，再决定是否投入。

### Cited Findings（方案各步对应的文献依据）
- **只在高置信时覆盖原答案。** 选择性预测有增益：+9.5 点 @50% 覆盖率。[2603.18373](https://arxiv.org/abs/2603.18373) zoom 工具会把原本答对的题改错，且很少能修正内在失败。[MED](https://arxiv.org/abs/2602.01334)
- **用注意力而非 CLIP 定位**，依据见 Q3。[ViCrop](https://arxiv.org/html/2502.17422)，[HiDe](https://arxiv.org/abs/2510.00054)，[DAFS](https://arxiv.org/abs/2607.15689)
- **把裁剪图而不是整帧放进上下文**，可以减少背景干扰。[HiDe](https://arxiv.org/abs/2510.00054) 这与我们观察到的“整帧放进回答上下文会伤害当前场景感知”方向一致。
- **用 Yes/No 概率做证据打分。** [VQAScore](https://arxiv.org/abs/2404.01291)，[ZoomEye](https://arxiv.org/html/2411.16044)
- **需要校准 yes 偏置。** [POPE](https://arxiv.org/html/2305.10355)，[Mirage](https://papers.nips.cc/paper_files/paper/2025/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)
- **避免带预设的问句。** [MAD-Bench](https://arxiv.org/abs/2402.13220)

### Inferences（具体方案，以下都是推断或设计，没有文献直接验证）

**0. 写入时**
- 每个 16 s 片段存若干张降采样帧，例如 2–4 张，短边约 448 px 的 JPEG，与 caption 一起存。
- 一张 448×448 的帧大约对应：Qwen2.5-VL 256 个视觉 token（28 px 一个 token）；Qwen3-VL 约 196 个（按 patch 16 加 2×2 合并估算）。
- 存储开销可以忽略，1 小时、每片段 4 帧约 900 张。
- 注意：降采样会损失小细节，和 zoom 的初衷有冲突。可以保留较高分辨率原图，只对选中区域裁剪。

**1. 触发门**
- 只对记忆类问题触发，并且只在 reader 原始答案的选项概率边际低时触发。
- 依据：我们第 9–13 轮的结论是 gate 本身不是增益来源，所以门控应以“不伤害”为目标。

**2. 片段选择（文本级）**
- 让 reader 读 caption 时间线后输出最相关的 1–2 个片段编号；或者用选项词在 caption 中的命中来选。
- 有歧义的那 44% 题里，正确选项词和干扰词都出现在 caption 中，正好可以把两边命中的片段都列为候选。

**3. 片段内定位**
- 输入片段的 2–4 帧（低分辨率）加问题加选项，做一次前向，取答案位置对图像 token 的注意力。
- 按 ViCrop 的 rel-att 思路，再用通用指令做一次前向来归一化，或按 VEA 的方法预先挑选“定位层”。
- 选最高分的帧，在该帧上按注意力峰值裁剪，约为原图的 1/4 到 1/2 边长。
- 需要 eager attention。8B 模型、约 1k 视觉 token 的规模通常可行，但要实测。

**4. 校准的逐选项验证**
- 对选项 k = 1..K（OVO-Bench 通常 4 个），用中性陈述 s_k 计算：
  - Δ_k = logit_Yes(裁剪图, s_k) − logit_Yes(空图或无关片段帧, s_k)
- 可以共享图像前缀的 KV cache，所以每个选项只需要很短的续算。

**5. 融合与覆盖**
- 只有 argmax Δ 不等于原答案、并且 top1−top2 边际 > τ 时才覆盖；否则保留原答案。
- 也可以做 logit 融合：log P_reader(k) + λ·Δ_k。
- τ 和 λ 必须在独立的开发子集上确定，避免对测试集调参。

**计算成本（粗估）**
- 每道触发题：定位 2 次前向（各约 1–1.5k token），验证在共享前缀时为 1 次裁剪图前缀 + 1 次空图前缀 + 2K 个短续算，总计约 4–10 次短前向。
- 7B/8B 模型在 A100/H100 级 GPU 上一次 1–2k token 的 prefill 约 0.1–0.3 s，所以每题约 1–3 s。这与 ViCrop 报告的约 1–2 s 开销同一量级。
- 对比：ZoomEye 每题 5–8 个搜索步、每步多次 Yes/No 询问，大约贵 3–10 倍。

**风险评估（结合我们的负面结果）**
1. **片段选错，验证器自信地错。** 44% 的题记忆中没有信息，说明 caption 时间线常常指不到正确片段。文献里的 zoom 方法都假设目标就在给定图里。必须有“弃权即保留原答案”的出口。
2. **动作 / 时序类问题不适合单帧验证。** 例如 “you placed four cards” 涉及计数和动作序列，单帧裁剪可能根本看不出来。V*、HR-Bench、POPE 全是静态物体或属性题，迁移性弱。
3. **预设诱导仍在。** 改成中性陈述后，“The person took a knife” 这类选项本身仍带预设。MAD-Bench 显示开源模型在错误前提下只有 9–50% 准确率；To See or To Please 发现扩大 Qwen-VL 规模反而放大视觉谄媚，所以换用 Qwen3-VL-8B 未必就更安全。
4. **yes 偏置和校准失真。** 空图基线未必能抵消偏置，因为模型对空图可能有特殊反应。可以改用同一视频中无关片段的帧作为基线，更接近真实分布。
5. **强 reader 的余量小。** Qwen2.5-VL-7B 在 V* 上已有约 79，Qwen3-VL-8B 据报约 86（未核实），图像域 zoom 的边际增益会被压缩。我们的增益上限主要取决于片段选择的召回率。
6. **注意力工程成本。** 关闭 FlashAttention 后显存明显增加，VEA 在部分模型族上遇到显存溢出。多帧注意力的质量也没有经过验证。

**建议的决策顺序**
1. **Oracle 实验。** 若 OVO-Bench 提供证据时间戳，直接把真值片段的帧给验证器，看逐选项校准打分在 EPM 触发题上能否超过当前的 35。
   - 若 oracle 下都不能明显超过 35，说明验证器本身做不到，应放弃这条线。
   - 若 oracle 有效、实际流水线无效，说明瓶颈在片段选择，应把精力投到 caption 时间线检索或 DAFS 式注意力选帧上。
2. **同时对比 Qwen3-VL-8B 和 Qwen2.5-VL-7B**，并把“自由描述 vs 逐选项打分”作为显式消融。这样也补上 Q2 里文献缺失的直接证据。
3. **不建议**把 VCD/OPERA/M3ID 当主手段（见 Q1 的 Mirage 证据），最多作为便宜的消融。

### Gaps
- 没有任何文献在流式 / 第一人称记忆问答中验证过上述组合。效果、阈值、成本全部需要我们自己测。
- 延迟估计是按常见 7B prefill 速度推算的，不是实测。
- 不确定 OVO-Bench 是否为 EPM 等题型提供证据片段的时间戳；这决定了 oracle 实验能否直接做。
- 没找到在 Qwen3-VL-8B 上复现 ViCrop、ZoomEye、HiDe 的公开结果。
