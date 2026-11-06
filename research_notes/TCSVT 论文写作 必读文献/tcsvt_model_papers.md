# TCSVT 近作调研：可作为 ReadPast 结构与实验范本的论文（2024–2026）

> 调研日期 2026-11-03。检索方式：Crossref API（TCSVT ISSN 1051-8215 期刊过滤 + DOI 元数据）、OpenAlex（TCSVT 源过滤 + 摘要）、arXiv API（comment/journal_ref 含 TCSVT）、WebSearch。dblp 被反爬拦截，Semantic Scholar 被限流（429），两者均未用上。
> 页数由 Crossref 登记的正式页码区间计算（含参考文献与作者简介）；图/表数量来自 arXiv 版本，可能与 TCSVT 正式版略有出入。
> 文中 "已核实" 表示 Crossref 登记的 DOI 前缀为 10.1109/TCSVT、container-title 为 IEEE Transactions on Circuits and Systems for Video Technology。

## 问题 1：哪些 TCSVT 论文与 ReadPast 在主题和"免训练 / 即插即用"上最接近？venue 是否已核实？

### Takeaway
TCSVT 2025–2026 已发表的论文中，最接近 ReadPast 的是两篇免训练长视频论文：**EventLens**（2026，免训练，5 个 benchmark × 7 个 MLLM，backbone 含 Qwen2.5-VL-7B / Qwen3-VL-8B）和 **FDVS**（2025，把长视频转成分层文本"故事"、不做任何微调，与我们的文本时间线思路最像）。另有一批免训练 / 推理时的 MLLM 方法（ResPrune、PTP、AQAH），可参考它们的效率表和"为什么有效"分析怎么写。我**没有找到任何已在 TCSVT 发表的流式（streaming/online）视频 LLM 论文**。这对 ReadPast 来说既可能是一个空位，也意味着找不到同方向的直接范本。

### Cited Findings

**A. 主推范本（免训练 + 长视频/文本记忆，强烈建议精读）**

- **EventLens: Event-Aware Hierarchical Evidence Construction for Long-Video Understanding**。作者 Zhengkun Zhu, Xiao Dong, Yang Xu, Yixiu Liu, Zhilou Yu, Li Liu, Huaxiang Zhang。TCSVT vol.36 no.11, pp.17510–17524（15 页），2026-11，DOI 10.1109/TCSVT.2026.3723106，Xplore https://ieeexplore.ieee.org/document/11653436。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3723106)
  - 摘要原意：长视频受限于"海量视觉内容 vs. 有限上下文窗口"；已有方案"typically require costly training"，免训练检索式方法潜力大，但大多是"flat frame-level retrieval"，且只用原问题做一次匹配。EventLens 由四个部件组成：semantic boundary parser、query-answer co-retriever、query-guided evidence selector、contrastive verdict refiner。在"five long-video understanding benchmarks and seven representative MLLMs"上实验，LVBench 最高 +8.1%、VideoEval-Pro 最高 +9.2% — [OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2026.3723106)
  - GitHub README：自称 "a training-free framework"。benchmark 为 Video-MME、LongVideoBench、LVBench、MLVU、VideoEval-Pro（后者用开放式短答 + LLM judge 评分）；backbone 为 Qwen2.5-VL-7B/32B、Qwen3-VL-8B/32B、LLaVA-Video-7B-Qwen2、InternVL2.5-8B、InternVL3-8B。超参：1 FPS 采样、边界阈值取第 12 百分位、5 帧平滑、最短事件 5 s、Top-16 事件、最终帧预算 64、α=0.4/γ=2.0/ρ=0.8。README 没有写 venue、arXiv 链接或结果数字 — [GitHub](https://github.com/zhengkun332/EventLens)
  - 未找到 arXiv 版本，按标题检索 arXiv 无结果 — [arXiv API 检索](https://export.arxiv.org/api/query?search_query=ti:EventLens)

- **Toward Long Video Understanding via Fine-Detailed Video Story Generation (FDVS)**。作者 Zeng You, Zhiquan Wen, Yaofo Chen, Xin Li, Runhao Zeng, Yaowei Wang, Mingkui Tan。TCSVT vol.35 no.5, pp.4592–4607（16 页），2025-05，DOI 10.1109/TCSVT.2024.3514820，Xplore https://ieeexplore.ieee.org/document/10789221，arXiv 2412.06182。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2024.3514820)；[arXiv](https://arxiv.org/html/2412.06182)
  - 摘要原意：把长视频"interprets ... into detailed textual representations"。用 Bottom-up Video Interpretation（clip → video）加视觉/文本两级 Semantic Redundancy Reduction，得到分层文本表示，"applicable to various tasks without any fine-tuning"，在 3 类任务、8 个数据集上评测 — [OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2024.3514820)

- **Asking Questions to Alleviate Object Hallucination in Large Vision-Language Models (AQAH)**。作者 Chao Bi, Tiantian Dang, Shuhui Wang, Feng Cao, Qingming Huang。TCSVT vol.36 no.3, pp.3497–3512（16 页），2026-03，DOI 10.1109/TCSVT.2025.3618949，Xplore https://ieeexplore.ieee.org/document/11195828。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2025.3618949)
  - 方法：比较多次采样结果来定位幻觉因素，再就不确定信息提问，把答案补进 prompt 后重新生成。摘要原文强调 "does not directly constrain the decoding"，因此不会显著拖慢推理，也不会缩短输出；与其他去幻觉方法相比"fewer training/inference time costs"。注意它需要训练一个小的问题生成模型，不是完全免训练 — [OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2025.3618949)

**B. 免训练 / 即插即用的 MLLM 推理时方法（参考效率与分析的写法；不是视频任务）**

- **ResPrune: Text-Conditioned Subspace Reconstruction for Visual Token Pruning in Large Vision-Language Models**。作者 Xu Li, Yi Zheng, Yuxuan Liang, Zhe Liu, Xiaolei Chen, Haotian Chen, Rui Zhu, Xiangyang Xue。TCSVT Early Access 2026，DOI 10.1109/TCSVT.2026.3738719，Xplore https://ieeexplore.ieee.org/document/11717542，arXiv 2603.21105。**已核实**（Crossref 页码 1-1，即尚未分配卷期）；arXiv 页面本身没写 venue — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3738719)；[arXiv](https://arxiv.org/html/2603.21105)
- **Pyramid Token Pruning for High-Resolution Large Vision-Language Models via Region, Token, and Instruction-Guided Importance (PTP)**。作者 Yuxuan Liang, Xu Li 等（同组）。TCSVT vol.36 no.6, pp.8372–8384（13 页），2026，DOI 10.1109/TCSVT.2026.3663658，Xplore https://ieeexplore.ieee.org/document/11393550，arXiv 2509.15704。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3663658)；[arXiv](https://arxiv.org/html/2509.15704)
- **A Glimpse to Compress: Dynamic Visual Token Pruning for Large Vision-Language Models (GlimpsePrune)**。作者 Quan-Sheng Zeng, …, Ming-Ming Cheng, Qibin Hou。TCSVT vol.36 no.10, pp.15539–15552（14 页），2026-10，DOI 10.1109/TCSVT.2026.3702147，Xplore https://ieeexplore.ieee.org/document/11557376，arXiv 2508.01548（comment 为 "15 pages, 10 figures"）。**已核实**；正文未读 — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3702147)；[arXiv API](https://export.arxiv.org/api/query?id_list=2508.01548)
- **DiViCo: Disentangled Visual Token Compression for Efficient Large Vision-Language Model**。作者 Xin Wang, Zirui Pan, Hong Chen, Wenwu Zhu。TCSVT vol.36 no.2, pp.1392–1405（14 页），DOI 10.1109/TCSVT.2025.3567138。**已核实**。这是带训练的模块（变分建模 + KL），效率写法可参考："LLaVA-v1.5-7b equipped with DiViCo is able to reduce 67.7% FLOPs and save 51.7% time while maintaining 95.6% of the accuracy" — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2025.3567138)；[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2025.3567138)

**C. TCSVT 上的视频 LLM / 长视频 QA 论文（多为需训练的方法，主要参考结构与叙事）**

- **VideoExpert: Augmented LLM for Temporal-Sensitive Video Understanding**。作者 Henghao Zhao, Ge-Peng Ji, Rui Yan, Huan Xiong, Zechao Li。TCSVT vol.36 no.5, pp.5893–5908（16 页），2026-05，DOI 10.1109/TCSVT.2026.3653742，Xplore https://ieeexplore.ieee.org/document/11348931，arXiv 2504.07519。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3653742)；[arXiv](https://arxiv.org/html/2504.07519)
- **Reducing Spatio-Temporal Redundancy for Efficient Long-Term Video Understanding via Prototype-Guided Memory-Augmented Large Multimodal Models (PGM)**。作者 Jisheng Dang, …, Qi Tian, Tat-Seng Chua。TCSVT Early Access 2026，DOI 10.1109/TCSVT.2026.3743402，Xplore https://ieeexplore.ieee.org/document/11737651。**已核实**。属于记忆增强 + 提效（摘要称"improving accuracy and reducing GPU memory usage"），用到训练问题的统计先验，不是免训练；未找到 arXiv — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3743402)；[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2026.3743402)
- **Spatial–Temporal Clue Reasoning Chain for Long Video Question Answering**。作者 Haibo Gong, Chenggang Yan, Jiehua Zhang, Yaoqi Sun, Yuhan Gao, Liang Li。TCSVT vol.36 no.3, pp.2757–2770（14 页），DOI 10.1109/TCSVT.2025.3609441，Xplore https://ieeexplore.ieee.org/document/11162559。**已核实**。属于训练式方法（asynchronous clue selector + self-excitation loss），未找到 arXiv — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2025.3609441)；[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2025.3609441)
- **Temporal-Guided Mixture-of-Experts for Zero-Shot Video Question Answering (T-MoENet)**。作者 Yixin Qin, Lei Zhao, Lianli Gao, Haonan Zhang, Pengpeng Zeng, Heng Tao Shen。TCSVT vol.35 no.9, pp.9003–9016（14 页），2025，DOI 10.1109/TCSVT.2025.3556422，Xplore https://ieeexplore.ieee.org/document/10946169。**已核实**。零样本 VideoQA，TGIF-FrameQA +5.6%、MSRVTT-QA +2.3%，代码 github.com/qyx1121/T-MoENet — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2025.3556422)；[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2025.3556422)
- **Prompting Video-Language Foundation Models With Domain-Specific Fine-Grained Heuristics for Video Question Answering (HeurVidQA)**。作者 Ting Yu, Kunhao Fu, Shuhui Wang, Qingming Huang, Jun Yu。TCSVT vol.35 no.2, pp.1615–1630（16 页），2025-02，DOI 10.1109/TCSVT.2024.3475510，Xplore https://ieeexplore.ieee.org/document/10706923，arXiv 2410.09380。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2024.3475510)；[arXiv](https://arxiv.org/html/2410.09380)
- **Event-Centric Augmentation for Video Question Answering (EcA)**。作者 Xize Wu, Zheng Wang, Jiasong Wu, Lei Zhu, Heng Tao Shen。TCSVT vol.36 no.11, pp.16572–16586（15 页），DOI 10.1109/TCSVT.2026.3716450。**已核实**。自称"plug-and-play, event-centric augmentation framework"，但即插即用指的是训练阶段的数据增强，不是推理时插件 — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3716450)；[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2026.3716450)
- **Visual Evidence-Aware for Object Hallucinations Rectification in LLM-Based Video Captioning**。作者 Ye Wang, Jiancheng Zhou, Qun Liu, Feng Hu, Guoyin Wang。TCSVT vol.36 no.3, pp.2842–2853（12 页），DOI 10.1109/TCSVT.2025.3613993。**已核实** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2025.3613993)
- **Video Understanding With Large Language Models: A Survey**。作者 Yunlong Tang, Jing Bi, Siting Xu 等。DOI 10.1109/TCSVT.2025.3566695，arXiv 2312.17432（comment 为 "Accepted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)"）。**已核实**。Related Work 里引用它，是向 TCSVT 读者表明"了解本刊语境"的自然方式 — [Crossref 检索](https://api.crossref.org/journals/1051-8215/works?query.bibliographic=video+large+language+model&filter=from-pub-date:2024-01-01)；[arXiv API](https://export.arxiv.org/api/query?id_list=2312.17432)

**D. 流式视频 LLM 在 TCSVT 的覆盖情况**

- 在 TCSVT 范围内做了多组检索：Crossref 期刊过滤（streaming / online video / long video / training-free 等关键词，每组最多 600 条，合并去重 1544 条）和 OpenAlex 标题检索（20 个关键词）。2024 年以来所有标题含 "streaming" 的 TCSVT 论文都属于网络传输、QoE、事件相机或流式感知，没有一篇是流式视频 LLM / 流式 VideoQA — [Crossref 标题检索](https://api.crossref.org/journals/1051-8215/works?query.title=streaming&filter=from-pub-date:2025-01-01&rows=200)；[OpenAlex](https://api.openalex.org/works?filter=primary_location.source.issn:1051-8215,from_publication_date:2024-01-01,title.search:streaming)
- 两个 benchmark 本身的 venue：OVO-Bench 是 CVPR 2025，StreamingBench 被引为 ICASSP 2026。都不是 TCSVT — [WebSearch 结果汇总，引自 arXiv 2604.02317 等参考文献表](https://arxiv.org/html/2604.02317v1)
- 网上多篇流式记忆方法的 arXiv 预印本（StreamEMS 2608.27881、CurveStream 2603.19571、VideoScan 2503.09387 等）均未在 TCSVT 登记 DOI。CurveStream 在 Crossref 登记为 LNCS 会议论文，FluxMem 为 CVPR 2026 — [arXiv API](https://export.arxiv.org/api/query?id_list=2603.02096,2603.19571,2608.27881,2503.09387)；[Crossref](https://api.crossref.org/works?query.bibliographic=CurveStream)

**E. 非 TCSVT 补充（IEEE TMM 2026，主题相近，正文未读）**

- GCAgent: Long-Video Understanding via Schematic and Narrative Episodic Memory，DOI 10.1109/TMM.2026.3724344 — [Crossref](https://api.crossref.org/works/10.1109/tmm.2026.3724344)
- Rewind-Augmented Generation for Long Video Understanding，DOI 10.1109/TMM.2026.3679059；Long Video Understanding With Learnable Retrieval in Video-Language Models，DOI 10.1109/TMM.2026.3664996；LoKR: Memory-Efficient Long Video Understanding via Low-Rank Key Residual Decomposition，DOI 10.1109/TMM.2026.3744094 — [Crossref TMM 检索](https://api.crossref.org/journals/1520-9210/works?query.bibliographic=training-free+long+video+understanding+multimodal&filter=from-pub-date:2025-01-01)
- MovieChat+ 已发表在 **TPAMI**（DOI 10.1109/TPAMI.2025.3604614），不是 TCSVT — [Crossref](https://api.crossref.org/works?query.bibliographic=MovieChat%2B)

### Inferences
- 精读优先级建议：EventLens > FDVS > ResPrune ≈ PTP > VideoExpert > AQAH > HeurVidQA。EventLens 和 ReadPast 一样，是用 Qwen2.5-VL / Qwen3-VL 做多 backbone 的免训练插件，其中 contrastive verdict refiner 作用在答案端，可类比我们的弃答模块。它最适合拿来对照 Experiments 的组织方式和 backbone 数量。
- FDVS 的定位是"把视频变成文本记忆再交给 LLM"，与 ReadPast 的自写文本时间线同源，可以作为 TCSVT 上的直接先例来引用。我们需要在 Related Work 里明确区分：FDVS 是离线、分层、带冗余消除；ReadPast 是流式、增量、定时写入，并配有 present/past 门控。
- TCSVT 上目前没有流式视频 LLM 论文，ReadPast 可以写成"把流式 VideoQA 引入本刊"。但这也意味着审稿人可能主要来自长视频 / 免训练 MLLM 方向，会拿 EventLens 这类论文的实验规模（5 benchmark × 7 backbone）来比较。

### Gaps
- IEEE Xplore 的 TCSVT Early Access 列表无法直接抓取，dblp 被反爬拦截，Semantic Scholar 返回 429。因此可能漏掉 2026 年下半年刚上线、Crossref/OpenAlex 尚未收录的 Early Access 论文。OpenAlex 和 Crossref 的覆盖都不完整，例如 FDVS 就没有出现在 OpenAlex 的标题检索结果里。
- EventLens、PGM、Clue Reasoning Chain、AQAH、T-MoENet、EcA 没有 arXiv 版，正文结构（章节、图表数）无法核实。
- GlimpsePrune、DiViCo、AQAH 的正文未读。

## 问题 2：每篇论文的结构细节（章节、benchmark/backbone 数、消融类型、效率表、页数、图数）

### Takeaway
以下按正文可读的 5 篇（FDVS、ResPrune、PTP、VideoExpert、HeurVidQA）加上 EventLens 的元数据整理。共同模式：正式版 12–16 页；Introduction 有 3–4 条贡献；Related Work 只有 2–3 个子节；Experiments 依次为 Settings → Main Results（ResPrune 按 backbone 分小节）→ Ablation（组件 + 超参扫描）→ Efficiency Analysis（单独小节）→ Case Study / Qualitative；**都没有单独的 Limitations 章节**，局限性写在 Conclusion 或实验讨论段里。

### Cited Findings

**FDVS（TCSVT 2025，16 页，108 篇参考文献）** — [arXiv HTML](https://arxiv.org/html/2412.06182)；[Crossref](https://api.crossref.org/works/10.1109/tcsvt.2024.3514820)
- 章节：I Introduction；II Related Works（II-A Long-form Video Understanding / II-B Video Foundation Models / II-C Video Understanding based on Large Language Models）；III Fine-Detailed Video Understanding（III-A Overview / III-B Efficient Keyframe-based Video Segmentation / III-C Visual Semantic Redundancy Reduction / III-D Perception-based Prompt Generation / III-E Textual Semantic Redundancy Reduction）；IV Experiments（IV-A Dataset Details / IV-B Experimental Setup / IV-C PRVR / IV-D VideoQA / IV-E Video Retrieval / IV-F Further Experiments and Discussions）；V Conclusion。
- 贡献：3 条，第一条即强调 "without any fine-tuning"。
- 任务与数据：3 个任务（PRVR、视频检索、VideoQA），7 个数据集（ActivityNet Captions、Charades-STA、MSRVTT、MSRVTT-QA、ActivityNet-QA、EgoSchema、NExT-QA）。摘要写"8 datasets"，对应 Table I 中的 8 个"数据集-任务"组合。
- 组件：Grounding DINO、InternVideo、BLIP-2、CLIP、Sentence-BERT、Vicuna-v1.5。LLM 换用 Vicuna-7B、Vicuna-13B、ChatGPT-3.5 做对比（Table XV）。
- 消融（共 7 类）：采样策略（全帧/均匀/关键帧）、视觉与文本冗余消除开关、感知模块组合、编码器与 LLM 替换、clip 描述数 K 与是否加 story、局部记忆长度 L、每 clip 帧数 4/8/16/32。
- 效率：Table X 比较每个视频的存储字节数（FDVS 87.3，Clip4Clip 840.0，减少 89.6%）；Table XVI 报告表示提取和下游任务的 FPS。FDVS 提取阶段 13.22 FPS，慢于基线，作者**主动承认**这一点。
- 定性：1 张与 VideoLLaVA 的对比图；没有专门的失败案例。
- 图/表：6 图、19 表。
- 局限：放在 IV-F 的 "Discussion of advantages and potential limitations" 段落（提取慢，因为串联多个大模型），后接 "Discussion about optimization of FDVS"。

**EventLens（TCSVT 2026，15 页，62 篇参考文献）** — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2026.3723106)；[GitHub](https://github.com/zhengkun332/EventLens)
- 5 个 benchmark（Video-MME、LongVideoBench、LVBench、MLVU、VideoEval-Pro），7 个 backbone（覆盖 Qwen2.5-VL、Qwen3-VL、LLaVA-Video、InternVL2.5/3，7B–32B）。固定帧预算 64，超参在 config.py 中明示。正文章节与图数无法核实，原因是没有 arXiv 版。

**ResPrune（TCSVT 2026 EA）** — [arXiv HTML](https://arxiv.org/html/2603.21105)
- 章节：II Related Work（II-A The Development of LVLMs / II-B Visual Token Pruning for LVLMs）；III Method（III-A Problem Formulation / III-B Surrogate Objective / III-C Greedy Subspace Expansion / III-D Conditioning on Textual Relevance / III-E Practical Implementation / **III-F Theoretical Efficiency Analysis**）；IV Experiments（IV-A Settings：Implementation Details / Evaluation Benchmarks / Comparison Methods；**IV-B Main Results 按 backbone 分为 IV-B1 LLaVA-1.5-7B / IV-B2 LLaVA-NeXT-7B / IV-B3 Qwen2.5-VL-7B**；IV-C Ablation Analysis 含 4 个子节；IV-D Practical Efficiency Analysis；IV-E Case Study）；V Conclusion。
- 贡献：3 条，依次为免训练框架、新公式与算法、多 backbone 评测。
- 9 个图像 benchmark，3 个 7B backbone，没有视频任务。
- 消融：组件消融；文本引导强度 α 扫描（唯一的超参敏感性实验）；相关性公式的 3 种写法；种子 token 选择的 5 种策略。
- 效率：理论部分给出复杂度和每层 prefill FLOPs；实测部分在 4×L20、batch=1 下测 FLOPs、KV-cache、GPU 显存、吞吐（tokens/s）和总评测时间，例如剪枝 66.7% 时 FLOPs 从 30.6T 降到 9.6T。
- 图/表：2 图、8 表，另有 1 个 Algorithm。没有失败案例。
- 局限：写在 Conclusion 末段。原文承认 α 是"fixed, manually tuned hyperparameter per model family"，不能随输入自适应。

**PTP（TCSVT 2026，13 页）** — [arXiv HTML](https://arxiv.org/html/2509.15704)
- 章节：II Related work 共 2 个子节；III Method 共 4 个子节；IV Experiments 依次为 Implementation Details → Main Results（按 InternVL-2B / 8B 分小节）→ Ablation（视觉编码器层选择、α、关键组件）→ **IV-D Efficiency Analysis** → IV-E Case Study；附录含逐层注意力偏好分析、注意力可视化、补充消融和 token 预算分析。
- 贡献：3 条（bottom-up 剪枝、top-down 剪枝、评测与洞察），全文多处声称 training-free、plug-and-play。
- 13 个图像 benchmark，2 个 backbone（InternVL2-2B/8B）。
- 效率（Table V）：token 数、每 token 延迟、总时间、TFLOPs、GPU 显存、KV-cache，单卡 L20 48GB。
- "为什么有效"：逐层注意力分析（含深层的 "attention drift"）；按任务类型解释最优 α（OCR 类任务偏低，开放域偏高）。
- 图/表：9 图、7 表。没有 Limitations 章节，也没有失败案例。

**VideoExpert（TCSVT 2026，16 页，74 篇参考文献）** — [arXiv HTML](https://arxiv.org/html/2504.07519)
- 章节：II Related Work（II-A Multimodal Large Language Model / II-B Temporal-sensitive Video Tasks）；III Methodology（5 个子节）；IV Experiments（IV-A Datasets and Settings / IV-B Main Results / IV-C Ablation Studies / IV-D Qualitative Result）；V Conclusion。
- 贡献：4 条，最后一条写"在 4 个数据集、4 个任务上的实验"。
- 4 个评测 benchmark（Charades-STA、QVHighlights、YouCookII、NExT-GQA）。只用 1 个 LLM（Vicuna-1.5 7B）。
- 动机实验：Fig. 2 是一个定性的 pilot，展示 TimeChat / VTimeLLM 对不同视频-查询对预测出相同的时间段，用来论证模型依赖语言模式。
- 效率：Table VIII 列出输入 token 数、推理速度（约 31.2 → 28.4 tokens/s）、GPU 显存（约 16.8–17.0 GB）。
- 图/表：7 图、9 表。失败案例只在正文中用文字描述。

**HeurVidQA（TCSVT 2025，16 页，85 篇参考文献）** — [arXiv HTML](https://arxiv.org/html/2410.09380)
- 章节：II Related Work（VideoQA / Vision-and-Language Pre-training / Prompt Learning）；III Method；IV Experiments（Datasets / Implementation / Results 按数据集分小节 / Ablation 含 4 个子节 / **IV-E Complexity analysis** / IV-F Qualitative Analysis）；**V Conclusions and Future Directions**。
- 4 个数据集，只有 1 个 backbone（ALPRO）。复杂度表列出可训练参数量、总参数量和 GFLOPs。
- 失败分析：按问题类型讨论，例如反事实问题落后最佳方法 4.7%，"How" 类问题会引入噪声。
- 图/表：9 图、11 表。局限写在 Conclusion 第二段："A key limitation lies in its reliance on domain-specific heuristic prompts"。

**页数汇总（正式版页码）** — 各篇 [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2025.3613993) 记录
- 12 页：视频字幕幻觉纠正（TCSVT.2025.3613993）。13 页：PTP。14 页：Clue Chain、T-MoENet、DiViCo、GlimpsePrune。15 页：EventLens、EcA。16 页：FDVS、VideoExpert、AQAH、HeurVidQA。
- 参考文献数：50（Clue Chain）至 108（FDVS），多数在 55–90 之间。

### Inferences
- 目标篇幅约 14–16 页（双栏、含参考文献和作者简介），参考文献 60–90 篇；15 张以上的表很常见（FDVS 19 张，HeurVidQA 11 张）。
- Main Results 建议仿照 ResPrune 按 backbone 分小节，或用"一张大表 + 每个 backbone 一组行"。ReadPast 有 4 个 backbone × 2 个 benchmark，按 backbone 分组呈现最清楚。
- 效率分析在免训练论文中几乎是必备的独立小节，指标通常为 FLOPs、延迟/吞吐、GPU 显存、KV-cache、token 数。FDVS 额外报告了存储字节数和 FPS，这与 ReadPast 的文本时间线很对口：可以报告每小时视频的文本 token 数和存储量，以及每 16 s 写一次 caption 的额外 VLM 调用开销。

### Gaps
- 以上图/表数取自 arXiv 版，TCSVT 正式版可能有变化（例如 GlimpsePrune 的 arXiv 版是 15 页、10 图，正式版是 14 页）。
- EventLens 的章节结构、消融内容和效率表无法确认。

## 问题 3：这些论文如何论证"免训练 / 即插即用"贡献的新颖性（新颖性主张、分析章节、"为什么有效"、局限性）

### Takeaway
TCSVT 上的免训练论文主要用三种方式论证新颖性：(1) **先把已有免训练方法的具体缺陷写清楚**，比如 EventLens 批评检索是 "flat frame-level"、只做"一次匹配"，ResPrune 批评注意力分数是 "one-shot"、与 FlashAttention 冲突；(2) **给出形式化或理论包装**，比如 ResPrune 写 Problem Formulation、代理目标和复杂度分析；(3) **用多 backbone、多 benchmark 证明通用性，再加效率表**。"为什么有效"通常靠 pilot 实验或注意力可视化（VideoExpert Fig. 2、PTP 的逐层注意力）来支撑。局限性普遍只占一段，而且常常就是"某个超参需要手调"。

### Cited Findings
- EventLens 的问题陈述三连：训练式方法"costly"→ 免训练检索有潜力 → 但现有方法"flat frame-level retrieval, lack coarse-to-fine hierarchical evidence localization, and depend solely on one-shot matching" — [OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2026.3723106)
- ResPrune 把自己与三类方法对比：注意力类（one-shot，选出的 token 容易冗余聚集）、跨模态注意力类（需要经过 LLM 层，与 FlashAttention 冲突）、多样性类（与文本无关）。新颖点表述为"residual-energy criterion + 在输入嵌入空间计算文本相关性，不依赖 LLM 注意力" — [arXiv](https://arxiv.org/html/2603.21105)
- ResPrune 的贡献原文强调"integrates into existing pipelines without model modification or retraining"；局限承认 α 需要按模型族手调，并提出"adaptive or learning-based mechanisms"作为未来工作 — [arXiv](https://arxiv.org/html/2603.21105)
- PTP 的局限与未来工作是"fully training-free, adaptive α-fusion strategies"。值得注意的是，它的 α 是在报告结果的同一批 benchmark 上扫出来的 — [arXiv](https://arxiv.org/html/2509.15704)
- FDVS 的辩护方式有三点：不训练、可跨任务迁移、文本存储比深度特征"more interpretable and cheaper"。它与有监督方法直接比较，并坦承部分指标落后，例如 LLM-assisted MSRVTT-QA 上 LLaMA-VID 58.9，FDVS 53.7 — [arXiv](https://arxiv.org/html/2412.06182)
- AQAH 的辩护方式：与指令微调（要重训）和解码约束（会拖慢推理、缩短输出）两条路线对立，突出"不约束解码"带来的推理效率优势 — [OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2025.3618949)
- VideoExpert 用一个定性 pilot（Fig. 2）证明 MLLM "rely more on language patterns than visual cues"，作为方法动机 — [arXiv](https://arxiv.org/html/2504.07519)
- PTP 的"为什么有效"：逐层注意力分析（浅、中、深层的偏好与 "attention drift"）、按任务解释 α，以及解释剪枝反而涨分的原因（减少噪声） — [arXiv](https://arxiv.org/html/2509.15704)

### Inferences
- **ReadPast 可以直接利用的论证点**：ResPrune 和 PTP 都把"阈值或权重 α 需要手调"列为局限或未来工作，而 ReadPast 的门控阈值是**用 content-free 输入无标签校准**的（思路沿用 Zhao et al. 2021 的 contextual calibration）。这正好对应同刊近作承认的弱点，可以在 Introduction 或 Related Work 里点明："现有免训练插件的超参依赖人工或 benchmark 调参，我们的门控阈值无需任何标注"。注意 PTP 的 α 是在测试集上扫的，这一点适合作为对比的反例。
- 新颖性段落可仿照 EventLens 的"三连"写法：SimpleStream 证明了只看最近 4 帧很强 → 但所有回看型问题都没有信息来源 → 已有记忆方法要训练，或存视觉 token / KV（开销大）→ ReadPast 用自写文本时间线加问题端门控，零训练、记忆开销小。
- "为什么有效"需要有一个类似 VideoExpert Fig. 2 的 pilot：例如统计 SimpleStream 在 OVO backward 与 real-time 子任务上的失败分布，或者展示门控的 log P("past") − log P("now") 分布在校准前后的变化。这比只放消融表更符合 TCSVT 近作的写法。
- 弃答模块借用自 D-HSM，新颖性弱，建议像 FDVS 处理借用组件那样，明确写成"we adopt"，并单独做消融证明它的增益与适用条件（只在选项含弃答项时触发）。

### Gaps
- 没有拿到 EventLens 正文，无法确认它的 "why it works" 分析和局限段落具体怎么写。
- 没有找到 TCSVT 审稿意见公开样本（TCSVT 不公开审稿意见），审稿人偏好只能从已发表论文的结构反推。

## 问题 4：综合清单——该方向 TCSVT 审稿人期望看到的实验与章节

### Takeaway
综合 6 篇正文可读或元数据充分的论文，TCSVT 免训练 MLLM / 长视频论文的"标配"是：3–4 条贡献、2–3 个 Related Work 子节、方法中有 Problem Formulation 和复杂度分析、多 backbone 主结果、组件消融、至少一个超参扫描、独立的效率小节、Case Study，以及写在结论里的一段局限。失败案例和单独的 Limitations 章节在同刊论文中很少见，补上这两项能形成差异化。

### Cited Findings
- 3–4 条贡献，最后一条通常是"大规模实验验证"：FDVS 3 条、ResPrune 3 条、PTP 3 条、HeurVidQA 3 条、VideoExpert 4 条 — [FDVS](https://arxiv.org/html/2412.06182)；[ResPrune](https://arxiv.org/html/2603.21105)；[PTP](https://arxiv.org/html/2509.15704)；[VideoExpert](https://arxiv.org/html/2504.07519)；[HeurVidQA](https://arxiv.org/html/2410.09380)
- Related Work 只有 2–3 个子节，分别对应"基础模型线 + 任务线 + 技术线"：FDVS 为长视频 / 视频基础模型 / 基于 LLM 的视频理解；ResPrune 为 LVLM 发展 / 视觉 token 剪枝 — 同上
- 独立的效率小节：ResPrune IV-D（FLOPs、KV、显存、吞吐、总时间）、PTP IV-D（token 数、延迟、TFLOPs、显存、KV）、HeurVidQA IV-E（参数量、GFLOPs）、FDVS（存储字节数、FPS）、VideoExpert（token 数、tokens/s、显存） — 同上
- 多 backbone：EventLens 7 个、ResPrune 3 个、PTP 2 个（2B/8B）、FDVS 换用 3 种 LLM；VideoExpert 和 HeurVidQA 只用 1 个，但它们是训练式方法 — [GitHub](https://github.com/zhengkun332/EventLens)；以及上述各篇 arXiv
- 超参敏感性：ResPrune 和 PTP 扫 α，FDVS 扫记忆长度 L 和每 clip 帧数，VideoExpert 扫 α / GOP 设置，HeurVidQA 扫帧数和候选词表大小 — 同上
- 失败案例：FDVS、ResPrune、PTP 都没有专门的失败案例；HeurVidQA 按问题类型做了失败讨论；VideoExpert 只用文字描述 — 同上
- Limitations：5 篇都没有单独章节，局限写在 Conclusion 或实验讨论段里 — 同上

### Inferences（面向 ReadPast 的清单）
1. **Introduction**：动机图（SimpleStream 在 backward 题上的失败示例）→ 三连式问题陈述 → 3–4 条贡献（文本时间线、无标签校准门控、弃答指令的条件触发与系统集成、在 2 个 benchmark × 4 个 backbone 上的验证与分析）。
2. **Related Work**（3 个子节）：Video LLMs 与长视频理解（引 TCSVT 综述、FDVS、EventLens）；流式 / 在线视频理解（SimpleStream、OVO-Bench、StreamingBench 及记忆类方法）；免训练推理时方法与校准（contextual calibration、token 剪枝类插件，可引 ResPrune / PTP 作为 TCSVT 先例）。
3. **Method**：Problem Formulation（流式 QA 的形式化，区分 present / past 查询）→ 三个模块各一小节 → 复杂度与开销分析（caption 写入频率、文本记忆 token 随时长线性增长、门控只读问题因此只需一次前向）。
4. **主结果**：OVO-Bench（backward + real-time 的标准平均分和逐任务分数，遵循已有决定：不加 noHLD 列）和 StreamingBench real-time。4 个 backbone 按组展示，相对 SimpleStream 的增益用点数（+6.0/+5.7/+5.2/+2.5）。对照 EventLens 的规模，审稿人可能要求 ≥1 个更大的 backbone（如 32B）或另一个模型族（InternVL），建议评估可行性。
5. **消融**：三个模块逐一加减；门控 vs. 始终读过去 vs. 从不读过去（oracle 上界）；content-free 输入集合的选择（N/A、""、[MASK] 及其组合）；caption 间隔（如 8 / 16 / 32 s）与每段帧数；时间线最大长度或截断策略；弃答指令只在含弃答选项时触发 vs. 始终触发。
6. **门控分析（"为什么有效"）**：门控对 past/now 的分类准确率或混淆矩阵（若有子任务标签可近似）、校准前后的分数分布图、跨 backbone 的阈值稳定性。这对应同刊论文"超参需手调"的痛点，是 ReadPast 最好的分析卖点。
7. **效率小节**：每个查询的额外延迟（门控一次前向 + 读文本时间线的 prefill）、每小时视频的 caption 写入次数与 GPU 时间、文本时间线 token 数与存储（参照 FDVS 的"字节/视频"表）、与存储视觉 token 或 KV 的记忆方法的显存对比。
8. **定性与失败案例**：成功案例（门控正确路由到时间线）加失败案例（caption 漏记细节、门控误判），按任务类型讨论。同刊论文普遍缺这一项，补上可以加分。
9. **Limitations 段落**：写入端瓶颈（caption 的信息损失）、依赖同一个 VLM 的 caption 质量、弃答模块借用自 D-HSM。可以放在 Conclusion 前的 Discussion 小节。
10. **篇幅**：14–16 页，60–90 篇参考文献，10 张以上的表。

### Gaps
- TCSVT 官方的页数上限（regular paper 的免费页数与超页费阈值）未能核实。作者指南页只说明了超长页费的估算方法（总页数除以 3）和 2026 年开放获取 APC 2,800 美元，没有给出页数上限 — [TCSVT Guidelines for Authors](https://ieee-cas.org/publication/tcsvt/guidelines-authors)

## 问题 5：TCSVT 社论 / 审稿指南 / EDICS 中与 MLLM 视频理解相关的信息

### Takeaway
TCSVT 曾开设 **"Special Issue on Large Language Models (LLMs) for Video Understanding"** 专刊，截稿日 2025-01-02，已截稿；2024 年第 34 卷第 10–11 期登记了 4 条同名单页条目。EDICS 列表页无法访问，具体类别名称未能核实。

### Cited Findings
- 专刊 CfP 标题为 "IEEE Transactions on Circuits and Systems for Video Technology Special Issue on Large Language Models (LLMs) for Video Understanding"。摘要部分强调 LLM 在视觉-语言任务中的 "versatility ... without extensive retraining"，以及需要解决"enhancing cross-frame context ... improving efficiency"等挑战。征稿主题包括零/少样本视频表示、视频-语言基础模型、可解释性与安全等。截稿 2025-01-02，最终录用通知 2025-05-01。客座编辑为 Jungong Han、Liqi Yan、Zheng (Thomas) Tang、Dongfang Liu、Zhuang Shao、Roland Goecke — [CfP PDF](https://ieee-cas.org/files/ieeecass/2024-12/TCSVT-SI-LLMVU-CfP.pdf)
- Crossref 中有 4 个同名条目 "Large Language Models (LLMs) for Video Understanding"（DOI 10.1109/TCSVT.2024.3471329 / .3483389 / .3489299 / .3489309），无作者，vol.34 no.10 的一条页码为 9758–9758，即单页 — [Crossref](https://api.crossref.org/works/10.1109/tcsvt.2024.3471329)
- 投稿时须至少选择一个 TCSVT EDICS 类别，选得越具体越有利于匹配 AE 和审稿人。投稿类型分为 Transactions Papers 和 Transactions Letters（来自 WebSearch 对 ieee-cas.org 投稿说明页的摘要，原页未直接抓取） — [TCSVT Submission of a Manuscript](https://ieee-cas.org/node/3112)
- 作者指南页面没有列出 EDICS 类别 — [TCSVT Guidelines for Authors](https://ieee-cas.org/publication/tcsvt/guidelines-authors)

### Inferences
- 这 4 条无作者、单页、与专刊同名的条目，推测是刊登在期刊中的专刊征稿广告页，不是社论正文（未核实）。
- 专刊 CfP 明确把"无需大量重训"和"效率"列为关注点，说明 TCSVT 编辑层面认可免训练加效率的叙事，ReadPast 的 cover letter 可以呼应这一点。
- EDICS 选择建议（未核实具体名称，投稿时以 ScholarOne 下拉列表为准）：优先选与"视频分析 / 理解"和"多媒体内容分析 / 跨模态"相关的类别，次选机器学习类。

### Gaps
- EDICS 列表页（tcsvt.polito.it/edics.html）SSL 握手失败，无法获取，因此**没有核实任何 EDICS 类别的确切名称**。建议在 ScholarOne 投稿界面直接查看。
- 没有找到 TCSVT 针对 MLLM 方向的审稿指南或社论正文。
- 专刊收录的具体论文列表没有找到。
