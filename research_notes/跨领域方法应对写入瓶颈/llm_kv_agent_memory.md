# LLM 长上下文 / KV 压缩与 Agent 记忆：query-agnostic vs query-aware、写入期记忆构建（对照 StreamKV 写入瓶颈）

> 本笔记服务的现象：7B VLM 每 16 s 写一条通用 caption（query-agnostic 文本压缩），问答时读 caption 时间线 + 最近 4 帧。OVO-Bench memory 题：约 44% 记忆里**没有**答案信息（absent），约 44% 正确选项与干扰项的词**同时**出现（ambiguous），约 6% clean；换写入模型、分辨率、粒度、prompt、实体图，这些比例都不变；读取端没有余量；7B 在问答时回看帧会**编造证据**（fabrication）。
> 写作约定：Cited Findings 中每条均附来源；凡是我自己的推算或迁移判断，一律放在 Inferences。2026 年的论文均来自本次检索到的 arXiv 摘要/HTML 页面，我没有独立复现。

---

## Q1. query-agnostic 压缩比 query-aware 压缩丢掉多少信息？是否存在仍保持 query-agnostic 的补救方法（重建目标、多假设、保留原始块）？

### Takeaway
不止一项研究定量证明：依赖“问题可见”的打分方法（SnapKV 类）一旦在看到问题之前压缩，就会大幅退化。SCBench 中 SnapKV 的后续轮次掉到接近 0；2026 年的 matched-budget 审计里 SnapKV 掉 0.198（准确率，0–1 标度），甚至输给“保留开头 + 最近窗口”这种平凡基线。能在 query-agnostic 下站住的方法分两类：一类是**不含 query 项的打分**（KeyDiff），另一类是**重建目标**（KVzip、EchoPress、KV²）。但这些方法“近乎无损”的区间只到约 30% 保留率，而 caption 记忆的压缩率比这激进约两个数量级。在任何领域里我都没找到证据表明，压到 1–2% 的 query-agnostic 压缩还能保住细节，除非另外保留原始数据作为回退。

### Cited Findings
**A. 量化 query-agnostic 损失的证据**
- SCBench（Li et al., ICLR 2025）：在 shared-context 多轮设置下（LLaMA-3.1-8B，压缩预算 1/32），full attention 的 multi-turn 平均为 48.7 / multi-request 为 37.2；SnapKV 为 20.9 / 15.2，PyramidKV 为 21.0 / 15.3，StreamingLLM 为 15.5 / 14.5。动态稀疏的 MInference 为 42.5 / 36.4。值得注意的是，保留全部 KV、在 decode 时做 query-aware 选择的 Quest 只有 22.1 / 19.6，RetrievalAttention 只有 29.4 / 25.8 — [SCBench HTML](https://arxiv.org/html/2412.10319v2)
- SCBench Table 5（SnapKV，首轮 vs 不带 query 的后续轮）：Retr.Semantic 19.0 → 9.7，Global 17.9 → 14.6，Multi-task 5.1 → 0.0，Retr.String 0.0 / 0.0。作者的解释是“important KVs remain stable within a turn, they vary significantly between queries”：SnapKV 与 MInference 都用输入末段（通常就是问题）来估计重要性，问题不在就失效。作者还指出，sparse decoding 在后续请求上掉点，而 sparse encoding + dense decoding 更稳 — [SCBench HTML](https://arxiv.org/html/2412.10319v2)
- KVzip（Kim et al., NeurIPS 2025 Oral）：在多 query 复用同一份缓存的场景下，现有 query-aware 驱逐方法“suffer from performance degradation even at a 90% cache budget ratio”，引言里说“significant performance degradation even at 10% KV eviction”。Fig. 2（SQuAD，LLaMA3.1-8B）中，SnapKV 按 query 逐次压缩时准确率很高，但复用首个问题压缩出的缓存时准确率骤降 — [KVzip abs](https://arxiv.org/abs/2505.23416)，[KVzip HTML](https://arxiv.org/html/2505.23416v2)
- 2026 matched-budget 审计（Luo, Liang, Xuan, 2026-07）：6 种方法（SnapKV、H2O、TOVA、ExpectedAttention、AdaKV、KeyDiff）× 3 个 7–9B 模型，在 RULER-8192 上跑了 144,300 对配对实验，只切换“问题是否对打分可见”。query-aware 减 query-agnostic 的差 Δ 为：SnapKV +0.198（最大），AdaKV +0.172，TOVA +0.103，ExpectedAttention +0.024，KeyDiff +0.011（最小）。在 agnostic 协议下，SnapKV 平均落后于“三个平凡基线取最优”（Random / Knorm / StreamingLLM）0.066，36 个格子只赢 13 个；KeyDiff 赢 31/36。单格示例：Qwen2.5-7B 检索任务 r=0.9 时，agnostic 下 KeyDiff 0.663、TOVA 0.260、SnapKV 0.077，aware 下 SnapKV 也只有 0.230。作者对照源码确认，掉点幅度与问题对打分信号的可见度一致：SnapKV 的 64-token 观察窗口包含问题，KeyDiff 的打分没有 query 项 — [Audit abs](https://arxiv.org/abs/2607.11942)，[Audit HTML](https://arxiv.org/html/2607.11942v1)
- StreamMem（Meta/NYU, 2025-08，流视频，LLaVA-OV-7B，MLVU）：用真实问题做 query-aware 剪枝得 68.1，用 chat template 当代理 query 得 66.9（−1.2），用通用问题“What is happening in the video?”得 66.7；multi-detail 子类 46.4 vs 43.0（−3.4）。作者承认 query-agnostic 方法很难留住 multi-detail 题需要的全部细节 — [StreamMem HTML](https://arxiv.org/html/2508.15717v1)
- LongMemEval（Wu et al., ICLR 2025）：把 session 压成 summary 或 user facts 作为 value，会“negatively impacts QA performance due to information loss”。唯一例外是：fact 分解在 multi-session reasoning 上一直有提升 — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)
- ReadAgent（Lee et al., 2024，Google DeepMind）：QuALITY（PaLM 2-L）上，只用 gist（压缩率 85.53%）得 77.52%，原文全文得 85.83%。说明 query-agnostic 的 gist 压缩比原文低约 8.3 个点 — [ReadAgent HTML](https://arxiv.org/html/2402.09727v3)
- LoCoMo（Maharana et al., ACL 2024）：RAG 的检索单元若是 session summary，F1 最高 32.5；原始对话轮最高 35.8（top-25）；observation（逐 session 抽取的断言）在 top-5 时 41.4。observation 的分数随 k 增大而下降（信噪比变低）— [LoCoMo HTML](https://arxiv.org/html/2402.17753v1)

**B. 保持 query-agnostic 的补救方法**
- **重建目标 / KVzip**：先输入 “Repeat the previous context:”，再对原文做 teacher-forced 重建（一次前向），每个 KV 的重要性取重建过程中它收到的最大注意力。分块打分（2K chunk）使复杂度对上下文长度线性。打分成本约为 2× prefill，额外显存 <2%。结果：KV 缩小 3–4×，decode 延迟约 2× 降低，在 30% 保留率下退化可忽略（Retr.Prefix-Suffix 例外）；GSM8K 约 20–30% 仍近无损；高冗余任务可压到 10%。只重建前 10%、后 10% 或只用 prompt 都会退化（Fig. 12），必须重建**全文** — [KVzip HTML](https://arxiv.org/html/2505.23416v2)
- **EchoPress**（2026-09）：用标准 prefill 的 Q/K 近似 KVzip 的重建注意力，只重建第一个 chunk 来校准打分。在 50–90% 驱逐率下与 KVzip 准确率相当，压缩开销降低 1.7–19.6×，prefill 总时间最多快 2.9×（Qwen3-8B、Llama-3.1-8B，LongBench / RULER）— [EchoPress abs](https://arxiv.org/abs/2610.00412)
- **KV²**（2026-10）：先用轻量代理打分器选出信息量大的 token，只对这一子集做选择性重建。RULER 16K、2% 预算下平均分比次优基线高 40 个百分点以上；LongBench 在 2–10% 预算下平均最高。摘要没有给出逐基线数字 — [KV² abs](https://arxiv.org/abs/2610.03198)
- **不含 query 的打分 / KeyDiff**：见 A 中审计。它是 agnostic 协议下唯一稳定胜过平凡基线的方法；但在自然文本（LongBench）上，TOVA、ExpectedAttention 能追平或超过它 — [Audit HTML](https://arxiv.org/html/2607.11942v1)
- **代理 / 元 query**：
  - StreamMem 用 chat template token 当通用 query。理由是预训练里大量 captioning 数据，使这些 token 隐式等同于“描述视频” — [StreamMem HTML](https://arxiv.org/html/2508.15717v1)
  - TaskPress（EMNLP 2026）用高层“任务说明”作为 prefill 期间的 meta-query，提前过滤无关 token，生成可复用缓存；摘要没有数字 — [TaskPress abs](https://arxiv.org/abs/2608.03276)
  - Expected Attention（Devoto, Jeblick, Jégou, 2025-10，NVIDIA KVPress）用激活的分布性质闭式估计“未来 query 的期望注意力”。但在上面的审计中，它在 agnostic 下反而只赢 14/36，平均 −0.051 — [Expected Attention abs](https://arxiv.org/abs/2510.00636)，[Audit HTML](https://arxiv.org/html/2607.11942v1)
- **延迟压缩**（agent 场景，2026-08）：“immediate compaction often hurts performance”，推迟压缩、等拿到 agent 后续的 query 再压，可以“recovers much of the gap”。token eviction 可在保住大部分准确率的同时把 KV 减少 80%；代理 query 怎么选（boundary / repeat-prefill / delayed future-generation）是“a core design choice”— [Online KV Compaction abs](https://arxiv.org/abs/2608.00902)
- **保留原始块、按需回读**：ReadAgent 先看 gist，再查原文页（QuALITY 平均查 2.3 页），得到 86.83%，**高于**全文 85.83%。QMSum 上顺序查页（ReadAgent-S）最好（46.57% vs gist-only 40.20%），但并行查页（ReadAgent-P）37.99% 反而低于 gist-only — [ReadAgent HTML](https://arxiv.org/html/2402.09727v3)
- **流视频的 query-agnostic KV 压缩是可行的**：StreamMem 在 6K KV 预算下，Qwen2-VL-7B 的 MLVU 65.9，与 full KV（50K）的 65.9 持平；预算到 24K 时 66.3，略超 full KV。InfiniPot-V 在 6K 下 65.8 — [StreamMem HTML](https://arxiv.org/html/2508.15717v1)；InfiniPot-V 同为 training-free、query-agnostic，报告峰值显存最多降 94% — [InfiniPot-V (HF papers)](https://huggingface.co/papers/2506.15745.md)

### Inferences
- **压缩率决定 query-agnostic 能否成立。**文本 KV 文献中“近乎无损”的 query-agnostic 区间约在 20–30% 保留率（KVzip）。StreamMem 能成立，是因为 6K token 对 32 帧的视觉 token 来说仍是中等压缩。我们的记忆是“16 s 视频 → 一条 caption（几十到一百多 token）”，相对 16 s 视觉 token（以 1 fps、每帧数十到数百 token 计，约数千 token），压缩率大约 1–3%（我的粗估，非实测）。这已落在 KV² 测试的 2% 极端区间，而且是**跨模态有损**（视觉 → 文本），无法像 KVzip 那样用“能否重建输入”来约束。所以 absent 约 44% 的根因更像是**信息论意义上的容量不足**，不是写入 prompt 不好。这和我们“换写入模型 / prompt / 粒度比例不变”的观察一致。
- **我们的 caption 写入相当于 SnapKV-agnostic + 极端预算。**caption prompt 只是一个通用代理 query（等价于 StreamMem 的“describe the video”），模型写的是它认为显著的东西。审计表明，代理 query 依赖越强，换成真实问题后的落差越大（SnapKV Δ≈0.2）。而 query 无关、偏向多样性的打分（KeyDiff）在 agnostic 下最稳。启示：若存视觉 token / 帧，选择准则应偏向**新颖性 / 多样性**（KeyDiff、InfiniPot-V 的时间冗余），不要用“caption 注意力”这类代理 query 显著性。后者只会强化 caption 已有的偏向，正好与 absent 桶重叠。
- **KVzip 的“重建目标”无法直接迁移到 caption。**caption 不能重建视频；它的文本类比（让模型从 caption 回答一组通用自问题）本质上就是“多假设 prompt”，而我们已证伪 memory-oriented prompt 有效。可迁移的只是一种诊断思路：用“caption 能否回答针对该片段自动生成的问题”来度量写入覆盖率。这只能当诊断，不能当补救。
- **SCBench 的反直觉结果：Quest / RetrievalAttention 保留全部 KV 但做 sparse decoding，后续轮依然低。**“保留原始数据”本身不够，读取时的选择机制（检索召回）同样是瓶颈。对应到我们：如果存原始帧，检索步骤（caption 当索引）在 absent 桶上会天然失败，因为索引里没有那个实体。

### Gaps
- 没找到把“视觉 → 文本 caption”这种跨模态 query-agnostic 压缩与“保留视觉 token”直接对比、并按信息缺失 / 歧义分桶统计的工作。
- 审计论文（2607.11942）、TaskPress、KV²、EchoPress 都是 2026 年 arXiv 预印本，我只读到摘要或部分 HTML。KV² 没有给出与 KVzip 的逐项对比；TaskPress 摘要没有数字。
- InfiniPot（EMNLP 2024）的 Catalyst Prompt / NuC 细节在摘要页里看不到，本次没核实，不作为证据使用。
- 二手页面声称一项对照消融中“verbatim chunks 比 extracted memories 在 LoCoMo 高 15.9、在 LongMemEval-S 高 22.0”（[Agent Memory Field Guide](https://theagentfieldguide.com/research/agent-memory-systems/)），但该页参考文献为空，**找不到原始来源**，不能引用为事实。

---

## Q2. Agent 记忆基准（LongMemEval、LoCoMo）里，原始历史存储 + 检索是否胜过 summary / fact 记忆？哪些写入期策略有效？

### Takeaway
结论并不一边倒：
- LongMemEval 明确指出 summary / fact 作为 value 会因信息丢失而掉点，而把 fact 当作**key 扩展**（value 仍是原文）能提升召回和准确率。
- LoCoMo 上，抽取出的 observation 作为检索单元最好。
- 商业抽取式记忆（Mem0）比 full-context 低约 6 个 J 分点；保留原始 episode 的 Zep 比 full-context 高，但在“抽取 schema 没覆盖的内容”（assistant 说过的话）上掉了 14 个点。

最稳定的写入原则是：**key 和 value 解耦**，key 做扩展（fact / 时间戳），value 保留原文（或可回溯到原文）。时间感知的 query 扩展有效，但依赖强 LLM，8B 模型会幻觉。

### Cited Findings
**LongMemEval（Wu et al., ICLR 2025；500 题，5 种能力：信息抽取、多 session 推理、时间推理、知识更新、拒答）**
- 长上下文 LLM 从 oracle 到 LongMemEval_S 掉 30–60%：GPT-4o 0.870 → 0.606（−30.3%），Llama 3.1 70B 0.744 → 0.334（−55.1%），Llama 3.1 8B 0.710 → 0.454（−36.1%）。商业助手 ChatGPT 0.5773、Coze 0.3299，对比离线阅读 0.9184 — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)
- 粒度：把 session 拆成 round 能显著提升 GPT-4o 的阅读表现；summary / fact 作为 value 因信息丢失而伤害 QA — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)
- **Fact-augmented key expansion（K = V + fact）**：所有模型平均 recall@k +9.4%、最终准确率 +5.4%。Round 粒度下 Recall@10 0.692 → 0.784，GPT-4o top-5 QA 0.615 → 0.657；Session 粒度下 Recall@10 0.783 → 0.862，QA 0.670 → 0.714 — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)
- **Time-aware query expansion**：时间推理子集上 recall 平均 +11.3%（round）/ +6.8%（session）；例如 Round、K=V+fact 时 Recall@10 0.550 → 0.722（用 GPT-4o 做扩展）。但用 Llama 3.1 8B 做扩展时，K=V 的 Recall@5 0.421 → 0.384，**下降**。作者写道该方法“depends on using a strong LLM”，Llama 8B“often hallucinat[es] or miss[es] temporal cues” — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)
- 读取：Chain-of-Note 在 oracle 下让 GPT-4o 0.870 → 0.924；但在 LongMemEval_S 上，Llama 3.1 70B 加 CoN 后 0.334 → 0.286，8B 0.454 → 0.420，**反而更差** — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)

**LoCoMo 与商业记忆系统**
- LoCoMo（ACL 2024）RAG F1：observation top-5 41.4 > dialog 最佳 35.8 > summary 最佳 32.5；GPT-3.5-16K 全上下文 37.8。事件摘要的错误类型包括遗漏信息、幻觉、说话人误归属、把琐碎交流当作显著事件 — [LoCoMo HTML](https://arxiv.org/html/2402.17753v1)
- Mem0（Chhikara et al., 2025-04）LoCoMo LLM-judge J：full-context 72.90，Mem0g 68.44，Mem0 66.88，Zep 65.99，best RAG（chunk 256, k=2）60.97，LangMem 58.10，OpenAI 52.90，A-Mem 48.38。Mem0 每 query 检索 1,764 token（full-context 为 26,031），p95 总延迟 1.44 s（full-context 为 17.1 s）。**抽取式记忆在准确率上不如 full-context**，优势在成本 — [Mem0 HTML](https://arxiv.org/html/2504.19413v1)
- Letta（2025-08-12 博客，非同行评审）：把原始 LoCoMo 对话放进一个文件，只给 grep / search_files / open / close 工具，用 GPT-4o mini 迭代搜索，得 74.0%，对比 Mem0 自报的 68.5%。作者认为关键在“agent 知道何时、如何检索”，不在存储结构。另有评论指出 LoCoMo 不考知识更新，且 Mem0 在 2026-04 已换新管线并自报 92.5% — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)；评论见 [Aktagon signals](https://signals.aktagon.com/articles/2026/02/benchmarking-ai-agent-memory-is-a-filesystem-all-you-need/)（二手）
- Zep（Rasmussen et al., 2025-01）保留**原始 episode**（“a non-lossy data store from which semantic entities and relations are extracted”），在此之上建实体 / 事实图，事实可回溯到源 episode。LongMemEval：gpt-4o-mini 55.4%（full-context，115k token）→ 63.8%（Zep，1.6k token）；gpt-4o 60.2% → 71.2%。时间推理 45.1 → 62.4（4o）。但 **single-session-assistant 94.6 → 80.4（4o），81.8 → 75.0（mini）**，knowledge-update 在 mini 上 76.9 → 74.4 — [Zep HTML](https://arxiv.org/html/2501.13956v1)

**其他写入期设计**
- SeCom（Pan et al., 2025，Microsoft）：turn 级、session 级、summary 式记忆单元各有局限；按话题分段的 segment 级单元更好；用 LLMLingua-2 对记忆单元做压缩去噪，能在各种粒度上提升检索准确率（摘要无具体数字）— [SeCom abs](https://arxiv.org/abs/2502.05589)
- MemoryAgentBench（Hu, Wang, McAuley, ICLR 2026）：四项能力（accurate retrieval、test-time learning、long-range understanding、selective forgetting）。RAG 在需要全局理解的任务上差，长上下文也不是万能解 — [MemoryAgentBench](https://arxiv.org/html/2507.05257v2)（数字未取到）
- “When Does Memory Help?”（2026-07，tool-using agents）：全量回放“never economical”。对更新过的事实，嵌入检索在不同模型上 0.30–0.95 不稳定，种子间最大差 0.45；update-on-write 的结构化事实库和 LLM 摘要保持 0.70–1.00；仅仅换实现就能让任务成功率变动多达 60 点 — [When Does Memory Help abs](https://arxiv.org/abs/2609.05441)
- 经典架构（方法描述）：
  - Generative Agents（Park et al., UIST 2023）存原始 observation 的 memory stream，按 recency + importance + relevance 检索，并周期性生成 reflection — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
  - MemGPT（Packer et al., 2023）用 OS 式分层：main context + recall storage（原始消息）+ archival storage，由 LLM 通过函数调用分页 — [arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
  - MemoryBank（Zhong et al., AAAI 2024）用 Ebbinghaus 遗忘曲线更新记忆强度 — [arXiv 2305.10250](https://arxiv.org/abs/2305.10250)
  - A-Mem（Xu et al., 2025）用 Zettelkasten 式笔记，LLM 写关键词 / 标签 / 上下文描述并建链接 — [arXiv 2502.12110](https://arxiv.org/abs/2502.12110)。它在 Mem0 的复现中只有 48.38 J — [Mem0 HTML](https://arxiv.org/html/2504.19413v1)

### Inferences
- **“raw vs summary”答案取决于 value 是否被替换。**summary / fact 替换原文作为 value 时掉点（LongMemEval、Mem0 vs full-context、ReadAgent gist-only）；作为 key 或索引、同时保留原文时涨点（LongMemEval K=V+fact、Zep、ReadAgent 查页）。LoCoMo 中 observation 胜出算是反例，但 LoCoMo 问题多是对用户事实的单跳 / 多跳，恰好与抽取 schema 对齐。
- **Zep 的 single-session-assistant 掉 14 点，是“写入 schema 决定丢什么”的直接证据。**抽取器关注用户事实，所以 assistant 讲过的细节丢了。对应到我们：通用 caption 关注“场景 / 主要动作”，OVO-Bench 的 episodic memory 题（物体位置、次序、细小属性）正是 schema 之外的内容。这与“换 prompt / 实体图比例不变”的观察一致：schema 换了，但都是同一类显著性先验。
- **我们当前的记忆让 caption 同时充当 key 和 value。**按 LongMemEval 的框架，这相当于 summary-as-value，是文献中最差的配置之一。最直接的迁移是：caption（加实体、时间戳）只当 key，value 换成该 16 s 片段的原始帧或紧凑视觉 token。
- **时间感知 query 扩展能对付 ambiguous 桶的一部分，但不要让 7B 做扩展。**OVO-Bench 题干常含时间指代；如果干扰项的词出现在时间上不相关的 caption 里，按时间限定窗口就能消歧。用规则或正则解析时间短语比让 7B 生成扩展更安全，因为 LongMemEval 显示 8B 做扩展会让 recall 下降。
- **小模型 + 复杂读取策略可能更差**（Llama CoN 掉点）。这与我们“读取端无余量、回看会编造”的观察一致。

### Gaps
- 没找到在 LongMemEval / LoCoMo 上，以“同一写入器、仅 value 是原文还是摘要”为唯一变量、并按“答案信息缺失 vs 歧义”分桶的对照。前述 15.9 / 22.0 的消融来源不可考。
- LongMemEval Fig. 5 / Fig. 6 的 QA 准确率只有图，HTML 文本里没有具体数值。
- Mem0 自报的 2026 年 92.5% 来自二手评论，未核实。

---

## Q3. “保存紧凑的过去视觉 KV / token 库，只在 gate 判定为过去题时读取”对 7B VLM 是否可行？KV 检索方法在 episodic memory 任务上报告了什么？

### Takeaway
KV 检索确实提升 episodic memory 题，但对 7B 的幅度有限：ReKV 在 QAEgo4D 上 LLaVA-OV-7B 从 53.0（uniform）到 56.0（internal retrieval），而 oracle 检索也只有 64.4。说明检索召回从 6% 提到 70% 时，读取端（7B 用检索帧作答）仍有约 8 点的天花板差距。query-agnostic 的紧凑视觉 KV（StreamMem、InfiniPot-V）在 6K 预算下接近 full KV，所以“紧凑存储”在技术上可行。真正的风险在两处：SimpleStream 报告的“历史上下文伤害实时感知”，以及我们已观察到的回看编造。gate 路由正是为了隔离前者，但不能消除后者。

### Cited Findings
- ReKV（Di et al., ICLR 2025，training-free）：滑窗编码，KV 卸载到 RAM / 磁盘，问答时用外部检索器或 VLM 自身注意力（internal）取回相关 KV。QAEgo4D test-mc：LLaVA-OV-7B 为 uniform 53.0 / external 54.2 / internal 56.0 / oracle 64.4；0.5B 为 42.6 / 48.0 / 50.0 / 52.0。检索 recall：uniform 6.1%、external 58.1%、internal 70.5%（7B）。“higher recall consistently leads to better accuracy”。检索帧从 8 增到 80 时准确率总体上升，但 MLVU 上会饱和，因为多余的无关帧伤害作答。RVS-Ego / Movie（7B）：uniform 56.2 / 43.0，internal 63.7 / 54.4。延迟 2.9 s → 3.3 s（internal）/ 5.8 s（external），显存 21 → 38 / 55 GB。作者列出的局限：缓存无界增长、固定块大小、固定检索帧数 — [ReKV HTML](https://arxiv.org/html/2503.00540v1)
- StreamMem（query-agnostic 视觉 KV，有界内存）：LLaVA-OV-7B 6K 预算下 MLVU 66.9 / EgoSchema 63.0 / VideoMME 59.4；ReKV（每小时 353K token，不限内存）为 68.5 / 60.7。RVS-Ego / Movie 上，ReKV 带卸载 63.7 / 54.4，不带卸载 55.8 / 50.8，StreamMem 57.6 / 52.7，InfiniPot-V 57.9 / 51.4。即在流式 QA 上，**无界原始 KV + 检索仍明显优于有界的 query-agnostic 压缩**（RVS-Ego 约 +6 点）— [StreamMem HTML](https://arxiv.org/html/2508.15717v1)
- SimpleStream（Shen et al., 2026-04）：只喂最近 4 帧给现成 VLM，OVO-Bench 平均 67.7%、StreamingBench 80.59%，与 13 个离线 / 在线基线对比后持平或更好。作者报告：增加历史上下文可以提升 recall，但常常削弱实时感知；长上下文的价值依骨干而定，不随规模单调；主张记忆 / 检索 / 压缩模块必须在同协议下明显超过 SimpleStream 才算进展，并建议基准把近场感知和长程记忆分开评测 — [SimpleStream abs](https://arxiv.org/abs/2604.02317)
- 2025–2026 同领域参考（只简列）：
  - StreamKV（2025-11）：分段 KV 检索 + 压缩 — [arXiv 2511.07278](https://arxiv.org/abs/2511.07278v1)
  - MuKV（CVPR 2026）：多粒度 KV 压缩，在线阶段半层次地检索相关 KV — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/39020)
  - DSCache（2026-05）：在 OVO-Bench / StreamingBench 上评测 — [arXiv 2605.01858](https://arxiv.org/pdf/2605.01858)
  - SAVEMem（2026-05，training-free 视觉记忆）：检索摘要称其在 OVO-Bench Backward Tracing（631 题）7B 上平均 50.44，hallucination detection +13.97。此条来自搜索摘要，我没打开全文核实 — [arXiv 2605.07897](https://arxiv.org/pdf/2605.07897)
  - V-Rex（HPCA 2026）：KV 检索的软硬件协同 — [arXiv 2512.12284](https://arxiv.org/pdf/2512.12284)
- 小模型回读的编造风险：ReadAgent 的影响声明写道，gist 里“many details are elided”，模型可能自己补出缺失细节 — [ReadAgent HTML](https://arxiv.org/html/2402.09727v3)；LongMemEval 中 Llama 8B 做时间扩展会幻觉时间线索 — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)

### Inferences
- **ReKV 的 oracle 差距（7B：56.0 vs 64.4，而 0.5B 的 internal 已接近 oracle）说明 7B 的瓶颈一半在检索、一半在“看着正确帧也答不对”。**我们 caption 路线已证明读取端没有余量；视觉 KV 路线等于换了一个读取端，上限由 oracle 决定。对 OVO-Bench memory 题，合理的预期是个位数点的提升，不会消掉 44% 的 absent。
- **存储成本（我的推算，按 Qwen2.5-7B / Qwen2-7B 的 28 层、4 KV 头、head_dim 128、bf16）：**每 token 的 K+V ≈ 56 KB。ReKV 的 353K token/h 约合 20 GB/h，GPU 2、3 上不现实，只能卸载到 CPU / 磁盘。更实际的做法是存**原始帧（JPEG）或视觉编码器输出**，问答时对 top-k 片段重新 prefill：存储近乎为零，代价是每题多几百到几千视觉 token 的 prefill。Qwen2.5-VL 的 M-RoPE 也让跨位置复用旧 KV 更麻烦，重编码可以绕开这个问题（推断，未验证）。
- **gate 路由能隔离“历史伤害实时感知”。**SimpleStream 的副作用出现在“所有题都喂历史”时；只给 gate 判为 past 的题喂检索帧，就不会污染 real-time 题。但 gate 的误判会把这种伤害带回来，所以必须报告 gate precision 与“误路由题”的损失。
- **编造风险的来源很可能是“先生成证据文本、再作答”。**ReKV / StreamMem 都是把检索到的视觉 KV 直接放进上下文回答 MCQ，不生成中间证据。建议改成“检索帧 + caption 时间线 → 直接 MCQ”或者对选项做似然打分，不要让 7B 自由描述后再推理。这是推断；我们之前的回看具体是怎么提问的，会影响这个判断能否成立。
- **ReKV 显示无关帧过多会伤害作答（MLVU 饱和）。**检索 top-k 要小，最好按时间聚到 1–2 个片段。

### Gaps
- 没找到 ReKV / StreamMem / InfiniPot-V 在 OVO-Bench Backward Tracing（EPM / ASI / HLD）上、在同一 7B 骨干下与 SimpleStream 同协议对比的公开数字。SAVEMem 的数字未核实。
- 没找到任何工作报告“检索帧回看”时 7B VLM 的编造率（fabrication rate）。
- MuKV、DSCache、StreamKV（2511.07278）的具体数字本次没有取到。

---

## Q4. 每个想法：具体的 training-free 适配、针对哪个失败桶（absent / ambiguous / fabrication）、成本、诚实风险

### Takeaway
文献里唯一既有一致证据、又能直接对准 absent 桶的方向，是“caption 当 key / 索引，原始视觉当 value，问答时懒回读”（LongMemEval K=V+fact、Zep、ReadAgent 的模式）。时间感知的检索限定是对付 ambiguous 桶最便宜的一步。各种“更好的 query-agnostic 写入”（代理 query、多假设 prompt、重建打分）都无法绕过 1–3% 的跨模态压缩率，预期收益低。所有涉及回读的方案都必须先解决 7B 的编造问题，这件事没有文献担保。

### Cited Findings
（证据均已在 Q1–Q3 给出，这里只列直接支撑各想法的数字）
- Key / value 解耦：K=V+fact 使 recall@k +9.4%、QA +5.4% — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)；保留原始 episode 的 Zep 比 full-context 高 8.4 / 11.0 点 — [Zep HTML](https://arxiv.org/html/2501.13956v1)；gist + 查原文 86.83 vs gist-only 77.52 — [ReadAgent HTML](https://arxiv.org/html/2402.09727v3)
- 时间限定：time-aware 扩展让时间推理 recall +6.8% 到 +11.3%，但 8B 做扩展会让 recall 下降 — [LongMemEval HTML](https://arxiv.org/html/2410.10813v2)
- 视觉检索召回：QAEgo4D 上 external 58.1%、internal 70.5%（7B），准确率 +1.2 到 +3.0，oracle 64.4 — [ReKV HTML](https://arxiv.org/html/2503.00540v1)
- query 无关的多样性打分在 agnostic 下最稳（KeyDiff 31/36）— [Audit HTML](https://arxiv.org/html/2607.11942v1)
- 代理 query 与真实 query 只差 1.2 点（MLVU），multi-detail 差 3.4 点 — [StreamMem HTML](https://arxiv.org/html/2508.15717v1)
- 延迟压缩能回收大部分差距 — [Online KV Compaction abs](https://arxiv.org/abs/2608.00902)
- 历史上下文伤害实时感知 — [SimpleStream abs](https://arxiv.org/abs/2604.02317)

### Inferences
下表是我的迁移判断，不是文献结论。“成本”按我们的设置（7B，GPU 2、3）粗估。

| # | 想法（文献来源） | training-free 适配 | 目标桶 | 成本 | 诚实风险 / 预期 |
|---|---|---|---|---|---|
| 1 | **Key / value 解耦 + 懒回读**（LongMemEval K=V+fact、Zep 的非损 episode、ReadAgent 查页、Letta 原始文件 + 检索） | 每 16 s 片段另存 1 fps 的 JPEG（或 SigLIP / CLIP 嵌入 + 帧）。caption（加时间戳、加抽取的实体名词）**只当索引**。gate 判为 past 时，按 caption 检索与 SigLIP 图文检索的并集取 top-1~2 片段，把原始帧（例如 8 帧）+ caption 时间线一起送进模型**直接答 MCQ**，不加最近 4 帧，或放在最后 | absent（主）、ambiguous（次） | 存储可忽略；每道 past 题多 1 次约 8 帧的 prefill（约 1–2k token），延迟约 +1–2 s | ① absent 题的 caption 里本来就没有该实体，caption 索引必然失败，只能指望 SigLIP 图文检索，而 ReKV 外部检索 recall 也只有 58%；② 7B 看着正确帧也答不对（ReKV oracle 天花板）；③ 编造：我们已见过。预期提升为个位数点，必须对照 SimpleStream 和“随机片段”对照组 |
| 2 | **时间感知检索限定**（LongMemEval time-aware query expansion） | 用规则 / 正则（不要用 7B 生成）解析题干中的时间短语（“at the beginning”“before X”“N minutes ago”），只把对应时间窗的 caption（及回读帧）给模型；无时间线索的题不改 | ambiguous | 几乎为零 | 先做一个廉价诊断：在 ambiguous 桶里统计干扰项词出现在**时间上不相关**的 caption 中的比例。比例低则此法无效。OVO-Bench 的干扰项可能本就来自同一片段，此时需要的是关系 / 次序信息，限定时间窗没用 |
| 3 | **选项条件的判别式回读**（针对编造；ReKV / StreamMem 都是直接 MCQ，不生成证据） | 把“描述证据 → 推理”改为：对每个选项，在检索帧上计算 yes / no 或选项 token 的对数似然，做对比打分；再用“有回读 vs 仅 caption”的答案一致性 / 置信度门控，低置信时回退到仅 caption 的答案 | fabrication（主）、ambiguous | 每题 4 次短前向或 1 次带 4 选项的前向 | 7B 有 yes 偏置、受选项长度影响，需要做校准（例如减去无图条件下的似然）。文献里没有 7B VLM 回读编造率的数据，属于无担保的工程尝试 |
| 4 | **新颖性 / 多样性的视觉 token 存储**（KeyDiff 在 agnostic 下最稳；InfiniPot-V / StreamMem 在 6K 预算下接近 full KV） | 存储侧不按“caption 注意力”选帧（等同代理 query，会和 caption 盲区重合），而是按 SigLIP 嵌入的去冗余或最远点采样选关键帧 / 视觉 token，作为想法 1 的 value | absent | 低 | 能提高“细节被存下来”的概率，但不解决检索与读取问题。直接存 KV 约 56 KB / token（推算），不划算；存帧更好 |
| 5 | **延迟写入**（online KV compaction：immediate compaction hurts） | 对最近 1–2 分钟（超出 4 帧窗口、但尚未被 caption 固化的片段）保留原始帧；提问时对这段做 **query-aware** 的即时重看或描述 | absent（仅限近期） | 低到中 | 只覆盖证据落在缓冲窗口内的题。OVO-Bench 的 memory 题多数指向更早的时间，收益有限 |
| 6 | **重建目标 / 多假设写入**（KVzip、TaskPress meta-query、StreamMem 代理 query） | 写入时用多个通用问题模板（谁 / 什么物体 / 在哪 / 先后）生成多条 caption，或用“caption 能否回答自动生成的自问题”来筛 caption | absent | 写入成本 ×K | **低预期**：我们已证伪 memory-oriented prompt 与实体图有效。代理 query 与真实 query 的差距（StreamMem multi-detail −3.4、SnapKV Δ≈0.2）说明代理无法覆盖题目分布；1–3% 的跨模态压缩率下没有成立的证据。只建议作为写入覆盖率诊断 |
| 7 | **全量视觉 KV 检索（ReKV 式）作为上界参考** | 不加 gate，存全部视觉 KV（CPU 卸载），internal retrieval | absent | 约 20 GB/h（推算），延迟 +0.4 s 以上 | 适合当上界和对照组。SimpleStream 指出历史上下文伤害实时题，须配合 gate 使用 |

- **整体判断（推断）：**跨领域文献给出的“写入瓶颈”统一答案是：别指望一次 query-agnostic 写入就把信息写对，要保留非损的原始数据，把精力放在索引和按需读取上。我们的数据（比例对写入端各种变量都不变）与这个结论吻合。但这条路在我们的设置里转成了**读取端**问题，而 7B 回读编造正是我们已撞上的墙。现实的论文叙事可能是“写入瓶颈是信息论意义上的；跨领域证据也支持非损存储 + 懒读取；在 7B 上这条路的上界受 oracle 读取限制”。不宜宣称新方法能大幅提升。
- **建议的最小实验顺序（按成本）：**② 时间相关性诊断（零 GPU）→ 想法 1 的 oracle 版（用 GT 时间戳取片段，测“看着正确帧 7B 能否答对”，直接给出回读路线的天花板）→ 想法 1 的检索版 + 想法 3 的判别式作答。如果 oracle 版在 absent 桶上也没有明显提升，就可以停掉整个回读方向。

### Gaps
- 没有任何公开工作在 OVO-Bench memory 子集上给出“oracle 片段回读”的 7B 准确率；这需要我们自己测。
- 选项似然打分在 7B VLM 视频 MCQ 上的校准效果，本次没有检索到直接证据。
- Qwen2.5-VL / Qwen3-VL 的 M-RoPE 下跨时间复用已存 KV 的正确性，本次没有查到文献，属于未验证的工程假设。
