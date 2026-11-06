# 先读这十篇，再动笔写 ReadPast

写 ReadPast 的 TCSVT 稿之前，要读的论文分三组，约十篇。**第一组是同刊范本**：TCSVT 至今没有一篇流式视频 LLM 论文，最接近的是两篇免训练长视频论文。**EventLens**（TCSVT 2026）做了 5 个基准 × 7 个骨干，骨干含 Qwen2.5-VL-7B 和 Qwen3-VL-8B；**FDVS**（TCSVT 2025）把长视频写成分层文本"故事"。章节骨架和效率小节从这两篇和 ResPrune、PTP 里借。**第二组是流式同行**：SimpleStream 是基座；D-HSM 是弃答指令的出处，也是主对比对象；**StreamScout 免训练版的写入设计与 ReadPast 逐项相同**（同一骨干每 16 秒写一条字幕、近 4 帧作答），但没有代码，只能引用。**第三组是三个模块的谱系源头**：LLoVi / LangRepo（字幕即记忆），Mallen 等与 RetrievalQA（只看问题决定是否取记忆），Zhao 等的 contextual calibration（ReadPast 的 p_cf 正是它的三输入概率平均）。读完之后，定位就清楚了：写入端整体是沿用；ReadPast 能主张的，是读取端两条规则的组合——只看问题、无标签校准的现在/过去门，加上按选项条件触发的弃答指令——以及在 4 个骨干上相对 SimpleStream **+6.03 / +5.68 / +5.18 / +2.45** 的增益（Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3-VL-4B / LLaVA-OV-7B，标准 OVO 70.78 / 73.34 / 72.60 / 61.60）。"按固定间隔、不看问题地写字幕并事先缓存"在两条文献线上都是常规做法：离线长视频文献有 LLoVi（1 s 一条）、VideoAgent（2 s 一条），LangRepo 明确写了跨问题复用；流式同行里 OneStreamer、EventMemAgent、MaLoW 都预计算，D-HSM 在 StreamingBench 上按视频缓存。各家写法真正拉开差距的地方，是有没有报告写入开销；目前只有 StreamScout（每 16 s 一条，耗时 4.2 s）和 FOLIO（每 8 s 一段，耗时 5.88 s）给了数字。

> 口径：下文 OVO 一律为 (RT + BT) / 2，不含 Forward，按各论文表格重算。"代码"指 GitHub 文件树里确有推理代码。没有代码的方法只引用，数字标为论文值。所有调研截至 2026-11-03。

## TCSVT 还没有流式视频 LLM 论文，EventLens 和 FDVS 是最近的结构范本

先说空位。调研用 Crossref 和 OpenAlex 对 TCSVT 做了多组检索（streaming、online video、long video、training-free 等关键词，合并去重 1544 条）。2024 年以来，标题含 "streaming" 的 TCSVT 论文全部属于网络传输、QoE、事件相机或流式感知，**没有一篇是流式视频 LLM 或流式 VideoQA**（[Crossref](https://api.crossref.org/journals/1051-8215/works?query.title=streaming&filter=from-pub-date:2025-01-01&rows=200)；[OpenAlex](https://api.openalex.org/works?filter=primary_location.source.issn:1051-8215,from_publication_date:2024-01-01,title.search:streaming)）。OVO-Bench 本身发表在 CVPR 2025（[OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench)），不在本刊。所以 ReadPast 的章节骨架只能借同刊的免训练长视频论文和推理时 MLLM 插件论文，流式部分的写法借 CVPR / ACL / NeurIPS 的同行。编辑层面的口味倒很明确。TCSVT 开过 "Special Issue on Large Language Models (LLMs) for Video Understanding"，征稿词强调 LLM 的 "versatility ... without extensive retraining"，并把提升效率列为要解决的挑战；该专刊已于 2025-01-02 截稿（[CfP PDF](https://ieee-cas.org/files/ieeecass/2024-12/TCSVT-SI-LLMVU-CfP.pdf)）。"免训练 + 效率"的叙事与本刊一致，cover letter 可以呼应这一点。

**EventLens: Event-Aware Hierarchical Evidence Construction for Long-Video Understanding** 是最该精读的同刊论文（TCSVT 36(11), pp. 17510–17524, 2026）。它是免训练框架，由 semantic boundary parser、query-answer co-retriever、query-guided evidence selector 和 contrastive verdict refiner 四个部件组成。实验覆盖 5 个长视频基准和 7 个 MLLM，骨干包括 Qwen2.5-VL-7B/32B、Qwen3-VL-8B/32B、LLaVA-Video-7B 和 InternVL2.5/3-8B；LVBench 最高 +8.1%，VideoEval-Pro 最高 +9.2%（[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2026.3723106)；[GitHub](https://github.com/zhengkun332/EventLens)）。有三处可以直接借。第一是摘要里的三连式问题陈述：训练式方法 "costly"，免训练检索有潜力，但现有方法是 "flat frame-level retrieval"，并且 "depend solely on one-shot matching"。ReadPast 的引言可以照这个节奏写：SimpleStream 证明近 4 帧已经很强；回看题却拿不到任何信息；已有记忆方法要么要训练，要么要存视觉 token 或 KV。第二是 config 里逐项写明的超参（1 FPS、边界阈值取第 12 百分位、Top-16 事件、帧预算 64 等），这是同刊认可的复现透明度。第三，它的 contrastive verdict refiner 作用在答案端，与我们弃答指令的位置相同。它的规模（7 个骨干，最大 32B）也是同刊的参照点，ReadPast 目前 4 个骨干都在 4–8B。EventLens 没有 arXiv 版，正文章节、消融和效率表都没读到，需要从 [Xplore](https://ieeexplore.ieee.org/document/11653436) 拿 PDF。

**Toward Long Video Understanding via Fine-Detailed Video Story Generation（FDVS）** 是同刊"把视频写成文本再交给 LLM"的直接先例（TCSVT 35(5), 16 页, 2025）。它自底向上把长视频解释成分层文本，经视觉、文本两级冗余消除，"applicable to various tasks without any fine-tuning"（[OpenAlex 摘要](https://api.openalex.org/works/https://doi.org/10.1109/tcsvt.2024.3514820)）。它的实验部分是本刊免训练文本记忆论文的样板。消融共 7 类：采样策略、冗余消除开关、感知模块组合、编码器与 LLM 替换、clip 描述数 K、局部记忆长度 L、每 clip 帧数 4/8/16/32。效率方面，Table X 比较每个视频的存储字节数（FDVS 87.3，Clip4Clip 840.0，减少 89.6%）；Table XVI 报告 FPS，并**主动承认**提取阶段只有 13.22 FPS，慢于基线。局限写在 IV-F 的 "Discussion of advantages and potential limitations" 段落。全文 6 图、19 表、108 篇参考文献（[arXiv](https://arxiv.org/html/2412.06182)）。对 ReadPast 来说，"字节/视频"表可以改写成"每小时视频的时间线 token 数与存储量"；"提取慢"的坦白，对应我们每 16 秒一次的字幕写入开销。Related Work 里要把两者分开：FDVS 是离线、分层、带冗余消除；ReadPast 是流式、只追加、由作答模型自己写。

效率小节和主结果的组织方式，看 **ResPrune** 和 **PTP**。ResPrune（TCSVT Early Access 2026）的主结果按骨干拆成 LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B 三个小节。方法节末尾有 III-F 理论效率分析；实验节另有独立的 IV-D 实测效率，报 FLOPs、KV-cache、显存、吞吐和总评测时间，例如剪枝 66.7% 时 FLOPs 从 30.6T 降到 9.6T。它在结论末段承认，α 是 "fixed, manually tuned hyperparameter per model family"（[arXiv](https://arxiv.org/html/2603.21105)）。PTP（TCSVT 2026，13 页）的 Table V 列出 token 数、每 token 延迟、TFLOPs、显存与 KV-cache，用逐层注意力分析解释"为什么有效"；它的 α 是在报告结果的同一批 benchmark 上扫出来的（[arXiv](https://arxiv.org/html/2509.15704)）。这两篇给了 ReadPast 一个现成的对照：**同刊免训练插件的关键超参要按模型族手调，ReadPast 唯一的阈值不看任何标签**。引言里一句话就能点明。另有两篇可参考：VideoExpert（TCSVT 2026）用一张定性 pilot 图论证 MLLM "rely more on language patterns than visual cues"，作为方法动机（[arXiv](https://arxiv.org/html/2504.07519)）；HeurVidQA（TCSVT 2025）按问题类型讨论失败，局限写在 "Conclusions and Future Directions" 第二段（[arXiv](https://arxiv.org/html/2410.09380)）。Related Work 里引一篇本刊综述 "Video Understanding With Large Language Models: A Survey"（Tang et al., TCSVT, arXiv 2312.17432）也很自然（[arXiv API](https://export.arxiv.org/api/query?id_list=2312.17432)）。

| 范本 | 刊期 / 正式页数 | 训练 | 规模 | 效率怎么报 | 局限 / 失败案例放哪 | 从它身上取什么 |
|---|---|---|---|---|---|---|
| EventLens | TCSVT 2026, 36(11), 15 页 | 免训练 | 5 基准 × 7 骨干（7B–32B） | 正文未得 | 正文未得 | 问题陈述三连、超参明示、多骨干规模 |
| FDVS | TCSVT 2025, 35(5), 16 页 | 免训练 | 3 任务、8 组数据集-任务组合；换 3 种 LLM | 存储字节/视频（Table X）、FPS（Table XVI） | IV-F 讨论段 | 文本记忆先例、7 类消融、坦白提取慢 |
| ResPrune | TCSVT EA 2026 | 免训练 | 9 图像基准 × 3 骨干 | III-F 理论 + IV-D 实测 | 结论末段 | 主结果按骨干分小节、效率小节写法 |
| PTP | TCSVT 2026, 36(6), 13 页 | 免训练 | 13 图像基准 × 2 骨干 | Table V（延迟、TFLOPs、显存、KV） | 无专门章节 | "为什么有效"的分析图 |
| VideoExpert | TCSVT 2026, 36(5), 16 页 | 训练 | 4 基准 × 1 LLM | Table VIII（token、tokens/s、显存） | 失败案例只在正文文字描述 | 动机 pilot 图 |
| HeurVidQA | TCSVT 2025, 35(2), 16 页 | 训练 | 4 数据集 × 1 骨干 | IV-E 复杂度（参数量、GFLOPs） | 结论第二段 | 按题型的失败讨论 |

把这几篇放在一起看，规律很稳定。正式版 12–16 页，参考文献 50–108 篇，多数在 55–90 篇。贡献 3–4 条，最后一条通常是"多骨干、多基准验证"。Related Work 只有 2–3 个子节。实验节的顺序是：Settings → 按骨干分组的 Main Results → 组件消融加至少一个超参扫描 → 独立效率小节 → Case Study。**正文可读的 5 篇都没有独立的 Limitations 章节**，局限只占结论或讨论里的一段（[FDVS](https://arxiv.org/html/2412.06182)；[ResPrune](https://arxiv.org/html/2603.21105)；[PTP](https://arxiv.org/html/2509.15704)；[VideoExpert](https://arxiv.org/html/2504.07519)；[HeurVidQA](https://arxiv.org/html/2410.09380)）。映射到 ReadPast，各部分可以这样安排：

- **主表**：4 个骨干 × 2 个基准，按骨干分组。
- **消融**：三个模块逐一加减；内容无关输入集合（三输入集成，对比单用 "N/A"、""、"[MASK]"）；字幕间隔。
- **效率小节**：门的单次前向、时间线 prefill、写入实时率、时间线 token 数与存储。
- **"为什么有效"**：门的 log-odds 在校准前后的分布。项目里的 fig3 已按 log-odds 重画。
- **局限**：结论前的一段。

## 主对比表只收有代码的方法，StreamScout 和 FOLIO 只能按论文值引用

主对比表只收**免训练、有公开推理代码、骨干与我们重合**的方法。ReadPast 的四个数是 70.78 / 73.34 / 72.60 / 61.60（Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3-VL-4B / LLaVA-OV-7B），我们复现的 SimpleStream 是 64.75 / 67.66 / 67.42 / 59.15。下表列出 RELATED.md 已有的对手，加上这轮新找到的四篇：OASIS、FluxMem、MEMO、HERMES。四篇都已录用、免训练、有代码，而且用的是我们的骨干。

| 方法 | 录用 | 代码 | 重合骨干 | OVO（论文值） | SB-RT | 机制（一句） | 进表要注明 |
|---|---|---|---|---|---|---|---|
| SimpleStream | arXiv 2604.02317，未见录用 | [EvolvingLMMs-Lab/SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream) | Q2.5 / Q3 | 65.13 / 67.70 | 80.59（Q3） | 只喂最近 4 帧 | 基座；我们复现 64.75 / 67.66 |
| D-HSM | EMNLP 2026 Findings | [Oshikaka/D-HSM](https://github.com/Oshikaka/D-HSM) | Q2.5 / Q3 | 70.9 / 71.45 | 82.5 / 83.1 | 六字段观察 → 实体 hub-spoke 图 → 检索 + 近 4 帧 + 弃答指令 | 论文值、其自身协议；同协议重跑进行中 |
| ShallowStream | arXiv 2609.02780，未见录用 | [CURRENTF/ShallowStream](https://github.com/CURRENTF/ShallowStream)（无 OVO/SB 脚本） | Q3 / LLaVA-OV | 69.5 / 62.2 | 78.2 / 75.5 | 浅层 KV 索引 + 只看问题的 logit 门 | 按 REN 决定用论文值 |
| OASIS | CVPR 2026 | [Solus-sano/OASIS](https://github.com/Solus-sano/OASIS) | Q2.5 / Q3 | 59.94 / 67.67 | 70.59 / 78.22 | 2 s 块写事件摘要 → 事件森林，粗到细两阶段作答 | 基线是作者以 0.5 fps 重跑的全量帧 |
| FluxMem | CVPR 2026 | [ShareLab-SII/FluxMem](https://github.com/ShareLab-SII/FluxMem) | Q2.5 | 57.2 | 76.4 | 自适应分层视觉记忆 | BT 为 MemEvo 转引，属二手 |
| MEMO | ACM MM 2026 | [Cecilia384/MEMO](https://github.com/Cecilia384/MEMO) | Q2.5 / Q3 / LLaVA-OV | 只报 RT：66.9 / 76.0 / 66.6 | 78.5 / 83.7 / 73.2 | 语义块 + 实体级索引 + 高分辨率帧回看 | 没有 BT，需要用其代码补跑 |
| HERMES | ACL 2026 Main | [haowei-freesky/HERMES](https://github.com/haowei-freesky/HERMES) | Q2.5 / Q3 / LLaVA-OV | 59.2 / 61.3 / 58.1 | 79.4 / 81.3 | KV cache 作分层记忆 | 官方值经 MemEvo 表 5 转引；LLaVA 为 ShallowStream 重跑 |
| SAVEMem | arXiv 2605.07897，未见录用 | [wuhang03/savemem](https://github.com/wuhang03/savemem) | Q2.5 | 62.7 | — | 三级视觉记忆 + 伪问题库 + 门 | 门阈值按 RT/BT 类别分别设 |
| ObjectStream | arXiv 2607.28312，未见录用 | [DMK041218/ObjectStream](https://github.com/DMK041218/ObjectStream) | Q2.5 | 61.4 | — | 潜在物体作记忆锚点 | — |

数字来源：[OASIS 表 1/2](https://arxiv.org/html/2604.17052)、[FluxMem 表 1](https://arxiv.org/html/2603.02096)、[MEMO 表 1](https://arxiv.org/html/2609.38900)、[MemEvo 主表与表 5](https://arxiv.org/html/2609.36581)，其余见 [RELATED.md](/data/REN/streamkv/RELATED.md)。

逐个骨干看这张表：

- **Qwen3-VL-8B**：ReadPast 的 73.34 高于表里所有方法，依次是 D-HSM 71.45、ShallowStream 69.5、OASIS 67.67、HERMES 61.3。
- **Qwen2.5-VL-7B**：70.78 与 D-HSM 的 70.9 只差 0.1，可以算持平。D-HSM 这个数是论文值、用的是它自己的协议，第十八轮正在同协议重跑。
- **LLaVA-OV-7B**：ShallowStream 的 62.2 高于我们的 61.60。不过两边的 SimpleStream 基线不同（它重跑得 60.3，我们复现得 59.15），相对各自基线是它 +1.9、我们 +2.45（[RELATED.md](/data/REN/streamkv/RELATED.md)）。

另外两点要在表注里写明。MEMO 的基线是 1 fps 全量帧（Qwen3 RT 70.1），远低于 SimpleStream 近 4 帧的 81.4，所以它的 RT 增益不能直接和我们比（[MEMO](https://arxiv.org/html/2609.38900)）。OASIS 相对 SimpleStream 的增益未知。

训练方法放在表下方作参照分组，同样只收有代码的：

- **SelectStream**（NeurIPS 2026）：70.95 / 72.48。我们在 Qwen3 上高 0.9，在 Qwen2.5 上低 0.2（[arXiv](https://arxiv.org/abs/2606.16353)）。
- **ViSS**（arXiv 2610.34926，代码和权重都已放出）：在同样的近 4 帧接口下只做训练、不加记忆，Qwen3-VL-8B 达到 73.20，与我们的 73.34 持平。这正好支撑"免训练达到训练方法的水平"这一表述（[arXiv](https://arxiv.org/html/2610.34926)）。
- **EventMemAgent**（ECCV 2026，代码和 8B 权重已放出）：63.16，数字经 ViSS 表 1 转引（[ViSS](https://arxiv.org/html/2610.34926)；[README](https://github.com/lingcco/EventMemAgent)）。
- **Qwen3-VL-4B**：这一档能对照的只有训练方法。StreamTTT 69.0，它报的 SimpleStream-4B 是 66.06，我们复现的是 67.42（[arXiv](https://arxiv.org/abs/2608.13416)）；MaLoW 69.3（[arXiv](https://arxiv.org/abs/2610.22070)）；OneStreamer 79.65，它训练模型自己写带时间的字幕记忆，数字经 [TwinPast 首表](https://arxiv.org/html/2610.41983) 转引。

| 必引定位（无代码，只用论文值） | 录用 | 训练 | 骨干 | OVO（论文值） | 与 ReadPast 的关系 |
|---|---|---|---|---|---|
| **StreamScout** 免训练版 | arXiv 2609.00291 | 免训练（-S / -R 需训练） | Q2.5 / Q3 / LLaVA-OV-2-8B | 62.67 / 68.38 / 69.05；-R 71.61 / 76.18 | 写入设计相同；读取端由模型自己决定视觉逐级升级 |
| FOLIO | arXiv 2607.13298 | 免训练 | Q2.5 / Q3 | 67.1 / 75.55 | 同一 VLM 每 8 s 写实体中心 JSON；报告了写入开销 |
| WRWS | arXiv 2609.37345 | 免训练 | Q2.5 / Q3 | 63.86 / 65.37 | 检索熵决定检索还是只看近帧 |
| MemEvo | arXiv 2609.36581 | 免训练（VLM 冻结） | Q2.5 / Q3 | 62.9 / 68.75 | LLM 自动搜索"记忆程序" |
| Foresight | arXiv 2610.03123 | 免训练 | Q3 | 65.06 | 孪生 LLM 跑在流前面规划计算 |
| FlashBack | arXiv 2610.01192 | 免训练 | StreamingVLM-7B / Mage-VL-4B | 57.9 / 65.6 | 只看问题的结构化路由，不用阈值 |
| StreamClarify | WACV 2027 | 免训练 | LLaVA-OV-1.5 / InternVL-3.5 / Qwen3-VL | 自建 StreamClarify-QA，不可比 | 作答前二元判断的熵作不确定性门 |
| SPARE | arXiv 2610.14671 | 训练 | Q3 | 77.78 | 训练出的"近帧够不够"门 + 学会弃答 |
| WTI | arXiv 2609.37035 | 训练（SFT + RL） | Q3 底座 | 75.15 | 自然语言记忆条目 + 可回看源视频 |
| EGSD | arXiv 2609.36803 | 训练 | Q2.5 / Q3 | 只抽到 RT 70.0 / 73.4 | 可核验 Event 的增量记忆 |

数字来源：[StreamScout 表 1](https://arxiv.org/html/2609.00291)、[MemEvo](https://arxiv.org/html/2609.36581)、[Foresight 表 3](https://arxiv.org/html/2610.03123)、[StreamClarify](https://arxiv.org/abs/2611.01263)、[WTI 表 1](https://arxiv.org/html/2609.37035)、[EGSD 表 1](https://arxiv.org/html/2609.36803)，其余见 RELATED.md。

**StreamScout 是这一组里必须精读的一篇。** §4.1 原文是 "As the stream unfolds, the backbone itself captions every segment of Δ=16 seconds, forming the textual timeline"，作答时看最近 4 帧；§3.2 把这种写入称为 "query-agnostic stream maintenance"（[arXiv HTML](https://arxiv.org/html/2609.00291)）。也就是说，同一骨干、16 秒一条、与问题无关、近 4 帧作答，这几点和 ReadPast 完全一样。两者的区别全在读取端：StreamScout 让模型看过画面后自己发 token，决定是否从近 4 帧升级到回看 8 帧、再到检索 12 帧；ReadPast 不做视觉升级，门只读问题，阈值用空问题标定，只决定读不读时间线。分数上，在相同骨干上我们比它的免训练版高 +8.1（Qwen2.5）和 +5.0（Qwen3），这些都是论文值之间的比较。方法节和 Related Work 都需要一句明确的区分。

背景引用按四组写，大多已录用、有代码（会议与代码信息见 [MemEvo 表 5](https://arxiv.org/html/2609.36581) 及各仓库）：

- **训练型流式视频 LLM**：VideoLLM-online（CVPR 2024）、Flash-VStream（ICCV 2025）、Dispider（CVPR 2025）、VideoChat-Online（CVPR 2025）、StreamForest（NeurIPS 2025 Spotlight）、TimeChat-Online（ACM MM 2025）、StreamBridge（NeurIPS 2025）、StreamingVLM（ICLR 2026）、LiveCC（CVPR 2025）、WeaveTime（CVPR 2026），以及会议未确认的 MMDuet。
- **免训练 KV / 视觉记忆**：ReKV（ICLR 2025）、InfiniPot-V（NeurIPS 2025，Qwen2.5 上 56.75）、LiveVLM（DAC 2026）、未见录用的 StreamMem，以及上表的 HERMES、FluxMem、MEMO、ShallowStream。
- **文本 / 事件记忆**：D-HSM、FOLIO、StreamScout、OASIS、DeltaStreamer、EventMemAgent、OneStreamer、WTI。
- **近帧基线与"何时取历史"**：SimpleStream、WRWS、FlashBack、SPARE、StreamClarify、ViSS。

训练型流式模型的数字在不同转引来源之间互相冲突，例如 StreamBridge 有 69.68 与 63.4/59.5 两个版本（[Foresight](https://arxiv.org/html/2610.03123)；[MemEvo](https://arxiv.org/html/2609.36581)）。表注里每格只取一个来源，并写明出处。

基准方面：

- **必须引用**：OVO-Bench（CVPR 2025）和 StreamingBench（arXiv 2411.03628）。SimpleStream 的参考文献表把 StreamingBench 记为 ICASSP 2026，但没有从会议页核实。不报 Forward 时，可以援引 SimpleStream、ShallowStream、D-HSM 主表都只用 RT + BT 的惯例。
- **作为引用**：OVBench（CVPR 2025）、SVBench（ICLR 2025 Spotlight）、RTV-Bench（NeurIPS 2025 D&B）、OmniMMI（CVPR 2025）、ODV-Bench（[StreamForest](https://arxiv.org/abs/2509.24871)；[SVBench](https://arxiv.org/abs/2502.10810)；[RTV-Bench](https://arxiv.org/abs/2505.02064)；[OmniMMI](https://arxiv.org/abs/2503.22952)）。
- **TwinPast**（arXiv 2610.41983，有代码）：值得读来写引言动机。它发现 OVO 和 StreamingBench 抽样的历史题里 32% 只靠近帧就够；它的新基准有 350 对视频，最强流式系统的 PairAcc 只有 49.14%（[arXiv](https://arxiv.org/abs/2610.41983)）。

可选阅读：

- **效率类**：StreamingTOM（CVPR 2026，有代码）、Video-MemComp（ACM MM 2026）、StreamAhead、StreamFlow，以及免训练加速综述 arXiv 2610.36327。
- **主动 / 前向类**：PACE（MMM 2027）、ProV-Mem、AskStream、SyncPro。
- **文本记忆但不是流式基准**：CAM，以及 Retrieval-Augmented Online Textual Memory（arXiv 2610.31284）。

另外注意，arXiv 2511.07278 已有一篇方法叫 **StreamKV**（[arXiv](https://arxiv.org/abs/2511.07278)），和我们的工作目录同名，论文里只用 ReadPast 这个名字。

Related Work 的组织可以兼顾本刊"2–3 个子节"的惯例和流式文献的四组划分：

- **II-A 流式视频理解与记忆**：上面四组压成四段。
- **II-B 基于文本的长视频理解**：从 Socratic Models 到 LangRepo，加上 FDVS、EventLens 和本刊综述。
- **II-C 自适应检索、校准与弃答**。

## 三个模块都站在成熟谱系上，定位句该写 "we adopt"

### 字幕时间线：配方来自离线长视频文献，流式里 StreamScout 已有同款

先读下面五篇。它们共同确立了"先把视频写成带时间戳的文字，再让语言模型读文字作答"这一免训练范式。

- **Socratic Models**（ICLR 2023）：为关键帧写字幕，递归总结成带时间戳的 "language-based world-state history"，再用 "{world-state history} Q: {question} A:" 模板作答，把视频理解变成 "reading comprehension"（[ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00598)）。
- **LLoVi**（EMNLP 2024）：在 EgoSchema 上把视频切成 1 s 片段，LaViLa 写字幕、GPT-3.5 作答，字幕器看不到问题（[arXiv HTML](https://arxiv.org/html/2312.17235v3)）。
- **VideoAgent**（Fan et al., ECCV 2024）：为约 2 s 的片段写字幕并存特征，记忆构建与推理是两个分开的算法。作者发现用 "caption–query similarities" 定位比用 "video–query similarities" 更有效（[arXiv HTML](https://arxiv.org/html/2403.11481v2)）。
- **LangRepo**（Findings of ACL 2025）：写入只以字幕为输入、不看问题，并明确写出建好的仓库可以跨同一视频的多个问题复用（[arXiv HTML](https://arxiv.org/html/2403.14622v2)）。
- **DrVideo**（CVPR 2025）：先用固定提示生成与问题无关的初始文档，再按问题增补（[arXiv HTML](https://arxiv.org/html/2406.12846v2)）。

"文本记忆为什么好"的标准论证也出自这批论文。Vamos（ECCV 2024，需训练）在五个基准上发现 "text-based representations consistently achieve competitive performance ... visual embeddings provide marginal or no performance improvement"（[arXiv](https://arxiv.org/abs/2311.13627)）。LangRepo 的动机是 LLM 处理长信息的能力 "gradually declines with input length"（[arXiv](https://arxiv.org/abs/2403.14622)）。反方向的提醒来自 MVU（ICLR 2025）：LLM 方法 "sometimes even with no video specific information" 也能在长视频任务上拿到不错的准确率（[arXiv](https://arxiv.org/abs/2403.16998)）。离线工作里配置最接近 ReadPast 的是 Video-RAG（NeurIPS 2025），它用 "visually-aligned auxiliary texts" 免训练地补充 LVLM 的画面输入（[arXiv](https://arxiv.org/abs/2411.13093)）。可选阅读有 VideoAgent（Wang et al., ECCV 2024）、VideoTree（CVPR 2025）、Video ReCap（CVPR 2024）、LifelongMemory 和 MM-VID。

这条谱系里的 10 个方法，全部是"独立字幕器 → 另一个 LLM 作答"，粒度在 1–5 s 一条。相对离线谱系，ReadPast 能说的差别有三点：同一个冻结 VLM 在线、因果地既写又读；文字只负责过去，最近的帧仍以像素输入；16 s 一条，比离线谱系粗 3–16 倍。但这三点在流式同行里都已有先例。StreamScout 的写入和我们逐项相同（[arXiv HTML](https://arxiv.org/html/2609.00291)）；FOLIO 用同一 VLM 每 8 秒写一段（[arXiv HTML](https://arxiv.org/html/2607.13298)）；D-HSM 用同一 VLM 写结构化观察（[GitHub](https://github.com/Oshikaka/D-HSM)）。所以写入端整体写成沿用，引用离线谱系、StreamScout 和 D-HSM。

> *Converting video into timestamped text that a language model reads at question time is an established training-free recipe for long-video QA [Socratic Models; LLoVi; VideoAgent; DrVideo; LangRepo]. We adopt this recipe for streaming input in the same form as StreamScout: the frozen VLM that answers also writes a caption every 16 s, the text covers only the past, and the most recent frames remain visual.*

### 现在/过去门：属于"检索前只看问题"的一支，无标签阈值是唯一的新处

ReadPast 的门在作答前只读问题文本，决定要不要把整条时间线放进上下文。在 NLP 的"何时检索"研究里，这属于**检索前、只看问题**的一支：

- **Mallen et al.**（ACL 2023）：按问题中实体的流行度决定是否检索。阈值在 75% 的 PopQA 上按关系类型暴力搜索，再在剩下 25% 上评测（[ar5iv](https://ar5iv.labs.arxiv.org/html/2212.10511)）。
- **SKR**（Findings of EMNLP 2023）：把训练题在有检索和无检索两种设置下各答一遍，对照金标分成 known / unknown 两类，再判断新问题属于哪一类（[ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05002)）。
- **Adaptive-RAG**（NAACL 2024）：训练一个小 LM，按问题复杂度在不检索、单步检索和多步检索之间选择（[arXiv](https://arxiv.org/abs/2403.14403)）。
- **TA-ARE**（RetrievalQA 一文提出，Findings of ACL 2024）：在提示里加日期和 4 个示例，不设阈值（[arXiv HTML](https://arxiv.org/html/2402.16457v2)）。

另一支在生成过程中触发。FLARE（EMNLP 2023）在预测出的下一句含低置信 token 时才检索（[arXiv](https://arxiv.org/abs/2305.06983)），DRAGIN（ACL 2024）和 SeaKR（ACL 2025）也属于这一支。Self-RAG（ICLR 2024）训练 Retrieve 反思 token，阈值人工设为 0.2，ALCE 上设为 0（[arXiv HTML](https://arxiv.org/html/2310.11511)）。这条谱系里，阈值或判据只有四种来源：带标注的开发集、带标注的训练题、人工设定、少量示例。**调研没有找到用内容无关输入、无标签地设定检索门阈值的工作。** 立论最好用 RetrievalQA 的一句话："calibration-based methods heavily rely on threshold tuning, while vanilla prompting is inadequate"（[arXiv](https://arxiv.org/abs/2402.16457)）。它还测到，直接问 GPT-3.5 要不要检索时，检索准确率只有 49.3，TA-ARE 把它提到 86.3。SKR 的直接提示变体，本质上就是未校准的 ReadPast 门。多模态方向的代表是 ReflectiVA（CVPR 2025），它用训练出的反思 token 判断是否需要外部知识（[arXiv](https://arxiv.org/abs/2411.16863)）；IEEE TMM 上的 mR²AG 也值得一引（[arXiv](https://arxiv.org/abs/2411.15041)）。

流式同行里，最接近的是 **ShallowStream 的 Query-Logit Gate**。它只看问题文本，设两个单 token 选项，一次前向取 logit 差。阈值在 200 个带合成路由标签的生成问题上，按 95% Wilson 下界标定一次：Qwen3-VL-8B 上 τ_g = 11.0，LLaVA-OV-7B 上 0.875（[arXiv](https://arxiv.org/abs/2609.02780)）。其他几家的做法各不相同：FlashBack 做多类结构化路由，不用阈值；SAVEMem 的阈值按 RT/BT 类别分别设；StreamClarify 用作答前一次二元判断的熵作门（[arXiv](https://arxiv.org/abs/2611.01263)）。

项目内的对比说明，门本身不是增益来源。第九轮把 ShallowStream 自己的门接到我们的时间线上，6 个格子平均 +1.56，我们的门是 +1.38。四个骨干、8 个格子时，它是 +1.38，我们是 +1.19。把空问题校准套到 ShallowStream 的 10-shot 门上，均值 +1.41，在所有组合里最高（[EXPERIMENTS.md](/data/REN/streamkv/EXPERIMENTS.md)）。所以门的合适定位是"无需标注、可替换的轻量组件"。用词宜写 "adaptive memory access" 或 "conditional context inclusion"，再说明它与 adaptive retrieval 是同一个问题。

阈值的直接来源是 **Zhao et al., "Calibrate Before Use: Improving Few-Shot Performance of Language Models"（ICML 2021）**。它的做法是 q̂ = softmax(W p̂ + b)，其中 W = diag(p̂_cf)⁻¹、b = 0，p̂_cf 是 "N/A"、"[MASK]" 和空串三个内容无关输入上概率的平均，作者称之为 "a novel data-free procedure"。在 1-shot SST-2 上，未校准是 66.5，单用 "N/A" / "[MASK]" / 空串分别是 74.2 / 74.5 / 72.9，三者集成是 79.0（[ar5iv](https://ar5iv.labs.arxiv.org/html/2102.09690)）。

ReadPast 的实现与它完全一致（[route.py](/data/REN/streamkv/kfs/route.py)）：

- p_past = P(past) / (P(past) + P(now))；
- p_cf 是三个内容无关问题上 p_past 的平均；
- 判 "past" 当且仅当 p_past ≥ p_cf。

两类情形下，Zhao 的对角缩放就是比较 p/p_cf 与 (1−p)/(1−p_cf)，恰好化为 p_past ≥ p_cf，等价于把对数几率阈值移到内容无关输入的对数几率。同一个式子也可以读成 Holtzman et al. 的 PMI_DC（EMNLP 2021）的二类特例，此时"领域前提"是问题被置空的路由模板（[arXiv](https://arxiv.org/abs/2104.08315)）。所以论文应写 "we set the gate threshold by contextual calibration"，不要包装成新的校准方法。

与 Zhao 真正不同的是校准对象。Zhao、Fei et al.（ACL 2023）和 Batch Calibration（ICLR 2024）校准的都是任务答案的类别先验；ReadPast 校准的是一个控制决策，它的"标签"在任何基准上都观测不到。ReadPast 是零样本、不给示例，偏置主要来自 "past" 和 "now" 两个词的先验以及对话模板。项目里两个骨干的 p_cf 相差 4 个数量级，校准后 BT/RT 题的读取比例却几乎相同：65%/7% 对 63%/6%（[RELATED.md](/data/REN/streamkv/RELATED.md)）。VDD（ACM MM 2025）校准的是 MLLM 在无图输入下的答案分布（[arXiv](https://arxiv.org/abs/2403.05262)）；ReadPast 用的是无问题输入，校准的是一个纯文本的路由决策。

不选 Batch Calibration 的理由一句话就能写清：它用测试批的平均输出估计偏置（[arXiv](https://arxiv.org/abs/2309.17249)），而"过去题"的比例在 BT 与 RT 之间差别很大；内容无关校准每个骨干只需离线跑 3 次前向。Fei et al. 指出内容无关输入本身也可能带偏置，并改用领域内随机词估计偏置（[arXiv](https://arxiv.org/abs/2305.19148)），这正好是做"内容无关输入集合"敏感性消融的理由。

> *Our present/past gate belongs to the family of pre-retrieval, question-only decisions [Mallen et al.; SKR; Adaptive-RAG; TA-ARE], as opposed to triggers fired during generation [FLARE; DRAGIN; SeaKR]. It is neither trained nor tuned on labeled data: its only threshold is set by contextual calibration [Zhao et al.] on content-free questions, which for two classes reduces to comparing p_past with its content-free average. Unlike the query-logit gate of ShallowStream, calibrated on synthetic questions with routing labels, it requires no routing labels at all.*

### 条件弃答指令：文本来自 D-HSM，触发规则最接近 FOLIO 的 "(when available)"

第三个模块的指令文本取自 D-HSM，它的默认用法是给每道题都加（[GitHub](https://github.com/Oshikaka/D-HSM)）。ReadPast 只在选项里已有弃答项时才加。文献里最接近这条规则的是 FOLIO 附录 K.3 的 "prefer 'Unable to answer' (when available) over guessing"（[arXiv HTML](https://arxiv.org/html/2607.13298)）。

弃答方向的奠基文献有三篇：

- **SQuAD 2.0**（ACL 2018）：要求系统 "determine when no answer is supported by the paragraph and abstain from answering"。在 SQuAD 1.1 上有 86% F1 的强系统，到 2.0 上只有 66%（[arXiv](https://arxiv.org/abs/1806.03822)）。
- **Kamath et al.**（ACL 2020）：只靠 softmax 概率的弃答策略表现差。训练一个校准器后，在 80% 准确率下能作答 56% 的题，直接用概率只能作答 48%（[arXiv](https://arxiv.org/abs/2006.09462)）。
- **Whitehead et al.**（ECCV 2022）：把 VQA 在 1% 风险下的覆盖率从 6.8% 提到 15.6%（[arXiv](https://arxiv.org/abs/2204.13631)）。

与我们的规则形成对照的是 **MM-UPD**（ACL 2025）。它的 Instruction 设定原文是 "If all the options are incorrect, answer F. None of the above."，而且 "these additional options and instructions are also added to the questions in standard scenarios to make a fair comparison"。效果因模型而异：LLaVA-OV-7B 和 InternVL2-8B 从 Option 设定换到 Instruction 设定后，UPD 准确率反而下降（[arXiv HTML](https://arxiv.org/html/2403.20331v4)）。可选的补充文献有四篇：Kadavath et al. 发现大模型仍 "strongly biased against" 选 none-of-the-above（[arXiv](https://arxiv.org/abs/2207.05221)）；Tam et al. 记录 28 个 LLM 在 NA 为正确答案时准确率一致下降 30–50%（[arXiv](https://arxiv.org/abs/2503.01550)）；R-Tuning（NAACL 2024）代表训练式拒答（[arXiv](https://arxiv.org/abs/2311.09677)）；AbstentionBench（NeurIPS 2025）覆盖 20 个数据集（[arXiv](https://arxiv.org/abs/2506.09038)）。

写这个模块时要如实交代覆盖面。在我们的 OVO 题集里，选项含弃答项的 187 题是全部 186 道 HLD 题加 1 道 EPM 题；StreamingBench 没有这类选项，所以这条指令在 StreamingBench 上从不触发。第十八轮消融比较了"只在有弃答选项时加"和 D-HSM 的"每题都加"。条件触发减去每题都加，OVO 标准总分差为 **−0.52 / +1.75 / +1.24 / +0.30**（Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3-VL-4B / LLaVA-OV-7B），4 个骨干里 3 个是条件触发更好。Qwen2.5 是例外，原因是每题都加时它的实时题上涨了（[EXPERIMENTS.md](/data/REN/streamkv/EXPERIMENTS.md)）。这与 MM-UPD 观察到的"统一加指令会伤害部分模型"方向一致。这个模块的新意只是一条部署规则，正文写 "we adopt"，再给出这组消融即可。

> *Abstention is classically framed as selective prediction [Kamath et al.; Whitehead et al.]. ReadPast does not learn an abstention policy: it adopts the "unable to answer" instruction of D-HSM and exposes it only when the option set already contains an abstention choice, akin to FOLIO's "(when available)" rule, rather than adding it to every question as in D-HSM and MM-UPD.*

## "按固定间隔写好字幕再评测"在离线和流式两条线上都有先例，差别在是否报告写入开销

离线长视频文献里，"固定间隔、不看问题地写字幕，再由语言模型读字幕作答"是标准步骤。但明确写出"建一次、跨问题复用"的只有 LangRepo（§5.2.0.3）；其余论文只是在结构上隐含这一点：字幕不依赖问题，因此可以缓存。所以论文里宜写 "captions do not depend on the question and can therefore be cached"，不宜写"这些论文都缓存了"。

| 方法 | 每条字幕覆盖 | 字幕器 → 作答者 | 与问题无关 | 明说跨问题复用 |
|---|---|---|---|---|
| Socratic Models（均匀版） | "fixed time intervals"，间隔未给 | CLIP + GPT-3 → GPT-3 | 是 | 未说明 |
| LLoVi | 1 s（EgoSchema） | LaViLa → GPT-3.5 | 是 | 未说明；3 分钟视频写完全部字幕要 22.36 s |
| VideoAgent（Fan） | 约 2 s | LaViLa → GPT-4 | 是（构建与推理分开） | 结构上隐含 |
| LifelongMemory | 2 s，每条 4 帧 | LaViLa → GPT-3.5/4 | 字幕是；之后按问题过滤 | 未说明 |
| Video ReCap | 片段 4 s，再逐层汇总 | 训练过的 ReCap → GPT-3.5 | 隐含是 | 未说明 |
| DrVideo | 2 s / 5 s | LaViLa / LLaVA-NeXT → GPT-4 / DeepSeek | 初始文档是，之后按问题增补 | 未明说 |
| LangRepo | 预提取的 LaViLa 短片段字幕 | LaViLa → Mistral-7B / Mixtral | 是 | **是** |

出处：[Socratic Models](https://ar5iv.labs.arxiv.org/html/2204.00598)、[LLoVi](https://arxiv.org/html/2312.17235v3)、[VideoAgent](https://arxiv.org/html/2403.11481v2)、[LifelongMemory](https://arxiv.org/html/2312.05269v3)、[Video ReCap](https://arxiv.org/html/2402.13250v6)、[DrVideo](https://arxiv.org/html/2406.12846v2)、[LangRepo](https://arxiv.org/html/2403.14622v2)。

另一支先按问题选帧，再写字幕，例如 VideoAgent（Wang）、VideoTree 和 Socratic Models 的检索版（[VideoTree](https://arxiv.org/html/2405.19209v3)；[VideoAgent (Wang)](https://arxiv.org/html/2403.10517v1)）。在流式设定下，问题到达时过去的帧已经不在了，所以按问题写字幕做不到，与问题无关的写入是流式设定的必要条件，不是图省事。这句话放在方法节里，正好说明 ReadPast 与离线谱系的本质差别。

流式同行在评测时的做法分成三类：

- **按题重放或按视频在线处理**：DeltaStreamer、OASIS。
- **预计算并缓存**：OneStreamer、EventMemAgent、MaLoW，以及 D-HSM 在 StreamingBench 上的做法。
- **提问时才建记忆**：D-HSM 在 OVO 上的做法。

| 方法 | 写什么 / 谁写 / 间隔 | 是否看问题 | OVO / SB 评测时怎么生成 | 写入开销 |
|---|---|---|---|---|
| StreamScout（无代码） | 自由字幕，骨干自己写，Δ = 16 s | 否（"query-agnostic stream maintenance"） | 论文写 "As the stream unfolds"，无代码，无法核实 | **每 16 s 窗口一条字幕耗时 4.2 s**，约合每分钟视频 17.6 s GPU 时间，实时余量 3.4× |
| FOLIO（无代码） | 实体中心 JSON，同一 VLM，固定 8 s 一段 | OVO 单轮，否 | "writes memory online from the observed prefix"，是否预计算未写明 | **每段 5.88 s（OVO）/ 7.68 s（SB）**，首 token 6.58 / 7.31 s |
| D-HSM（有代码） | 六字段观察，同一 VLM | 通用提示；计数题换成针对问题的提示 | **OVO：提问时用 ≤20 个均匀采样的 1 s 片段建记忆；SB：按视频缓存字幕，跨题复用** | 论文：每片段写观察 1.55 s，每题作答 0.74 s（A100） |
| OneStreamer（训练，有代码） | Observe + Summary 记录，模型自己写，每秒更新 | 否（"without access to the current question"） | **预计算并缓存**到题目时间点，可按流共享时间线 | 未核实 |
| EventMemAgent（训练，有代码） | 事件字幕，另一个冻结 Qwen3-VL-4B 写 | 未核实 | **离线预处理**，直接读 HF 上的 memory parquet | 未核实 |
| MaLoW 文本摘要变体（训练，有代码） | 带时间戳的结构化笔记 | 否（"question-independent"） | 顺序回放生成，提供**预计算字幕下载** | 文档承认 "time-consuming" |
| DeltaStreamer（有代码） | 每采样帧写场景描述 + 变化 | 提问前看不到问题 | 按题、按流顺序逐轮重放；另有 cached 模式恢复提问前的记忆 | 按样本记录耗时 |
| OASIS（有代码） | 2 s 块事件摘要，MLLM 写 | 摘要与问题无关 | 按视频、按流顺序在线处理 | 写入延迟未核实 |

出处：[StreamScout](https://arxiv.org/html/2609.00291)、[FOLIO](https://arxiv.org/html/2607.13298)、[D-HSM](https://github.com/Oshikaka/D-HSM)、[OneStreamer](https://github.com/MCG-NJU/OneStreamer)、[EventMemAgent](https://github.com/lingcco/EventMemAgent)、[MaLoW](https://github.com/dragonlzm/MaLoW)、[DeltaStreamer](https://github.com/BBBBchan/DeltaStreamer)、[OASIS](https://github.com/Solus-sano/OASIS)。

这对 ReadPast 意味着两件事。

**第一，缓存的字幕和在线写出的内容相同。** 项目记录里的写入方式是因果的：每段只看本段 8 帧和之前的字幕，写入提示不含问题，作答时只用 t1 ≤ 提问时刻的字幕。所以事先写好只是缓存，内容和边播边写完全相同。OVO 和 StreamingBench 本身也是在提问时刻截断视频来模拟流式（[EXPERIMENTS.md](/data/REN/streamkv/EXPERIMENTS.md)）。

**第二，有一处与真在线不一致，要单独交代。** 离线评测假设第 [t0, t1] 段的字幕在 t1 时刻立即可用，在线写一段却要几秒。按 OVO 1468 题的提问时刻估算，写入延迟为 1 / 2 / 3 / 5 秒时，分别有 96（6.5%）/ 189（12.9%）/ 286（19.5%）/ 439（29.9%）题拿不到最新一段字幕（同上）。

报告写入开销的先例只有两家：StreamScout 是每 16 s 一条、耗时 4.2 s；FOLIO 在 OVO 上每 8 s 一段、耗时 5.88 s，在 StreamingBench 上 7.68 s。按我们的换算，FOLIO 的写入实时率在 OVO 上约 0.74，在 StreamingBench 上约 0.96，几乎贴着实时上限。D-HSM 在 OVO 上是提问时才建记忆，所以它日志里的每题耗时不是流式延迟，和我们的数字放在一起比较前要先统一口径。第十九轮方案里的"写入实时率 + 每题延迟 + 在线回放修正分数"正好补上这一块。实现细节一节可以这样写：

> *Following common practice in LLM-based video QA [LLoVi; VideoAgent; LangRepo] and in streaming memory methods [OneStreamer; D-HSM], captions are written at a fixed 16-s cadence without access to any question. Each caption depends only on the frames of its own segment and on the preceding captions, so precomputing and caching them yields exactly the timeline an online writer would produce; we report the writing cost separately (Sec. IV-F).*

## 结论

这批文献读下来，ReadPast 的贡献边界变清楚了，也变窄了。写入端的每一个设计选择都有先例：同一模型、16 秒一条、与问题无关、近 4 帧作答、事先缓存。StreamScout 甚至与之逐项相同。门的思想与 Mallen 以来的 adaptive retrieval 同构，阈值公式就是 Zhao 的二类情形；弃答指令的文本来自 D-HSM。剩下能立住的有两样：读取端两条无需标注的规则的组合，以及 4 个骨干上稳定的点数增益。后者在 Qwen3-VL-8B 上已经与训练方法 ViSS 持平。把写入端老实写成 "we adopt"，并不削弱论文；反过来，读取端的两条规则和跨骨干的结果会因此更可信。

同刊范本和流式同行的空白恰好互补，这是最值得利用的一点。TCSVT 的免训练插件普遍在局限里承认关键超参要按模型族手调，ReadPast 的阈值完全不看标签。反过来，TCSVT 范本几乎都有独立的效率小节，而流式同行里只有 StreamScout 和 FOLIO 报了写入开销。一张同时给出写入实时率、每题延迟和每小时时间线存储的效率表，既合本刊惯例，在流式文献里也少见。它应当和主表一样，是在动笔前就规划好的部分。

## 先读这 10 篇

| 序 | 论文 | 为什么先读 | 读完带走什么 |
|---|---|---|---|
| 1 | SimpleStream（arXiv 2604.02317） | 基座与评测协议 | (RT+BT)/2 口径、近 4 帧接口、§5–6 的分析写法 |
| 2 | StreamScout（arXiv 2609.00291） | 写入设计与 ReadPast 逐项相同 | §3.2 / §4.1 原句、4.2 s / 16 s 的写入开销写法、方法节里的区分句 |
| 3 | D-HSM（EMNLP 2026 Findings + 代码） | 弃答指令出处、主对比对象 | 指令原文；evaluate_ovo.py 提问时建记忆、evaluate_streamingbench.py 按视频缓存 |
| 4 | EventLens（TCSVT 2026） | 同刊最接近的免训练多骨干范本 | 问题陈述三连、5 × 7 的规模、超参明示（需拿正文 PDF） |
| 5 | FDVS（TCSVT 2025） | 同刊"视频写成文本"的先例 | 存储字节/视频与 FPS 效率表、IV-F 局限段、7 类消融 |
| 6 | Calibrate Before Use（Zhao et al., ICML 2021） | 门阈值的直接来源 | W = diag(p̂_cf)⁻¹ 在二类下化为 p_past ≥ p_cf；Table 3 的内容无关输入对比 |
| 7 | ShallowStream（arXiv 2609.02780） | 最接近的只看问题的门，主表按论文值放入 | Query-Logit Gate 的定义、200 道合成题加 Wilson 准则的标定方式 |
| 8 | OASIS（CVPR 2026） | 已录用、有代码的文本事件记忆对手 | 2 s 块事件摘要、粗到细两阶段、两骨干数字与 0.5 fps 基线注释 |
| 9 | LLoVi（EMNLP 2024）+ LangRepo（Findings of ACL 2025） | "字幕即记忆"与"跨问题复用"的出处 | 固定间隔、不看问题的写入；§5.2.0.3 的复用原文 |
| 10 | RetrievalQA（Findings of ACL 2024）+ Mallen et al.（ACL 2023） | 门控谱系的立论句与源头 | "calibration-based methods heavily rely on threshold tuning"；只看问题的决策要在标注开发集上调阈值 |

第二梯队按需读：

- **写效率小节时**：ResPrune、PTP。
- **主表补行时**：HERMES、FluxMem、MEMO。
- **写引言和相关工作时**：FOLIO（写入开销与 K.3 规则）、MM-UPD（ACL 2025）、Holtzman et al. 的 PMI_DC、ViSS、TwinPast、Socratic Models，以及本刊综述 Tang et al.。

## 带进写作阶段的待核实项

| 待核实 | 影响什么 | 去哪核 |
|---|---|---|
| TCSVT 页数上限与超页费：项目记录里写的是投稿稿 ≤14 页，但 ieee-cas.org 的 node/3112 与 node/3113 说法矛盾；EDICS 类别名称也未拿到（列表页 SSL 失败） | 篇幅规划、投稿类别 | ScholarOne 投稿界面 |
| EventLens 正文：章节、消融、效率表、局限段都没读到 | 最接近的范本 | IEEE Xplore PDF |
| 2026 下半年 TCSVT Early Access 是否已有流式视频 LLM 论文（Crossref / OpenAlex 收录滞后，dblp 被拦） | "本刊首篇"的说法 | Xplore EA 列表 |
| MEMO 缺 OVO BT；FluxMem 的 BT、HERMES 的官方 OVO/SB 都经 MemEvo 转引 | 主表空格与出处注释 | 各自代码补跑，或从原文表格逐项核对 |
| OASIS 的基线为 0.5 fps 全量帧；写摘要的 MLLM 是否就是作答模型 | 主表注释 | 原文与 config |
| 无代码论文（StreamScout、FOLIO、WRWS、FlashBack、SPARE、MemEvo、Foresight）投稿前是否放出代码；StreamFlow、WeaveTime、StreamAgent、StreamMem、LiveCC 的代码未确认；FluxMem 新仓库文件树未核 | 能否进主表 | GitHub |
| StreamScout、FOLIO、WTI、EGSD 评测时字幕是在线生成还是预计算 | 问题 (4) 中先例的强度 | 等代码放出，或联系作者 |
| 无标签内容无关校准的检索门是否已有先例：FLARE、DRAGIN、SeaKR、TARG 的阈值取法未读；mR²AG 是否微调未知；LLM-as-judge 的 yes/no 偏置校准未检索 | 门控唯一差异点的措辞 | Semantic Scholar 用 "contextual calibration retrieval gate"、"label-free threshold adaptive RAG" 再检索 |
| Venue：StreamingBench（SimpleStream 参考文献记为 ICASSP 2026）、MMDuet、StreamMem；Socratic Models、VideoAgent（Wang）、Generative Agents 只有 S2 佐证；LifelongMemory、MM-VID 无正式 venue；FLARE 是 main 还是 Findings；AbstentionBench 的 track | 参考文献表 | 会议页、ACL Anthology |
| 细节：Video-RAG 辅助文本的来源；VDD 的内容无关构造；VideoAgent（Fan）HTML 里印作 "44 帧"；LLoVi、ReCap、LangRepo 的片段参数；MM-UPD 附录中逐设定的 Standard 准确率 | 定位句措辞 | 原文 PDF |
| ICLR 2027 / AAAI 2027 的匿名投稿；10-30 之后的新 arXiv（S2 收录滞后） | 漏引同期工作 | OpenReview、arXiv 检索 |
| 已解决：笔记里担心 p_cf 是对 log-odds 取平均。实际是 p_past 概率的平均，与 Zhao 一致（route.py） | — | — |
