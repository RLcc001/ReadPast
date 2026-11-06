# 认知科学与神经科学的记忆编码/提取理论，及其 AI 实现：与 query-agnostic 记忆写入的关系

> 适用场景：StreamKV，training-free 流式视频问答。每 16 s 窗口写一条通用 caption，问答时读 caption 时间线加最近 4 帧。OVO-Bench 记忆类题（EPM/ASI）按词重叠代理统计：约 44% 记忆中无答案信息（absent），约 44% 正确项和干扰项的词都出现（ambiguous），约 6% 干净（clean）。这些比例不随 writer、分辨率、粒度、记忆导向 prompt、实体图结构而变。其他已知事实：基于 surprise 的保留无效；7B 模型在问答时回看帧会编造证据（confabulation）；原始历史帧进上下文会损害当前感知，文本 gist 不会；文本记忆的收益集中在问题前约 90 s 内。
> 撰写日期 2026-11-02。经典文献的 DOI 链接是按已知书目信息给出的，本次没有逐篇在线核对全文，下文标为“（书目未在线核验）”。AI 论文的数字凡本次读过原文或摘要的，都注明了出处。

---

## 问题 1：哪一种理论能预测我们的模式（query-agnostic gist 存储约 44% 缺失、约 44% 歧义）？人类和这些模型在这种情况下怎么办？

### Takeaway
没有哪个理论能单独解释全部现象，但两个理论合起来几乎能完整预测它。
- 编码特异性 / 迁移适当加工（encoding specificity / TAP）预测 absent 桶：只要编码操作与测验要求的信息不匹配，记忆就会失败，而且不存在一种对所有测验都最优的编码。
- 模糊痕迹理论（fuzzy-trace theory, FTT）加上 Nairne 的“线索诊断性 / cue overload”预测 ambiguous 桶：gist 痕迹会接受与意义一致的干扰项，拒绝干扰项需要 verbatim 痕迹（recollection rejection）。

人类的应对方式不是“写更好的 gist”，而是三件事：
1. 并行保留一份短命的 verbatim 痕迹。
2. 在提取时用测验本身提供的线索（即选项）做 cue-dependent 的再认，而不是自由回忆。
3. 用监控/控制过程（source monitoring、report option、grain size）决定是否采信。

我们“比例对 writer/prompt/结构不变”的结果，正是 TAP 意义上的容量与匹配瓶颈。在写入端换措辞，理论上不该有大的改善。

### Cited Findings

**编码特异性与 TAP（Tulving；Morris, Bransford & Franks）**
- Tulving & Thomson（1973）提出编码特异性原则：提取线索有效，当且仅当它的信息在编码时被编入了记忆痕迹。实验上出现了“可回忆但不可再认”（recognition failure of recallable words）的反常现象。— [Tulving & Thomson 1973, Psychological Review 80(5)](https://doi.org/10.1037/h0020071)（书目未在线核验）
- Morris, Bransford & Franks（1977）表明，“深”加工（语义判断）并不总是更好。测验是押韵再认时，押韵编码胜过语义编码。他们因此提出 transfer-appropriate processing：记忆表现取决于编码加工与测验要求的匹配，不取决于加工“深度”本身。— [Morris, Bransford & Franks 1977, JVLVB 16(5)](https://doi.org/10.1016/S0022-5371(77)80016-9)（书目未在线核验）
- 加工水平理论（Craik & Lockhart 1972）认为语义（深）加工产生更持久的记忆。TAP 就是针对这一点的修正。— [Craik & Lockhart 1972, JVLVB 11(6)](https://doi.org/10.1016/S0022-5371(72)80001-X)（书目未在线核验）

**线索诊断性 / cue overload（Nairne）**
- Poirier, Nairne 等（2012, JEP:LMC, “Memory as discrimination”）做了四个实验。提高编码-提取匹配可以有帮助、无影响，甚至有害，取决于这种改变是否同时改变了线索对目标的诊断性。匹配度提高、诊断性下降时，表现反而变差。作者结论：匹配度不是决定提取效果的因素。— [City Research Online（Poirier/Nairne 等）](https://openaccess.city.ac.uk/id/eprint/2519/)
- 老化研究同样发现：提高匹配度的线索，只有同时改善目标与竞争者的区分时才有帮助，跨年龄组都如此。该研究把“匹配本身不能预测提取”的观点归于 Nairne（2002, 2005）。— [Badham & Poirier, City Research Online](https://openaccess.city.ac.uk/id/eprint/15433/3/1%202016%2007%2025%20AGING%20AND%20MEMORY%20AS%20DISCRIMINATION%20submitted.pdf)
- 该框架下，cue overload、distinctiveness 等因素都通过“诊断性”起作用，人也会优先加工更有区分力的线索。— [Koutmeridou thesis, City 2013](https://openaccess.city.ac.uk/id/eprint/2982)

**模糊痕迹理论（Brainerd & Reyna）**
- FTT 假设 verbatim 与 gist 表征独立地编码、存储和提取，各自以不同速率衰退。— [Reyna, Corbin, Weldon & Brainerd 2016, J. Applied Research in Memory and Cognition 5(1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815269)
- verbatim 随时间变得难以提取，超过一周后真句再认主要基于 gist。gist 相对稳定，所以虚假记忆也能持久。— [Reyna et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815269)
- verbatim 记忆用于拒绝“意义一致但未呈现”的项目，称为 recollection rejection。强化的 gist 反而产生高信心的虚假记忆，称为 phantom recollection。— [Reyna et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815269)
- gist 支持接受相关干扰项，接受率随真假项相似度上升。叙事实验中，基于 gist 的接受率：相似推断 .29，不相似推断 .15。— [Reyna et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815269)
- 提高 verbatim 记忆的手段：学习时重复原文措辞、在真干扰项之前先测验呈现项、使用措辞独特（如隐喻）的句子。先做一次记忆测验能改善延迟回忆，但也会增加对“被测过的相关干扰项”的虚假再认。— [Reyna et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815269)
- 有一段 FTT 表述把核心主张写成：人们分别且大致并行地编码、存储、提取、遗忘 verbatim 和 gist。叙事句研究中，“呈现句的记忆”与“由其得出的推断的记忆”在随机意义上相互独立。— [Helm & Reyna 2023 chapter (PDF)](https://hdpublications.human.cornell.edu/hdpublic/docs/Helm%20&%20Reyna_2023-Fuzzy-trace%20theory-Memory%20and%20decision%20making%20in%20law,%20medicine,%20&%20public%20health.pdf)
- 理论的早期综合见 Reyna & Brainerd（1995）“Fuzzy-trace theory: An interim synthesis”。— [Learning and Individual Differences 7(1)](https://doi.org/10.1016/1041-6080(95)90031-4)（书目未在线核验）

**互补学习系统（CLS）**
- CLS 认为，海马快速学习模式分离的具体情节，新皮层缓慢地抽取统计结构（类 gist/semantic），两者通过回放（replay）巩固。— [McClelland, McNaughton & O'Reilly 1995, Psychological Review 102(3)](https://doi.org/10.1037/0033-295X.102.3.419)；[Kumaran, Hassabis & McClelland 2016, TICS 20(7)](https://doi.org/10.1016/j.tics.2016.05.004)（书目未在线核验）

**AI 中的“gist + verbatim 查原文”实现**
- ReadAgent（Lee et al., Google DeepMind, ICML 2024）：把长文本分成 episode，压缩成 gist memory，任务需要细节时回查原文段落。它在 QuALITY、NarrativeQA、QMSum 上胜过基线，有效上下文扩展 3–20 倍。局限是 gist memory 长度随数据量增长。— [ReadAgent, arXiv 2402.09727](https://huggingface.co/papers/2402.09727.md)；[ICLR 页](https://iclr.cc/virtual/2024/22239)
- HippoRAG（Gutiérrez et al., NeurIPS 2024）以海马索引理论为灵感，用 LLM、知识图谱和 Personalized PageRank 模拟新皮层与海马的分工。多跳 QA 上最多提升 20%；单步检索与 IRCoT 相当或更好，便宜 10–30 倍，快 6–13 倍。— [arXiv 2405.14831](https://arxiv.org/abs/2405.14831)
- HippoMM（2025）把海马式的 pattern separation/completion、短到长期巩固和跨模态联想检索用于连续视听流。在自建的 HippoVlog 上报告准确率 78.2% 对 64.2%，响应时间 20.4 s 对 112.5 s。这是作者自评，没有独立复现。— [arXiv 2504.10739](https://arxiv.org/pdf/2504.10739)

### Inferences
- **absent 桶约 44% 对应 TAP 的“编码-测验失配”**，而且是容量性的失配。一条 16 s caption 只有几十词，问题却可以问任意细节（物体位置、拿了什么、颜色、数量）。query-agnostic 写入等于在不知道测验类型的情况下编码，TAP 直接预测：不存在对所有探测都最优的编码。比例对 writer、分辨率、prompt、实体图都不变，与“瓶颈在于写什么，而不在于写得多好、怎么组织”一致。
- 这也解释了为什么记忆导向 prompt 无效：它通常让模型“更深地”语义加工，比如“记住重要事件”。但 OVO-Bench 的 EPM 探测的是浅层的知觉细节，Morris 等（1977）恰好说明“更深”不等于“更匹配”。这是我的推论。
- **ambiguous 桶约 44% 对应 FTT 加 Nairne**。gist caption 往往同时提到场景里的多个物体或动作。正确项与干扰项的词都在记忆中，说明线索（选项词）匹配到了记忆，但不具诊断性（cue overload）。FTT 说拒绝相似干扰项需要 verbatim 痕迹；只有 gist 时，相似干扰项的接受率会显著升高（.29 对 .15）。这与 7B 模型在 ambiguous 题上难以区分相符。
- **需要核查的代理缺陷**（推论，未核验）：OVO-Bench 的 ASI（动作序列）选项如果是同一组动作的不同排列，那么只要动作被提到，词重叠代理必然判为 both。这种 ambiguous 实际上是“顺序信息缺失”，与 FTT 意义上的歧义不同，应按任务拆开 ASI 和 EPM 再看比例。反过来，gist 常用同义改写（mug 对 cup），词重叠会高估 absent。建议用 LLM-judge 或嵌入相似度做一次复核。
- **人类“怎么办”对应到我们的系统**：
  - 人类并行保留 verbatim 痕迹，类比是存帧或视觉 token。
  - 测验以再认形式给出，选项本身就是 verbatim 线索（Tulving 的 cue-dependent retrieval）。类比是用选项做提取线索去检索存储的帧，而不是让模型自由回看。
  - 采信前做监控（见问题 2）。
  - 我们的负结果（回看编造、原始帧进上下文伤害当前感知）说明前两步在 7B 上缺少可靠的第三步，所以问题被转移到了 source monitoring。
- **“收益集中在约 90 s 内”与 FTT 的“verbatim 衰退快、gist 稳定”只是表面相似**，机制不同：我们的文本记忆本身不衰退。更可能的解释是近期题可以由“最近 4 帧（verbatim）+ 近期 caption（gist）”联合回答，远期题只剩 gist。这是推测，不宜在论文中当作理论预测。
- CLS 的启示：我们现在只有一个“新皮层式”的快速摘要，缺一个模式分离的情节索引。但实体图也没有改变比例，说明仅仅加索引结构（HippoRAG 式）不解决编码缺失。HippoRAG 的增益来自多跳关联检索，我们的读端没有 headroom，所以拟合度低。

### Gaps
- 没有找到在线可读的 Zacks（2020）Annual Review 原文，也没有找到 Brainerd & Reyna 关于“多选题中 gist 与干扰项相似度”的直接实验。多选题的外推是基于 Reyna et al. 2016 的叙事再认数据。
- 没有找到任何把 FTT 或 TAP 显式用于视频 LLM 记忆写入、并有消融的论文。ReadAgent 对 FTT 只是动机层面的类比，论文未引用 Brainerd & Reyna。
- 没有查证 OVO-Bench ASI 的选项构造，需要直接看数据。

---

## 问题 2：source monitoring 研究对减少 confabulation 有什么启示？AI 中是否有方法把它操作化（如检查某个提取到的细节是知觉支持的还是推断出来的）？

### Takeaway
source monitoring 框架（Johnson）认为记忆没有“来源标签”。来源是在提取时根据痕迹特征（知觉细节、时空上下文、认知操作痕迹）用启发式或系统性判断推断出来的。错误来自特征相似和判断标准过松，而“控制”（不报告、放粗粒度）是提高准确率的主要杠杆（Koriat & Goldsmith）。

AI 中已有零散的操作化：
- 写入前做一致性准入（ConsistencyGate）。
- 区分观察和推断的 provenance 分类法。
- 生成后逐条事实核查（CoVe）。
- 用检测器验证视觉断言（Woodpecker）。
- 视频事件幻觉的对比解码（TCD）。

但没有找到“标注观察对推断能显著降低幻觉”的对照实验，也没有用于视频记忆的现成方法。最贴近的 2026 年工作只做诊断（LLM reality monitoring、VLM source-modality monitoring），不提供修复方法。

### Cited Findings

**认知理论**
- Johnson & Raye（1981）的 reality monitoring：外部知觉来源的记忆通常含更多知觉和上下文细节，内部生成（想象、推断）的记忆含更多“认知操作”信息；人据此区分来源。— [Johnson & Raye 1981, Psychological Review 88(1)](https://doi.org/10.1037/0033-295X.88.1.67)（书目未在线核验）
- Johnson, Hashtroudi & Lindsay（1993）的 source monitoring 框架：来源归因是在提取时对记忆特征的判断，而非读取标签。判断有快速启发式和慢速系统性推理两种；标准放松或特征重叠时出错增多。— [Psychological Bulletin 114(1)](https://doi.org/10.1037/0033-2909.114.1.3)（书目未在线核验）
- Schacter 的“七宗罪”把 misattribution（来源错误）、suggestibility、bias 列为构建性记忆的系统性副作用。— [Schacter 1999, American Psychologist 54(3)](https://doi.org/10.1037/0003-066X.54.3.182)（书目未在线核验）
- Schnider（2003）认为自发虚构（spontaneous confabulation）源于不能抑制与当前现实无关的记忆痕迹，即不能把思维校准到“此刻的现实”，与眶额皮层的过滤功能有关。— [Nature Reviews Neuroscience 4](https://doi.org/10.1038/nrn1179)（书目未在线核验）
- Koriat & Goldsmith（1996, Psychological Review 103(3), 490–517）的监控-控制模型：人通过 report option（报告或放弃，包括“不知道”）调节准确率，存在准确率与信息量的权衡。提高报告阈值能提高所报内容的准确率，代价是信息量。— [Koriat & Goldsmith 1996 PDF](https://newiipdm.haifa.ac.il/wp-content/uploads/2015/05/Koriat-Goldsmith-1996-PsychRev.pdf)
- Goldsmith, Koriat & Weinberg-Eliezer（2002）把框架扩展到 grain size：人会策略性地调节答案的精细程度，以粗粒度换准确率。— [Goldsmith et al. 2002 JEP:General PDF](https://newiipdm.haifa.ac.il/wp-content/uploads/2015/05/Goldsmith-et-al-2002-JEPG.pdf)；[Haifa CRIS](https://cris.haifa.ac.il/en/publications/strategic-regulation-of-grain-size-in-memory-reporting/)
- Koriat 和 Goldsmith 的工作中也报告，主观信心对准确率的诊断性有时很差，这正是需要专门监控过程的原因之一。— [Koriat & Goldsmith 1996](https://newiipdm.haifa.ac.il/wp-content/uploads/2015/05/Koriat-Goldsmith-1996-PsychRev.pdf)

**AI 中的诊断与操作化**
- “Reality Monitoring in Large Language Models”（2026 预印本，未经同行评审）把 Johnson & Raye 框架用于 LLM，区分外部给定的“perceived”项和模型自生成的“imagined”项。阅读幻觉（模型把输入词替换成自己的词）是来源误归因的最强预测因子。作者认为部分 source monitoring 缺陷实为编码缺陷。— [arXiv 2607.23927](https://arxiv.org/html/2607.23927)
- “Source-Modality Monitoring in Vision-Language Models”（Hua, Yun, Pavlick；COLM 2026）评估了 11 个 VLM 能否追踪信息来自哪个模态。句法和语义信号都起作用；模态分布差异大时，语义信号压过句法信号。论文只做诊断，不提供改进方法。— [arXiv 2604.22038](https://arxiv.org/abs/2604.22038)
- EventHallusion（Zhang et al., 2024，多版本至 v6）发现 VideoLLM 受语言先验驱动，会“看到”高频事件（如 person drinks coffee），开源模型幻觉更多。他们提出 training-free 的 Temporal Contrastive Decoding：用时间下采样（或时间上模糊化）的视频做对照，在解码时用其 logits 校正原视频 logits，在多数指标上改善。我未核对具体数字。— [arXiv 2409.16597](https://arxiv.org/html/2409.16597v2)
- 一篇 2026 年的 agent provenance 综述主张，记忆库应区分 raw observations、extracted facts、inferred memories、revised memories，并标注每条 provenance 边的来源（instrumented / inferred / declared / derived）。这是设计要求，没有幻觉降低的量化证据。— [arXiv 2606.04990](https://arxiv.org/html/2606.04990)
- ConsistencyGate（2026）在写入记忆前，让 LLM 对候选事实做 K 次查询，平均支持分超过阈值才准入，以防记忆污染。它是“写入时 source check”的实现。— [arXiv 2607.22962](https://arxiv.org/pdf/2607.22962)
- Agent Zero Memory（2026）把“source-locked”原始事实与可修订的推断经验在类型上隔离，目的是不让推断污染事实。这是架构主张，没有对照实验。— [arXiv 2608.29606](https://arxiv.org/pdf/2608.29606)
- 在 YOLOv8 规则校验 VLM 回答的工作中，删除视觉上无支持的信息能提高可靠性，但拒答率上升，与 Koriat-Goldsmith 的权衡一致。— [SSRN 6840298](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6840298)
- EMNLP 2024 的分析发现，grounding 训练目标对开放式 caption 中的物体幻觉几乎没有作用。仅靠“更 grounded 的编码”不能保证不虚构。— [ACL Anthology 2024.emnlp-main.159](https://aclanthology.org/2024.emnlp-main.159.pdf)
- 已知的 training-free 事后核查方法（书目未在线核验）：
  - Chain-of-Verification：先生成，再规划核查问题并独立回答，最后修订。— [Dhuliawala et al. 2023, arXiv 2309.11495](https://arxiv.org/abs/2309.11495)
  - Woodpecker：抽取视觉断言，用开放词表检测器和 VQA 逐条验证后修正。— [Yin et al. 2023, arXiv 2310.16045](https://arxiv.org/abs/2310.16045)
- CAM（2026-09）在实体知识图谱上加入“planner-executor-verifier”管线，按问题意图组合图遍历、视频重看、音频重听。报告相对 SOTA 基线最多提升 23 个百分点；页面未说明是否 training-free，也未说明 verifier 机制细节。— [arXiv 2609.06504](https://arxiv.org/abs/2609.06504)

### Inferences
- **我们的 re-look 编造是典型的“cued confirmation”失败。** 问题和选项本身是强 gist 线索（相当于 DRM 范式中的关键诱饵）。7B 模型带着选项去看帧，相当于在宽松标准下做启发式来源判断，于是“想象到的”被归为“看到的”。source monitoring 框架给出的修复有三条，均为推论：
  1. **把判断从启发式切到系统性**：先盲描述（不给选项），后比对。
  2. **检查知觉特征**：要求给出时间戳和颜色、位置、手等知觉属性，再用第二次独立查询核对。
  3. **收紧标准**：设诱饵对照，测出模型的 yes-bias。
- **Koriat-Goldsmith 的启示是“控制”比“监控”更可靠。** 多选题不能弃答，但可以对“是否采信回看证据”做 report option：证据不过关时退回纯文本答案。也可以在写入端用 grain-size 控制，宁写粗而真（“拿起一个容器”），不写细而虚（“拿起红色马克杯”）。后者降低 confabulation，但会把更多题推入 ambiguous，是一种权衡。
- 我们之前的结论是“gate 不是增益来源”。在 source monitoring 视角下这并不矛盾：监控和控制本身只能减少有害采信，不能创造信息。它们的价值是让 verbatim 回看变得“非有害”，从而有可能吃到 absent 桶的部分题。
- Schnider 的“校准到当前现实”与我们“原始历史帧伤害当前场景感知”的现象，在形式上相似：历史痕迹侵入当前判断。启示是 verbatim 历史帧应放在独立推理通道里，只在被选项线索调用时使用，不常驻上下文。这是类比，不是机制证据。

### Gaps
- 没有找到“标注 observed 与 inferred 能降低 LLM/VLM 幻觉”的受控消融。provenance 相关论文多为设计主张或安全方向。
- 没有找到 source monitoring 思想用于视频记忆 QA（检查历史证据是知觉支持还是推断）并带 benchmark 结果的工作。CAM 的 verifier 最接近，但细节和是否 training-free 不明。
- 没有读到 EventHallusion 中 TCD 的具体增益数字。

---

## 问题 3：基于事件边界的分割（EM-LLM 等）相对固定窗口有提升吗？在哪些任务上？视频上有证据吗？

### Takeaway
- **证据弱，增益小，且集中在检索型任务。**
- 文本：EM-LLM 在检索类任务上增益明显（PassageRetrieval +32.7% 相对），摘要类增益很小，部分任务和模型上不如 InfLLM 固定块。
- 视频：唯一找到的“事件分割对固定 30 s”直接消融（EventMemAgent，RL 训练）在 OVO-Bench 上只差 0.59 分，在 StreamingBench 上差 0.20 分，无方差报告。
- 其他视频工作（Event-VStream、StreamForest、VideoScaffold、Video-EM）主张事件边界更好，但我没找到在同条件下隔离“边界对固定窗口”的数字。
- 认知侧有一个关键的反向预测：边界会削弱跨边界的时间顺序记忆（DuBrow & Davachi），这与 ASI 直接相关。

### Cited Findings

**认知理论（EST）**
- Event Segmentation Theory：事件边界的知觉是“用预测做知觉”的副产品。知觉者在工作记忆中维护“此刻发生什么”的 event model，在边界处更新；按 EST，边界在长时记忆中应有特权地位。— [MIT Open Encyclopedia of Cognitive Science: Event segmentation](https://oecs.mit.edu/pub/p8g1rykx)
- 编码时分割得更可靠的人，日后对该活动记忆更好；改善分割的干预也能改善记忆。— [OECS](https://oecs.mit.edu/pub/p8g1rykx)
- 事件边界之后，上一事件的信息往往更难提取。事件层级嵌套：粗时间尺度的事件可分解为细尺度子事件。— [OECS](https://oecs.mit.edu/pub/p8g1rykx)
- Event Horizon Model：在事件边界处，当前 event model 被存入长时记忆。— [Radvansky & Zacks 2017, PMC5734104](https://pmc.ncbi.nlm.nih.gov/articles/PMC5734104)
- EST 原始表述见 Zacks et al.（2007）。— [Zacks, Speer, Swallow, Braver & Reynolds 2007, Psychological Bulletin 133(2)](https://doi.org/10.1037/0033-2909.133.2.273)（书目未在线核验）
- DuBrow & Davachi（2013）：跨越上下文切换的项目比同一事件内的项目更难排序（chaining 观点：边界削弱项间联结）。— [Wen & Egner 2022 bioRxiv（综述了该发现）](https://www.biorxiv.org/content/10.1101/2022.01.02.474709.full.pdf)；原文 [DuBrow & Davachi 2013, JEP:General](https://doi.org/10.1037/a0034024)（书目未在线核验）
- Wen & Egner（2022）：只有测验时重现编码上下文，跨边界的顺序记忆才可能优于事件内顺序记忆。即提取上下文决定边界是帮助还是损害顺序记忆。— [bioRxiv 2022.01.02.474709](https://www.biorxiv.org/content/10.1101/2022.01.02.474709.full.pdf)
- 边界通过重置时间上下文塑造记忆的时间组织：增强事件内顺序记忆，削弱跨事件顺序记忆。— [PMC8810807](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8810807/)

**预测误差（与 surprise 分割相关）**
- Sinclair, Manalili, Brunec, Adcock & Barense（2021, PNAS）：预测误差改变海马的作用。预期结局下，海马激活越强越预测记忆保持；意外结局下，海马激活越强越预测记忆更新（被修改）。预测误差还打断海马表征的时间连续性。— [bioRxiv 2020.09.29.319418](https://www.biorxiv.org/content/10.1101/2020.09.29.319418.full.pdf)；[Sinclair lab 页](https://sinclairlab.owlstown.net/publications/48155-prediction-errors-disrupt-hippocampal-representations-and-update-episodic-memories)
- 有研究发现，准确的预测与更好的编码相关，预测距离加大会同时损害预测和编码。surprise 并非总是促进编码。— [Duke Scholars（Adcock 组）](https://fds.duke.edu/db/aas/pn/faculty/raa13/publications/362014)（来自搜索摘要，未读原文）

**AI：文本**
- EM-LLM（Fountas et al., Huawei Noah's Ark London 与 UCL；arXiv 2407.09450，v3 修订于 2025-10；通常列为 ICLR 2025，arXiv 页本身未标会议）：
  - 机制：用模型 surprise 定初始边界，再用图论指标（modularity/conductance）细化，检索时加入 temporal contiguity buffer。
  - 人类标注播客对齐实验：surprise 分割比固定或随机分割更接近人类边界；InfLLM 式固定分割（F）低于随机。
  - PG-19 上 modularity（×10⁵，相对随机）：F −2.3±4.1，FM 29.2±44.0，S 18.6±29.6，SM 39.9±55.5，方差很大。
  - 相对 InfLLM 的最大增益（Mistral-7B）：PassageRetrieval +32.69%（64.00→82.67），HotpotQA +10.15%，Musique +6.17%，2WikiMQA +5.20%；摘要类 QMSum +2.13%，GovReport +1.90%，MultiNews −0.30%。
  - 也有负结果：LLaMA-3-8B 上 Math.Find −27.68%；∞-Bench 平均 48.81–49.00 对 InfLLM 50.31；Mistral v2 的 LongBench 平均 42.8 对 43.2。这与摘要中“consistently outperform”的说法矛盾。
  - contiguity buffer 在 44% 的任务上最好，逐任务表现有升有降。
  - 没有视频或多模态实验。
  - 出处：[EM-LLM arXiv HTML v3](https://arxiv.org/html/2407.09450v3)
- ES-Mem（2026-01）把事件分割用于长期对话 agent 记忆，用 LLM 验证候选边界。— [arXiv 2601.07582](https://arxiv.org/pdf/2601.07582)
- Titans（Behrouz et al., 2025）用“surprise”（梯度大小）决定测试时写入神经记忆的内容，但需要训练专门架构，不满足 training-free。— [arXiv 2501.00663](https://arxiv.org/abs/2501.00663)（书目未在线核验）

**AI：视频**
- EventMemAgent（arXiv 2602.15329，2026）：
  - 分割：灰度直方图 Pearson 相关低于 δ=0.2 即切分新事件，事件至少 8 帧。
  - 长期记忆每个事件存（首帧, caption, 文本嵌入, change log）。
  - 基座 Qwen3-VL-8B，使用 GRPO 训练，非 training-free。
  - 消融：固定 30 s 切分对分层事件记忆，OVO-Bench 60.16 对 60.75（−0.59），StreamingBench 76.80 对 77.00（−0.20），无方差。
  - OVO-Bench EPM 59.60、ASI 70.95。
  - 出处：[arXiv 2602.15329](https://arxiv.org/html/2602.15329v1)
- Event-VStream（CVPR 2026 Findings；arXiv 2601.15655）：融合运动、语义、预测线索检测状态转变，只在边界触发语言生成，并把事件嵌入存入持久记忆。相对 VideoLLM-Online-8B，六个子任务平均 28.15 对 17.73（+10.4）。这个对比是对另一个系统，不是对固定窗口的消融。去掉任何边界线索都会降低 caption 质量。我没找到与固定间隔对照的数字。— [arXiv 2601.15655](https://arxiv.org/html/2601.15655v1)
- StreamForest（NeurIPS 2025）：Persistent Event Memory Forest，按事件边界组织树节点，惩罚时间距离、内容相似度和合并频率。— [hyper.ai 摘要](https://hyper.ai/en/papers/neurips__2025__9loSPaBwGO)
- VideoScaffold（2025-12）：认为固定长度 chunk 导致事件边界碎片化、表征冗余或摘要过压缩，提出流式中的 Elastic-Scale Event Segmentation。— [arXiv 2512.22226](https://arxiv.org/html/2512.22226v1)
- Video-EM（arXiv 2508.09486，v2 2026-03）：
  - training-free 但 query-dependent：以 CLIP 检索的关键帧为锚，向两侧扩展到 TransNetV2 镜头边界，用 Qwen2.5-VL-7B 写 when/where/what 叙述，用 Grounding-DINO 追踪物体数量与空间关系变化。
  - 增益：LVBench 38.7→45.7（Qwen2.5-VL），EgoSchema 60.2→64.4。
  - 没有与均匀/固定分割的数值对比；去掉事件扩展分割，EgoSchema 只从 64.4 降到 63.2。
  - 出处：[arXiv 2508.09486](https://arxiv.org/html/2508.09486)
- FOLIO（2026-07，training-free）：短期视觉缓冲加以“受关注实体”为中心的长期语义记忆，并链接到 visual-evidence cache。写入随流进行（作者说未来查询未知），提取时做混合匹配和语义查询扩展。Qwen3-VL-8B 在 OVO-Bench 上 Perception 82.0、Backward 69.1，StreamingBench 74.5。论文未提认知科学灵感。— [arXiv 2607.13298](https://arxiv.org/abs/2607.13298)

### Inferences
- **增益量级**：最干净的视频证据（EventMemAgent，30 s 固定对事件分割）只差约 0.6 分，且有 RL 训练加持。EM-LLM 的增益集中在“大海捞针式检索”任务，即边界让检索单元更内聚，摘要和推理类几乎无益。我们的读端没有 headroom，而且问题出在编码内容而非检索单元，所以事件分割的预期增益很可能小于 1 分（我的判断）。
- **对 ASI 的双刃剑**：DuBrow & Davachi 预测，边界切分会削弱跨事件的顺序信息。固定 16 s 窗口本身就在随机位置切断动作，事件边界至少不会把一个动作切成两半。EventMemAgent 存 change log 的做法，在认知上可以看作“显式写下跨边界的过渡”，用来补偿顺序信息损失。如果 ASI 的 ambiguous 主要是顺序缺失，显式的跨窗口顺序账本可能比改用事件边界更直接。
- **surprise 无效的理论解释**：在人类研究中，预测误差决定的是“什么被更好记住或被更新”，不决定“将来被问什么”。OVO-Bench 的题目并不按 surprise 采样。Sinclair 等还显示，高预测误差会导致更新（覆盖）而非单纯增强。所以“按 surprise 保留”在理论上本就不保证命中查询目标，我们的负结果与此一致。EM-LLM 用 surprise 定边界而非筛选保留，这是两回事。
- Video-EM 这类 query-dependent 方法的高增益（LVBench +7）不能迁移到我们的 query-agnostic 写入设定。它说明的是“知道问题后去取 verbatim 证据”有价值，与 FTT 的结论相同。

### Gaps
- 没能读到 Event-VStream 全文（CVF PDF 返回 403）。arXiv 摘要和片段中没有固定间隔对照的数字。
- StreamForest、VideoScaffold 的边界对固定窗口消融数字未核实。
- 没有找到任何 training-free、query-agnostic、在 OVO-Bench EPM/ASI 上单独报告“事件分割对固定窗口”的结果。

---

## 问题 4：每种理论对应的 training-free 改动、目标失效桶（absent / ambiguous / confabulation）和风险估计

### Takeaway
理论给出的处方高度一致：**不要指望改写 gist 提升覆盖率。** 应当：
1. 并行保留 verbatim 痕迹（帧或视觉嵌入），放在独立通道，不进主上下文。
2. 问答时用选项作为再认线索做检索（cue-dependent recognition）。
3. 用 source monitoring 式的盲描述、知觉一致性、诱饵校准来防止编造。
4. 写入端只做“提高诊断性”的改动（差分/对比式 caption、显式顺序账本），不做“更深”的改动。

鉴于已有负结果，所有写入端改动的预期增益都偏小；最有理论依据的组合是 1+2+3，但它直接撞上我们已观察到的 7B confabulation，风险中高。

### Cited Findings
（各项依据已在问题 1–3 中给出出处，这里只补充 AI 记忆系统的事实。）
- Generative Agents（Park et al., UIST 2023）：memory stream 存观察，检索分数由 recency、importance（LLM 打分）、relevance 组合；reflection 周期性合成更高层的概括。消融显示完整架构（observation + reflection + planning）可信度最高。— [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)（书目未在线核验）
- MemoryBank（Zhong et al., AAAI 2024）：按 Ebbinghaus 遗忘曲线随时间和重要性更新记忆强度。— [arXiv 2305.10250](https://arxiv.org/abs/2305.10250)（书目未在线核验）
- LongMemEval（Wu et al., ICLR 2025）提出三项记忆设计优化：session decomposition（调整 value 粒度）、fact-augmented key expansion（写入时用抽取的事实扩展索引 key）、time-aware query expansion。摘要称这些“greatly improve both memory recall and downstream QA”，但未给出具体数字。商用助手和长上下文 LLM 在持续交互记忆上准确率下降 30%。— [arXiv 2410.10813](https://arxiv.org/abs/2410.10813)
- M3-Agent（ICLR 2026）：以实体为中心的多模态 episodic 加 semantic 记忆，但经 RL 训练。在 M3-Bench-robot/web 和 VideoMME-long 上比最强 prompting agent 高 6.7、7.7、5.3 个百分点，非 training-free。— [arXiv 2508.09736](https://arxiv.org/abs/2508.09736v1)；[ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009685)
- HippoRAG 2（2025）把 HippoRAG 扩展为更通用的“非参数持续学习”记忆。— [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)（书目未在线核验）

### 改动清单（改动设计和风险估计均为我的推论，没有直接实验证据）

| # | 理论来源 | Training-free 改动（具体做法） | 目标桶 | 风险 / 预期 |
|---|---|---|---|---|
| A | FTT（并行 verbatim）+ 编码特异性（cue-dependent retrieval）+ CLS（快速情节库） | **选项线索驱动的 verbatim 检索**。写入时，除 caption 外存每窗口 2–4 帧的 SigLIP/CLIP 嵌入和低分辨率帧（不进上下文）。问答时把每个选项的文本作为线索，各检索 top-k 帧，在一次独立的前向中只看这些帧（不混当前 4 帧），输出“哪个选项有视觉支持”；最终答案仍在主通道生成，只作为附加证据。与之前 re-look 的区别：线索是选项（再认）而非自由问题（回忆），且按选项对称检索。 | absent、ambiguous | 高。之前的 re-look 已编造证据，而对称的按选项检索会给每个选项都“找到”帧，7B 的 yes-bias 可能更强。没有 C 的监控，预期为负。配合 C，乐观估计 EPM +1–3 分，ASI 近 0。 |
| B | Nairne 诊断性 / cue overload + FTT（独特措辞提高 verbatim） | **差分/对比式写入**。写当前窗口 caption 时给出前 1–2 条 caption，要求“只写与之前不同的：新出现或消失的物体、状态变化（开/关、放入/取出）、哪只手、放在哪里、数量”，并保留一条短的场景 gist。目标是让每条记忆对其他窗口更有区分度。 | ambiguous（以及部分 absent） | 中高。记忆导向 prompt 已被证明不改变比例，本改动与之不同之处只在于以过去记忆为条件的“对比”。也可能遗漏持续存在的信息，造成新的 absent。预期 0–1.5 分。用词重叠代理先测 ambiguous 比例是否下降，成本低。 |
| C | Source / reality monitoring（Johnson）+ 监控-控制（Koriat & Goldsmith）+ CoVe/Woodpecker | **回看证据的来源核查**：(1) 盲描述先行：对检索到的帧先不给问题和选项，让模型自由描述，再判断描述是否支持某个选项。(2) 知觉一致性：要求时间戳加知觉属性（颜色/位置/手），用第二次独立查询核对。(3) 诱饵校准：加入一个确定不存在的伪选项（如所有 caption 都没出现的物体），若模型也“找到”它的证据，该题的回看证据作废（类 DRM 关键诱饵、recollection rejection）。(4) report option：不过关就退回纯文本答案。 | confabulation | 中。机制上能减少有害采信，但正如“gate 不是增益来源”，它本身不创造信息。最好情况是让 A 从负变为小正。代价是每题多次前向，盲描述可能漏掉选项相关细节。 |
| D | DuBrow & Davachi（边界削弱跨事件顺序）+ EventMemAgent 的 change log | **显式顺序账本**。每个窗口 caption 以“接续上一窗口最后动作”开头，并维护一个全局编号的动作列表（1. 拿起…，2. 打开…），跨窗口延续编号；问答时读这个账本。 | ambiguous（ASI 中的顺序缺失） | 中。前提是 ASI 的 ambiguous 主要源于顺序未写明（需先核查选项是否为排列）。若是，这是最对症的改动；7B writer 漏动作、乱序会产生错误账本，存在 confabulation 风险。预期 ASI +0–3 分，EPM 无影响。 |
| E | EST（边界驱动编码）+ EM-LLM | **事件边界触发写入**。用帧嵌入余弦下降或直方图相关（EventMemAgent 用 δ=0.2）检测边界，并设最大窗口长度（如 16 s）兜底；在边界处写“before → after”式的 event model 更新。 | ambiguous（动作被固定窗口切断）、少量 absent | 高（对增益而言）。最直接的视频证据只有 +0.6 分（且带 RL），EM-LLM 增益集中在检索任务。适合当作与 B/D 合并的附带改动，不宜单独立项。预期 ≤1 分。 |
| F | 预测误差（Sinclair/Barense；EST 的预测机制） | **用 PE 触发加密编码，而不是筛选保留**：高 PE（嵌入突变）时额外写一条高分辨率细节 caption，并多存几帧 verbatim。 | absent（只在高 PE 时刻） | 高。题目不按 surprise 采样，且我们的 surprise 保留已无效。人类 PE 增强的是“意外项本身”，与 OVO 问的细节未必重合。预期约 0 分，低优先级。 |
| G | CLS 回放 / 巩固 + 再巩固 | **实体触发的回溯标注（retroactive encoding）**：某实体在后续窗口变得显著时（被反复提及、被操作），在空闲时用盲描述回扫早期存储帧，补写该实体在早期出现的位置和状态。仍然 query-agnostic，但由内容触发。 | absent（后来才变重要的实体） | 中高。计算开销大；回扫用盲描述比带线索确认更不易编造，但 7B 的检测/再认可靠性本身是瓶颈。OVO-EPM 中有多少题针对“后来才显著的实体”未知。预期 0–2 分。 |
| H | Generative Agents 的 importance + reflection | 给 caption 打重要性分、周期性 reflection 生成高层摘要。 | 不推荐 | 高，可能有害。importance 等价于我们已失败的 surprise 保留。按 FTT，reflection 强化 gist 会增加 phantom recollection 和对相似干扰项的接受，可能恶化 ambiguous。只适合作为对照或负面预测来验证理论。 |
| I | TAP + LongMemEval fact-augmented key expansion / doc2query | **写入时按题型族模板生成候选 QA**（“X 现在在哪？”“Y 之后做了什么？”），存成 key 加短答案。 | absent（逼出细节） | 高。读端没有 headroom，key 只帮检索；它唯一的作用是作为另一种“逼出细节”的 prompt，而这类 prompt 已证明不改变比例。预期约 0 分。 |
| J | FTT（verbatim 衰退快、gist 稳定）+ 我们的 90 s 现象 | **分层保留**：最近 2–3 分钟保留压缩 verbatim（帧嵌入或少量低分辨率帧，在独立通道），更远只留 gist。 | absent（近期题） | 中。依赖于 absent 是否集中在近期。若 absent 主要在远期则无用。建议先按“证据距问题的时间 × 桶”做交叉统计（零成本诊断）。 |

### Inferences
- **总体判断**：理论最一致地指向“query-agnostic 的单一 gist 写入有结构性上限”。这恰好支持当前“转向 findings”的叙事：可以把 44/44/6 写成编码特异性/TAP 加 FTT 的计算实例，把“surprise 无效”“结构化无效”“回看编造”分别对应到 PE 理论、CLS 中索引与内容的区分、source monitoring 失败。这比继续寻找写入端增益更稳妥。
- **如果还要尝试一个方法**，理论上最有依据、且与负结果不重复的是“B（诊断性写入）+ D（顺序账本）”。它们只改写入、成本低，可直接用词重叠代理量化 ambiguous 是否下降。
- **“A+C”是唯一可能触及 absent 桶的路线**，但与已观察到的 7B confabulation 正面冲突，应先在小样本上测诱饵校准下的 yes-bias，再决定是否投入。
- **零成本的诊断**（理论驱动，可先做）：
  1. 按任务拆分（EPM 对 ASI）三桶比例，并核查 ASI 选项是否为排列。
  2. 用嵌入或 LLM-judge 复核词重叠代理（同义改写会高估 absent）。
  3. 按“证据时刻到问题的距离”对三桶做交叉统计，以检验 FTT 式的“近期 verbatim 可用、远期只有 gist”。
- **不要高估类比**：FTT、TAP 描述的是人类在约束下的记忆规律，并不提供算法。“选项作为再认线索”在人类中有效，是因为人的再认对 lure 有较好的 recollection rejection，而 7B VLM 的这一能力正是未知数。上表的风险估计已经考虑了这一点。

### Gaps
- 上表中所有改动都没有在 OVO-Bench 上的已发表 training-free 先例可以直接借用增益数字。FOLIO（training-free、实体为中心加 visual-evidence cache，Backward 69.1）是最接近的竞品，值得细读它的写入与证据缓存机制，但本次只读到摘要。
- 没有找到 Generative Agents 或 MemoryBank 式 importance/decay 在视频 QA 上的消融。
- LongMemEval 的 key expansion 增益数字本次未读到（摘要中没有）。
- CAM（2026-09）是否 training-free、verifier 如何避免回看编造，本次无法确认，值得读全文，因为它直接触及“回看加验证”路线。
