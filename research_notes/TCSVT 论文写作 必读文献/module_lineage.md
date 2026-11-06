# ReadPast 三个模块在流式视频文献之外的思想谱系（Related Work 必读与措辞）

> 调研日期：2026-11-03。本笔记只覆盖流式视频**以外**的谱系；流式同行（FOLIO、ShallowStream、FlashBack、StreamScout 等）见 `/data/REN/streamkv/RELATED.md`，答案端弃答见 `/data/REN/streamkv/reports/R1 答案端弃答 设计参考.md`（下称"R1 调研"）。
>
> **venue 核验来源记号**：
> - 【Anth】：ACL Anthology ID，来自 Semantic Scholar API 的 externalIds 或 Anthology 页面。
> - 【CVF】/【ECVA】/【ICLR-proc】：会议开放获取页面。
> - 【arXiv注】：arXiv 页上作者自己写的录用注记，未再查会议页。
> - 【S2】：Semantic Scholar 的 venue 字段。S2 会出错，例如它把 LangRepo 记为 ACL 2024，Anthology 实际是 Findings of ACL 2025。
> - 【未核实】：本轮没有找到正式 venue。
>
> "引用数"是 2026-11-03 Semantic Scholar API 的读数，只用来排优先级。英文定位句是给论文用的草稿，其余均为中文笔记。
>
> **必引 / 可选**：必引指 Related Work 里应当出现；可选指有篇幅再引，或在正文某处顺带引用。

## (a1) 文本/字幕记忆谱系：该读哪些、哪些免训练、"文本记忆优于视觉 token"的标准论证是什么

### Takeaway
非流式长视频文献早已把"先把视频写成带时间戳的文字，再让语言模型读文字作答"确立为主流的免训练范式，主线是 Socratic Models → LLoVi / VideoAgent / LifelongMemory / DrVideo / LangRepo。所以 ReadPast 不能把"文本记忆"本身当贡献。相对这条谱系，能主张的差别只有三点：
1. 字幕由**作答用的同一个冻结 VLM** 在线、因果地自己写，本谱系全部用独立的字幕器。
2. 文本**补充**近帧，而不是替代画面。
3. 粒度粗（16 s 一条），本谱系是 1–5 s 一条。

### Cited Findings

**A1. Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**。Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski 等 13 人；ICLR 2023【S2】；arXiv 2204.00598；引用 776。**必引（谱系源头）**
- 多个预训练模型通过多模态提示零样本组合，"without requiring finetuning"，应用之一是第一人称视频的自由问答。— [arXiv](https://arxiv.org/abs/2204.00598)
- 做法分三步：抽取关键时刻，为关键帧写字幕，再递归总结成 "language-based world-state history"，每条带时间戳。LM 用 "{world-state history} Q: {question} A:" 模板作答。附录 C："All pretrained models are used off-the-shelf with no additional finetuning"。字幕由 CLIP 排序地点、物体、活动后交给 GPT-3 总结，写字幕和作答是不同模型。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00598)
- 关键时刻有两种取法（附录 D.2/D.3）："frames sampled at fixed time intervals" 的均匀采样，得到与问题无关的日志；或按问题做视频检索。作者说回忆类问题用检索式索引效果最好，但没给采样间隔。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00598)
- 论证：语言日志是"对可观察世界事件的即时重建"，把视频理解变成 "reading comprehension"，并提供"an interpretable window"。作者没有明确主张"更紧凑"。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00598)

**A2. A Simple LLM Framework for Long-Range Video Question-Answering（LLoVi）**。Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, Gedas Bertasius；EMNLP 2024 main【Anth 2024.emnlp-main.1209】；arXiv 2312.17235；引用 240。**必引**
- 用短时字幕器（BLIP2 / LaViLa / LLaVA）给 0.5–8 s 的短片段写描述，再由 LLM（GPT-3.5/4）汇总做长程推理。作者把它与需要 "memory queues, state-space layers" 等专门长程建模的方法对立；"先总结噪声字幕再作答"的提示明显提分。— [arXiv](https://arxiv.org/abs/2312.17235)
- EgoSchema 默认把视频切成 1 s 片段，用 LaViLa 写字幕，GPT-3.5 作答（§3.3）。字幕器输入不含问题，问题只在 LLM 阶段出现（§3.1–3.2）。给 3 分钟视频写完所有字幕要 22.36 s（§4.2.4）。论文没有明说字幕跨问题缓存复用。— [arXiv HTML](https://arxiv.org/html/2312.17235v3)

**A3. VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding**。Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, Qing Li；ECCV 2024【ECVA】；arXiv 2403.11481；引用 319。**必引**
- 构建结构化记忆，包括时间事件描述和以物体为中心的跟踪状态，再由 LLM 调用工具查询记忆作答。相对基线在 NExT-QA 上 +6.6%、EgoSchema 上 +26.0%。— [arXiv](https://arxiv.org/abs/2403.11481)；venue 见 [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3241_ECCV_2024_paper.php)
- 时间记忆：每个约 2 s 的片段由 LaViLa 写一句字幕，同时存 ViCLIP 片段特征和字幕文本嵌入（§1、§2.2）。记忆构建（Alg. 1）和推理（Alg. 2）是两个分开的阶段（§2.1）。作答 LLM 是 GPT-4（§2.4），全程零样本。
- 作者的设计原则是 "minimal but sufficient"，并发现 "it is more effective to use the caption–query similarities for video grounding than using video–query similarities"。— [arXiv HTML](https://arxiv.org/html/2403.11481v2)

**A4. VideoAgent: Long-form Video Understanding with Large Language Model as Agent**。Xiaohan Wang, Yuhui Zhang, Orr Zohar, Serena Yeung-Levy；ECCV 2024【S2】；arXiv 2403.10517；引用 475。**必引（作为"按问题自适应取证"的对照）**
- LLM 作为中心代理，迭代地找出并汇总关键信息，VLM 作为工具。EgoSchema 零样本 54.1%，NExT-QA 71.3%，平均只用 8.4 / 8.2 帧。— [arXiv](https://arxiv.org/abs/2403.10517)
- 先用固定提示 "describe the image in detail" 给 5 帧均匀采样帧写字幕（§3.1、§4.4）。之后由 LLM 根据问题写 CLIP 查询，取回新帧再写字幕（§3.3–3.4）。字幕器用 LaViLa 或 CogAgent，作答用 GPT-4（§4.2）。— [arXiv HTML](https://arxiv.org/html/2403.10517v1)

**A5. Vamos: Versatile Action Models for Video Understanding**。Shijie Wang, Qi Zhao, Minh Quan Do, Nakul Agarwal 等 6 人；ECCV 2024【arXiv注】；arXiv 2311.13627；引用 51。**必引（"文本记忆"论证的实证来源）**；需要训练
- 主张重新审视 "text-based representations, such as general-purpose video captions, which are interpretable and can be directly consumed by LLMs"。在 Ego4D、NeXT-QA、IntentQA、Spacewalk-18、EgoSchema 五个基准上，"text-based representations consistently achieve competitive performance ... visual embeddings provide marginal or no performance improvement"。— [arXiv](https://arxiv.org/abs/2311.13627)

**A6. Language Repository for Long Video Understanding（LangRepo）**。Kumara Kahatapitiya, Kanchana Ranasinghe, Jongwoo Park, Michael S. Ryoo；**Findings of ACL 2025**【Anth 2025.findings-acl.294；S2 误记为 ACL 2024】；arXiv 2403.14622；引用 78。**必引（"建一次、跨问题复用"的明确出处）**
- 动机是 LLM "effectiveness in handling long-term information gradually declines with input length"，因此维护简洁、结构化、全文本的仓库，写入和读取操作负责剪除冗余。— [arXiv](https://arxiv.org/abs/2403.14622)；venue 见 [Anthology](https://preview.aclanthology.org/setup/2025.findings-acl.294)
- 免训练（§4："requires no training"）。写入只以字幕为输入，不看问题（Eq. 1–8），问题只在最后作答时出现（Eq. 10）。§5.2.0.3 指出建好的仓库可跨同一视频的多个问题复用。默认用 LaViLa 预提取的短片段字幕（§5.2.0.4），LLM 是 Mistral-7B / Mixtral（§5）。— [arXiv HTML](https://arxiv.org/html/2403.14622v2)

**A7. DrVideo: Document Retrieval Based Long Video Understanding**。Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, Jianfei Cai；CVPR 2025【CVF】；arXiv 2406.12846；引用 98。**必引（与 LLoVi 组引）**
- 核心是把长视频理解转成长文档理解：先生成粗的文本长文档，再由代理循环补充缺失信息。在 EgoSchema、MovieChat-1K 和 Video-MME long 上超过此前基于 LLM 的方法。— [arXiv](https://arxiv.org/abs/2406.12846)；[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_DrVideo_Document_Retrieval_Based_Long_Video_Understanding_CVPR_2025_paper.html)
- 采样率：EgoSchema 0.5 fps，Video-MME 0.2 fps（§4.2）。初始文档用固定提示 "describe the picture in no more than 50 words" 生成，与问题无关（§3.1），之后按问题增补（§3.2–3.3）。字幕器是 VLM（LaViLa / LLaVA-NeXT），规划和作答用 LLM（GPT-4 / DeepSeek V2.5）（§4.2、§4.4）。论文称 "DrVideo is a training-free framework"（§1）。— [arXiv HTML](https://arxiv.org/html/2406.12846v2)

**A8. VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos**。Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon 等 7 人；CVPR 2025【arXiv注】；arXiv 2405.19209；引用 353。**可选（"按问题选帧再写字幕"的对照）**
- 免训练，构建按问题自适应的层次表示。EgoSchema 61.1%，NExT-QA 75.6%。— [arXiv](https://arxiv.org/abs/2405.19209)
- "VideoTree selects only sparse keyframes for captioning"（§1）。每帧都提视觉特征，但只给聚类关键帧写字幕（§3.1）。EgoSchema 按 1 fps 采样，平均 62.4 条字幕（§4）。— [arXiv HTML](https://arxiv.org/html/2405.19209v3)

**A9. Video ReCap: Recursive Captioning of Hour-Long Videos**。Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan 等 6 人；CVPR 2024【arXiv注】；arXiv 2402.13250；引用 120。**可选**；需要训练
- 递归视频-语言模型，处理 1 秒到 2 小时的输入，输出多个层级的字幕，用课程学习训练，并发布了 Ego4D-HCap。— [arXiv](https://arxiv.org/abs/2402.13250)
- 层级粒度见补充材料 S1：片段字幕每 4 s 一条；段落描述由每 4 s 一条的片段字幕生成；视频摘要由每 3 分钟一条的段落描述生成。用于 EgoSchema 时，把字幕喂给纯文本 GPT-3.5 零样本作答（§6.2）。— [arXiv HTML](https://arxiv.org/html/2402.13250v6)

**A10. LifelongMemory: Leveraging LLMs for Answering Queries in Long-form Egocentric Videos**。Ying Wang, Yanlai Yang, Mengye Ren；venue【未核实，arXiv 和搜索都没有正式 venue】；arXiv 2312.05269；引用 55。**可选（第一人称字幕记忆，引 arXiv 版）**
- LaViLa "Each caption is generated using 4 frames uniformly taken from a two-second video clip"（§4.1）。
- 按固定间隔逐片段写字幕（§3.1），再做 Caption Digest：去掉无信息字幕，按嵌入相似度去掉与问题无关的字幕，用 LLM 合并相邻的相似字幕（§3.1、§4.1）。
- 作答用 GPT-3.5 / GPT-4（§4.3）。作者把文字描述称为 "a form of episodic memory"。
- NLQ 精修阶段用了在 NLQ 训练集上训练的分类器，所以只是部分免训练。— [arXiv HTML](https://arxiv.org/html/2312.05269v3)

**A11. MM-VID: Advancing Video Understanding with GPT-4V(ision)**。Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin 等 12 人；只有 arXiv【S2 记为 arXiv；未见正式 venue】；arXiv 2310.19773；引用 102。**可选**
- GPT-4V 为片段写描述，通常每段 10 帧，片段边界由 PySceneDetect 给出；再加上 ASR，由 GPT-4 整合成带时间戳的长剧本（§4、§6.2），供 LLM 完成各种视频理解任务。— [arXiv HTML](https://arxiv.org/html/2310.19773v1)

**A12. Understanding Long Videos with Multimodal Language Models（MVU）**。Kanchana Ranasinghe, Xiang Li, Kumara Kahatapitiya, Michael S. Ryoo；ICLR 2025【arXiv注】；arXiv 2403.16998；引用 37。**可选，但建议引作"警示"**
- "LLM-based approaches can yield surprisingly good accuracy on long-video tasks with limited video information, sometimes even with no video specific information"。— [arXiv](https://arxiv.org/abs/2403.16998)

**A13. Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension**。Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin 等；NeurIPS 2025【arXiv注】；arXiv 2411.13093。**可选（优先级高：配置上最接近 ReadPast）**
- "a training-free and cost-effective pipeline that employs visually-aligned auxiliary texts"，在视觉内容之外给 LVLM 提供额外信息。本轮只读了摘要前段，辅助文本的具体来源需回原文核对。— [arXiv](https://arxiv.org/abs/2411.13093)

**A14. Generative Agents: Interactive Simulacra of Human Behavior**。Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris 等；venue 通常引作 UIST 2023【未核实，本轮没打开 ACM 页面】；arXiv 2304.03442。**可选（跨领域："自己写的自然语言记忆"）**
- 架构让 LLM "store a complete record of the agent's experiences using natural language"，随时间综合成更高层的反思，并动态检索。— [arXiv](https://arxiv.org/abs/2304.03442)

**备选（只在篇幅允许时考虑）**：Long Story Short（Chung & Yu，BMVC 2023【arXiv注】，arXiv 2311.01233），先把叙事视频总结成情节再检索。— [arXiv](https://arxiv.org/abs/2311.01233)

**文献里"文本记忆优于视觉 token"的标准论证（有出处的版本）**：
1. **LLM 可直接读、可解释**：Vamos 的 "interpretable and can be directly consumed by LLMs"（[arXiv](https://arxiv.org/abs/2311.13627)）；Socratic Models 的 "interpretable window"（[ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00598)）。
2. **省掉专门的长程视频建模**：LLoVi 把自己与 memory queue、状态空间层等方法对立（[arXiv](https://arxiv.org/abs/2312.17235)）；DrVideo 把问题化为长文档理解，"to effectively leverage the power of LLMs"（[arXiv](https://arxiv.org/abs/2406.12846)）。
3. **长输入会让 LLM 变差，所以要压成简洁文本**：LangRepo 的动机（[arXiv](https://arxiv.org/abs/2403.14622)）。
4. **实证上不输视觉嵌入**：Vamos 五个基准的结论（[arXiv](https://arxiv.org/abs/2311.13627)）；VideoAgent(Fan) 发现字幕-问题相似度比视频-问题相似度更适合定位（[arXiv HTML](https://arxiv.org/html/2403.11481v2)）。
5. **反面警示**：MVU 显示，几乎不给视频信息时 LLM 也能答对不少（[arXiv](https://arxiv.org/abs/2403.16998)）。文本路线的增益可能部分来自语言先验。

### Inferences
- **整体定位句（英文草稿）**："Converting video into timestamped text that a language model reads at question time is an established training-free recipe for long-video QA [Socratic Models, LLoVi, VideoAgent (Fan et al.), LifelongMemory, DrVideo, LangRepo]. ReadPast adopts this recipe for streaming input, but (i) the captions are written online and causally by the same frozen VLM that answers, rather than by a separate captioner feeding a separate LLM; (ii) the text memory complements, rather than replaces, the visual input, since the most recent frames stay as pixels; and (iii) captions are written at a much coarser 16-s cadence to bound online cost."
- **逐篇定位句（英文草稿）**：
  - A1 Socratic Models："Our timeline is a streaming, single-model instance of the language-based world-state history of Socratic Models."
  - A2 LLoVi："Like LLoVi, we caption at a fixed cadence without seeing the question and let a language model reason over the captions; unlike LLoVi, captioner and reader are the same VLM and the recent frames remain visual."
  - A3 VideoAgent (Fan)："VideoAgent builds a 2-s caption memory plus object tracks offline and queries it with tools; ReadPast keeps only a 16-s caption timeline, built online, and reads it in a single pass without tool calls."
  - A4 VideoAgent (Wang)："Instead of iteratively deciding which frames to caption for each question, ReadPast writes question-agnostic captions once and only decides whether to read them."
  - A5 Vamos："Vamos showed that free-form captions are a competitive representation for video reasoning; we exploit this for the past, while keeping visual tokens for the present."
  - A6 LangRepo："As in LangRepo, the text memory is built without the question and reused across questions; we do not prune or restructure it."
  - A7 DrVideo："DrVideo turns a long video into a document and iteratively augments it per question; ReadPast's document is append-only and is never augmented per question."
  - A12 MVU："Because LLMs answer many long-video questions from priors alone [MVU], we report question-only/blind controls alongside the memory gains."
  - A13 Video-RAG："Video-RAG also augments an LVLM's frames with auxiliary text in a training-free way; our text is the model's own running summary of the past."
- **借用 vs 新的**：
  - 借用的（需引用，不能主张）：字幕即记忆；时间戳日志；字幕与问题无关、按固定间隔写；LLM/VLM 读文字作答；免训练。
  - 相对这条谱系新的：同一个 VLM 既写又读，在线因果；"近帧画面 + 远处文字"的混合；"要不要读记忆"的门（见 b、c）。
  - 但"同一 VLM 在线写文本记忆"在流式同行里已有：FOLIO 每 8 s 写实体中心的文本记忆，DeltaStreamer 的 1F-TextDelta 基本就是我们的 delta16（见 `/data/REN/streamkv/RELATED.md`）。所以"自写"只能作为相对离线谱系的差异，不能作为整篇论文的新意。
- 本谱系在 EgoSchema 上普遍用 GPT-3.5/4 这类强 LLM 作答。ReadPast 用 7–8B VLM 自己读自己写的字幕，可以表述为"该范式在小型开源 VLM、在线设定下依然成立"的证据，但不宜说成首次。
- MVU 的警示与 R1 调研里"blind 预测"对照行是同一问题：审稿人可能质疑时间线的增益来自语言先验。建议主表或附录给出"只读题目 / 只读时间线不看帧"两行。

### Gaps
- Video-RAG 的辅助文本来源（OCR / ASR / 检测等）本轮没有读到正文，摘要只说 "visually-aligned auxiliary texts"。写入 Related Work 前需核对。
- Generative Agents（UIST 2023）、VideoAgent(Wang)（ECCV 2024）、Socratic Models（ICLR 2023）的 venue 只有 S2 或通行引法佐证，没有打开会议页面。LifelongMemory、MM-VID 没有找到正式 venue。
- 没有检索 egocentric 方向更多的字幕记忆工作，例如 Ego4D 挑战报告里的 caption-based 方案。

## (a2) 补充：固定间隔、与问题无关、离线缓存的字幕，是不是既定做法（字幕粒度与写读分工）

### Takeaway
是。"固定间隔、不看问题地写字幕，再由语言模型读字幕作答"在 LLoVi（1 s）、VideoAgent(Fan)（2 s）、LifelongMemory（2 s）、Video ReCap（4 s）、DrVideo（2–5 s）和 Socratic Models 的均匀采样版里都是标准步骤。"建一次、跨问题复用"只有 LangRepo 明确写出（§5.2.0.3），其余论文只是结构上隐含。另一支是**按问题选帧再写字幕**：VideoAgent(Wang)、VideoTree、Socratic 的检索版。所有方法都用**独立的字幕器**加**另一个 LLM** 作答，没有一篇由作答模型自己写字幕。ReadPast 16 s 一条，比这些方法粗 3–16 倍。

### Cited Findings

| 方法 | 每条字幕覆盖 | 每条用几帧 | 字幕器 → 作答者 | 字幕与问题无关？ | 预先算好、跨问题复用？ | 出处 |
|---|---|---|---|---|---|---|
| Socratic Models | 均匀版"固定时间间隔"，间隔未给；检索版按问题取关键时刻 | 关键帧（单帧） | CLIP 排序 + GPT-3 总结 → GPT-3 | 均匀版是；检索版否 | 未说明 | 附录 C、D.2、D.3，[ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00598) |
| LLoVi | 1 s（EgoSchema）；总体 0.5–8 s | 推理时未给 | LaViLa → GPT-3.5 | 是（§3.1） | 未明说；3 分钟视频写全部字幕要 22.36 s（§4.2.4） | §3.1、§3.3，[arXiv HTML](https://arxiv.org/html/2312.17235v3) |
| VideoAgent (Fan) | 约 2 s | 页面印作 "44 frames"，应为 4 帧（渲染问题，待核） | LaViLa → GPT-4 | 是：记忆构建 Alg. 1 与推理 Alg. 2 分开 | 结构上隐含，未明说 | §1、§2.1、§2.2、§2.4，[arXiv HTML](https://arxiv.org/html/2403.11481v2) |
| LifelongMemory | 2 s | 4 帧 | LaViLa → GPT-3.5 / GPT-4 | LaViLa 字幕是；之后 Caption Digest 按问题过滤；LLaVA 变体带问题提示 | 未说明 | §3.1、§4.1、§4.3，[arXiv HTML](https://arxiv.org/html/2312.05269v3) |
| Video ReCap | 片段 4 s；段落层由片段字幕生成；摘要由每 3 分钟的段落描述生成 | 未提取 | 训练过的 ReCap → GPT-3.5 | 隐含是（"like before"），未明说 | 未说明 | 补充 S1、§6.2，[arXiv HTML](https://arxiv.org/html/2402.13250v6) |
| DrVideo | 0.5 fps（2 s，EgoSchema）/ 0.2 fps（5 s，Video-MME） | 单帧 | LaViLa / LLaVA-NeXT → GPT-4 / DeepSeek V2.5 | 初始文档是（固定提示）；之后按问题增补 | 未明说 | §3.1–3.3、§4.2、§4.4，[arXiv HTML](https://arxiv.org/html/2406.12846v2) |
| LangRepo | 预提取的 LaViLa 短片段字幕，时长未给 | 未给 | LaViLa → Mistral-7B / Mixtral | 是（Eq. 1–8） | **是，明确写出可跨问题复用** | §5、§5.2.0.3–5.2.0.4，[arXiv HTML](https://arxiv.org/html/2403.14622v2) |
| MM-VID | 按场景切分（PySceneDetect） | 通常 10 帧 | GPT-4V → GPT-4 写剧本 → LLM 作答 | 隐含是（整段视频的"comprehensive description"） | 隐含，未明说 | §4、§6.2，[arXiv HTML](https://arxiv.org/html/2310.19773v1) |
| VideoAgent (Wang) | 按问题迭代取帧；初始 5 帧均匀采样；1 fps 解码 | CogAgent 单帧；LaViLa 片段 | LaViLa / CogAgent → GPT-4 | 初始 5 帧是；后续取帧否 | 否（逐轮生成） | §3.1、§3.3–3.4、§4.2，[arXiv HTML](https://arxiv.org/html/2403.10517v1) |
| VideoTree | 只给按问题选出的关键帧写字幕；1 fps 采样；EgoSchema 平均 62.4 条（推算约每 3 s 一条） | 单帧 | LaViLa / CogAgent / LLaVA-1.6 → GPT-4 | 否 | 未提缓存 | §1、§3.1、§4，[arXiv HTML](https://arxiv.org/html/2405.19209v3) |

### Inferences
- **"固定间隔、与问题无关、缓存后读"应写成借用的既定做法**。引用组合建议：
  - 固定间隔、与问题无关：LLoVi、VideoAgent(Fan)、LifelongMemory、DrVideo 和 Socratic Models（均匀版）。
  - 建一次、跨问题复用：LangRepo。
  - 写法示例："Following common practice in LLM-based long-video QA, captions are produced at a fixed cadence without access to the question and cached for reuse [LLoVi, VideoAgent, LifelongMemory, DrVideo, LangRepo]."
- 在流式设定下，"与问题无关 + 缓存"是**必须**而不是可选：问题到达时过去的帧已经丢掉，只能事先写好。这句话可以放在方法节，作为与离线谱系的本质差别；按问题选帧的 VideoAgent(Wang)、VideoTree 在流式里无法直接使用。
- **16 s 粒度要有理由**。离线谱系是 1–5 s 一条，ReadPast 粗 3–16 倍。建议正文给出写入开销（每 16 s 一次 VLM 生成）和粒度消融。如果已有 delta16 与其他间隔的对比，可以直接引用。
- **"同一模型自写"是本谱系里没有的配置**：10 个方法全部是"字幕器 → 另一个 LLM"。可以写成 "a single frozen VLM serves as both writer and reader"，但如上所述，流式同行里有先例。

### Gaps
- LLoVi 推理时每条字幕的帧数、Video ReCap 的帧数、LangRepo 的片段时长，三篇论文都没给出明确数字。
- VideoAgent(Fan) HTML 页面把帧数印作 "44"，需要回 PDF 核对。
- 除 LangRepo 外，各论文都没有明说缓存复用。"预先算好"只能写成"字幕不依赖问题，因此可以缓存"，不宜说"这些论文都缓存了"。

## (b) 自适应 / 条件检索（"何时检索"）：哪些只看问题作决定，阈值怎么定（标注、留出集还是无标签）

### Takeaway
"只看问题、在检索前就做决定"是成熟的一支：Mallen 等的实体流行度、SKR、Adaptive-RAG、RetrievalQA 的 TA-ARE，ReadPast 的门属于这一支。另一支在生成过程中触发：FLARE、DRAGIN、SeaKR、Self-RAG 的逐段 Retrieve token。本轮查到的方法，阈值或判据全部来自四种途径之一：带标注的开发集（Mallen）、带标注的训练题（SKR、Adaptive-RAG、Probing-RAG）、人工设定（Self-RAG 0.2 / 0）、少量示例（TA-ARE）。RetrievalQA 明确指出 "calibration-based methods heavily rely on threshold tuning, while vanilla prompting is inadequate"。**没有检索到用内容无关输入、无标签地设检索门阈值的工作。**这是 ReadPast 门控最站得住的差异点，但新意窄。

### Cited Findings

**B1. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories**。Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi；ACL 2023 long【Anth 2023.acl-long.546】；arXiv 2212.10511；引用 1626。**必引**
- LM 在长尾、低流行度知识上表现差；未增强的 LM 在高流行度实体上仍有竞争力。据此提出 "retrieves non-parametric memories only when necessary"。— [arXiv](https://arxiv.org/abs/2212.10511)
- 决策 "only based on input query information"：问题中实体的流行度低于阈值才检索。阈值 "is chosen to maximize the adaptive accuracy"，并且 "independently for each relationship type"：在 75% 的 PopQA 上暴力搜索，剩下 25% 评测，重复 100 次随机划分。推理时间最多省 9%，EntityQuestions 上 API 成本省 15%。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.10511)

**B2. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**。Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi；ICLR 2024【S2/DBLP conf/iclr/AsaiWWSH24】；arXiv 2310.11511；引用 3017。**必引**
- 训练单个 LM 通过反思 token 按需检索、生成和自评，批评模型的数据由 "prompting GPT-4 to generate reflection tokens" 得到。检索判据是 Retrieve=Yes 的归一化概率超过阈值，"we set the retrieval threshold to 0.2 for most tasks and to 0 for ALCE"。— [arXiv](https://arxiv.org/abs/2310.11511)；[arXiv HTML](https://arxiv.org/html/2310.11511)
- 在 RetrievalQA 上，Self-RAG 的检索率从 t=0.25 时的 100% 变到 t=0.75 时的 0%，作者称 "The effectiveness of Self-RAG largely depends on threshold tuning"。— [arXiv HTML](https://arxiv.org/html/2402.16457v2)

**B3. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**。Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park；NAACL 2024 long【Anth 2024.naacl-long.389】；arXiv 2403.14403；引用 794。**必引**
- 根据问题复杂度，在不检索、单步检索和多步检索之间选择。选择器是 "a smaller LM trained to predict the complexity level of incoming queries with automatically collected labels"，标签来自模型的实际预测结果和数据集的归纳偏置。— [arXiv](https://arxiv.org/abs/2403.14403)

**B4. Self-Knowledge Guided Retrieval Augmentation for Large Language Models（SKR）**。Yile Wang, Peng Li, Maosong Sun, Yang Liu；Findings of EMNLP 2023【Anth 2023.findings-emnlp.691】；arXiv 2310.05002；引用 114。**必引**
- 检索有时会损害原本正确的回答，所以让 LLM 参照见过的问题自适应地决定是否检索。— [arXiv](https://arxiv.org/abs/2310.05002)；venue 见 [Anthology](https://preview.aclanthology.org/new-sigs/2023.findings-emnlp.691/)
- 训练题各答两遍（有 / 无检索），对照金标答案分成 known / unknown 两类。之后有四种引出方式：
  - 直接提示，问 "Do you need additional information to answer this question?"
  - 上下文示例
  - BERT 分类器
  - SimCSE kNN：k 取 3–10，判正条件为 ℓ/(k−ℓ) ≥ m/n
  
  决定只依赖新问题本身。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05002)

**B5. RetrievalQA: Assessing Adaptive Retrieval-Augmented Generation for Short-form Open-Domain Question Answering（含 TA-ARE）**。Zihan Zhang, Meng Fang, Ling Chen；Findings of ACL 2024【arXiv注；Anth 2024.findings-acl.415】；arXiv 2402.16457；引用 45。**必引（最适合拿来立论）**
- 结论："calibration-based methods heavily rely on threshold tuning, while vanilla prompting is inadequate for guiding LLMs to make reliable retrieval decisions"。— [arXiv](https://arxiv.org/abs/2402.16457)
- 直接问 LLM 要不要检索时，检索准确率为 TinyLlama 39.1、Phi-2 94.1、Llama-2 80.3、GPT-3.5 49.3、GPT-4 67.6，大模型反而不如小模型。
- TA-ARE 在提示里加入当天日期和 4 个示例：2 个 [Yes] 示例是模型答错的长尾问题，按 SimCSE 选最相似的；2 个 [No] 示例人工编写。不设阈值、不训练。GPT-3.5 的检索准确率从 49.3 升到 86.3。— [arXiv HTML](https://arxiv.org/html/2402.16457v2)

**B6. Active Retrieval Augmented Generation（FLARE）**。Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun 等 9 人；EMNLP 2023【arXiv注；main 还是 Findings 未查 Anthology ID】；arXiv 2305.06983；引用 953。**必引（与 DRAGIN 组引，代表"生成中触发"一支）**
- 先预测下一句；"if it contains low-confidence tokens"，就用它作查询检索并重写该句。— [arXiv](https://arxiv.org/abs/2305.06983)

**B7. DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models**。Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, Yiqun Liu；ACL 2024 long【Anthology 页面，pp. 12991–13013】；arXiv 2403.10081；引用 109。**可选**
- 在生成过程中根据 LLM 的实时信息需求决定何时检索、检索什么，并批评已有方法 "rely on static rules"。— [arXiv](https://arxiv.org/abs/2403.10081)；[Anthology](https://aclanthology.org/2024.acl-long.702/)

**B8. SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation**。Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao 等 8 人；ACL 2025 long【Anth 2025.acl-long.1312】；arXiv 2406.19215；引用 68。**可选**
- 从内部状态提取自我感知的不确定性，"activates retrieval when the LLMs present high self-aware uncertainty"。— [arXiv](https://arxiv.org/abs/2406.19215)；[Anthology](https://aclanthology.org/2025.acl-long.1312)

**B9. Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home**。Viktor Moskvoretskii, Maria Lysyuk, Mikhail Salnikov, Nikolay Ivanov 等 10 人；ACL 2025 long【Anth 2025.acl-long.319】；arXiv 2501.12835；引用 51。**可选（支撑"简单信号就够"）**
- 在 6 个数据集上比较了 35 种自适应检索方法（8 种近期方法和 27 种不确定性估计），结论是 "uncertainty estimation techniques often outperform complex pipelines in terms of efficiency and self-knowledge, while maintaining comparable QA performance"。— [arXiv](https://arxiv.org/abs/2501.12835)；[Anthology](https://preview.aclanthology.org/setup/2025.acl-long.319)

**B10. Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering（ReflectiVA）**。Federico Cocchi, Nicholas Moratelli, Marcella Cornia, Lorenzo Baraldi 等 5 人；CVPR 2025【arXiv注 + S2】；arXiv 2411.16863；引用 30。**必引（多模态检索门控的代表）**
- 用反思 token "dynamically determine the need for external knowledge"，按两阶段、两模型的配方训练，在不需要外部知识的任务上保持原有性能。— [arXiv](https://arxiv.org/abs/2411.16863)

**多模态检索门控的备选**：
- mR²AG（Tao Zhang 等，IEEE TMM【arXiv注，年份未核】，arXiv 2411.15041）：把 "Performing retrieval even when external knowledge is not needed" 列为现有 mRAG 的缺陷，用 Retrieval-Reflection 判断是否需要检索。TCSVT 审稿人熟悉 IEEE 期刊，建议引用。— [arXiv](https://arxiv.org/abs/2411.15041)
- OmniSearch（Yangning Li 等，ICLR 2025【ICLR-proc】，arXiv 2411.02937）：自适应规划代理。— [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html)
- Windsock（Shu Zhao 等，NeurIPS 2025 UniReps Workshop【arXiv注】，arXiv 2510.22694）："a query-dependent module making decisions on retrieval necessity and modality selection"，配合 DANCE 指令微调。— [arXiv](https://arxiv.org/abs/2510.22694)

**其他备选**：
- TARG（Yufeng Wang, Lu Wei, Haibin Ling，arXiv 2511.09803【未核实 venue】）：免训练，用无上下文草稿前缀的 logits 计算熵、margin 或方差，"triggers retrieval only when the score exceeds a threshold"。这是本轮找到的最接近"免训练单次前向门"的工作，但阈值怎么定没有读到。— [arXiv](https://arxiv.org/abs/2511.09803)
- Probing-RAG（Baek 等，Findings of NAACL 2025【arXiv注】）用预训练探针读隐状态（[arXiv](https://arxiv.org/abs/2410.13339)）。UAR（Cheng 等，Findings of EMNLP 2024【arXiv注】）把四种判据统一为分类（[arXiv](https://arxiv.org/abs/2406.12534)）。
- Kadavath 等 "Language Models (Mostly) Know What They Know"（只有 arXiv 2207.05221）：模型能 "predict which questions they will be able to answer correctly"。— [arXiv](https://arxiv.org/abs/2207.05221)
- Feng 等 "Don't Hallucinate, Abstain"（ACL 2024【arXiv注】）批评基于校准的方法 "over-reliance on held-out sets"。— [arXiv](https://arxiv.org/abs/2402.00367)

**汇总：决策输入与阈值来源**

| 方法 | 只看问题？ | 训练？ | 阈值 / 判据从哪来 | 出处 |
|---|---|---|---|---|
| Mallen 等 | 是（实体流行度） | 否 | 带金标的开发集，按关系类型网格搜索 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2212.10511) |
| SKR | 是 | 分类器和 kNN 变体需要 | 带金标的训练题（有 / 无检索对比） | [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05002) |
| Adaptive-RAG | 是 | 是（小 LM 分类器） | 自动收集的银标签 | [arXiv](https://arxiv.org/abs/2403.14403) |
| TA-ARE | 是（加日期和示例） | 否 | 无阈值；[Yes] 示例取自模型答错的题，需要知道对错 | [arXiv HTML](https://arxiv.org/html/2402.16457v2) |
| Self-RAG | 输入加已生成的片段 | 是 | 人工设 0.2 / 0；反思 token 由 GPT-4 标注 | [arXiv HTML](https://arxiv.org/html/2310.11511) |
| FLARE / DRAGIN / SeaKR | 否（生成中触发） | 否 / 否 / 否 | 低置信或不确定性超过阈值；本轮未核对取值方式 | 见 B6–B8 |
| TARG | 否（需先生成无上下文草稿） | 否 | 分数超过阈值；取值方式未读到 | [arXiv](https://arxiv.org/abs/2511.09803) |
| ReflectiVA / Windsock | 问题（+图） | 是 | 训练出的反思 token / 模块 | 见 B10 |
| ShallowStream（流式，域内参照） | 是 | 否 | 在 200 道带合成路由标签的生成问题上，按 Wilson 下界标定一次 | `/data/REN/streamkv/RELATED.md` |
| **ReadPast** | **是（单次前向、只用文本）** | **否** | **内容无关输入（N/A、""、[MASK]），无标签** | — |

### Inferences
- **整体定位句（英文草稿）**："Our present/past gate belongs to the family of pre-retrieval, question-only decisions [Mallen et al.; SKR; Adaptive-RAG; TA-ARE], as opposed to triggers fired during generation [FLARE; DRAGIN; SeaKR]. Unlike these methods, it is neither trained nor tuned on labeled or held-out data: its only free parameter, the decision threshold, is set by contextual calibration on content-free inputs, which directly addresses the two failure modes identified by RetrievalQA, namely threshold tuning and uncalibrated prompting."
- **逐篇定位句（英文草稿）**：
  - B1 Mallen："Like Adaptive Retrieval [Mallen et al.], we decide from the question alone whether external memory is needed, but we use the model's own routing likelihood instead of entity popularity, and we need no labeled development set to set the threshold."
  - B2 Self-RAG："Self-RAG learns a retrieval token and thresholds its probability at a hand-set value; we elicit a similar binary decision from a frozen VLM without training and set the threshold without labels."
  - B3 Adaptive-RAG："Adaptive-RAG trains a query-complexity classifier on silver labels; our router is the frozen VLM itself, prompted once."
  - B4 SKR："Our routing prompt is the zero-shot analogue of SKR's direct-prompting variant; we show that its raw output is biased and calibrate it without SKR's labeled training questions."
  - B5 RetrievalQA："RetrievalQA finds that calibration-based gates hinge on threshold tuning and that vanilla prompting is unreliable; content-free calibration removes the tuning while correcting the prompting bias."
  - B10 ReflectiVA："In multimodal RAG, retrieval necessity is typically learned with reflective tokens or a trained module [ReflectiVA; mR²AG; Windsock]; ReadPast obtains it training-free."
- **术语提醒**：ReadPast 的门不做检索，只决定**要不要把整条时间线放进上下文**。写作时用 "adaptive memory access / conditional context inclusion"，再说明"与 adaptive retrieval 的'何时检索'问题同构"，避免审稿人追问检索器在哪里。
- **对 SKR 直接提示变体的呼应**：这个变体就是"未校准的 ReadPast 门"。RetrievalQA 记录的原始提示偏差（GPT-3.5 检索准确率 49.3）可以作为"为什么要校准"的外部证据。如果手上有"未校准门 vs 校准门"的路由比例或 OVO 结果，正好是这个论点的消融。
- **新意的边界**：相对 ShallowStream，差别只在阈值无标签和路由目标是文本时间线（见 RELATED.md 11-01 补充）。相对 NLP 这条谱系，差别是"免训练 + 无标签阈值 + 单次前向只看文本"的组合，不是门控思想本身。论文里应写 "we adopt" 或 "we instantiate"，不要写 "we propose a novel gating mechanism"。

### Gaps
- FLARE、DRAGIN、SeaKR 的阈值具体怎么取（开发集调还是固定），本轮没有读正文。表中只写了"超过阈值"。
- TARG 的阈值设定方式、mR²AG 是否需要微调，都没有读到，摘要截断。
- 只做了一次定向检索（"content-free calibration + adaptive retrieval threshold"），没找到相关工作，但不能据此断言不存在。投稿前建议用 "contextual calibration retrieval gate"、"label-free threshold adaptive RAG" 再在 Semantic Scholar 上检索一次。

## (c) 无标签的 LM 决策阈值校准：上下文校准到底做了什么，ReadPast 的用法差在哪

### Takeaway
Zhao 等的上下文校准做的是对**答案标签**概率做向量缩放：q̂ = softmax(W·p̂ + b)，W = diag(p̂_cf)⁻¹，b = 0。p̂_cf 是 "N/A"、"[MASK]"、空串三个内容无关输入的**概率平均**，整个过程不需要任何标注。对二分类，这在数学上**等价于把对数几率阈值移到内容无关输入的对数几率**，所以 ReadPast 的门就是上下文校准在 2 类问题上的直接实例；从同一个式子看，它也是 PMI_DC 的二类特例。真正的差别不在公式，而在**校准对象**：ReadPast 校准的是一个控制用的元决策（读不读记忆），它的"标签"永远观测不到，错判两类的代价也不对称；不是在校准任务答案的类别先验。论文应写 "we adopt contextual calibration to set the gate threshold"，不应包装成新的校准方法。

### Cited Findings

**C1. Calibrate Before Use: Improving Few-Shot Performance of Language Models**。Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh；ICML 2021【arXiv注 + S2】；arXiv 2102.09690；引用 2195。**必引（直接来源）**
- 少样本 ICL 不稳定，原因是模型偏向某些答案，例如 "those that are placed near the end of the prompt or are common in the pre-training data"。做法是用内容无关输入（如 "N/A"）估计偏置，再 "fit calibration parameters that cause the prediction for this input to be uniform across answers"。平均准确率最多提升 30.0 个点。— [arXiv](https://arxiv.org/abs/2102.09690)
- 具体做法：
  - 公式是 q̂ = softmax(W p̂ + b)，"we restrict the matrix W to be diagonal"，"We then set W = diag(p̂_cf)^-1 and b to the all-zero vector"。
  - 内容无关输入是 "N/A"、"[MASK]" 和空串，"we average the probabilities from three content-free inputs"。这个组合在 AGNews 上选定后用于所有数据集。
  - p̂ 是对标签名概率重归一化后的分布。论文称这是 "a novel data-free procedure"，不需要额外训练数据。
  - 脚注提到另一种设定（W = I，b = −p̂_cf）在生成任务上更好。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2102.09690)
- 选哪种内容无关输入影响很大（Table 3，1-shot SST-2）：未校准 66.5；"N/A" 74.2；"[MASK]" 74.5；空串 72.9；三者集成 79.0；"the" 69.1。作者的说法是 "The choice of content-free input matters, however, many good choices exist"。
- 二分类任务包括 SST-2 和 RTE。GPT-3 175B 在 RTE 上结果有好有坏：4-shot 从 58.7 到 60.4，8-shot 从 66.2 到 65.5。— [ar5iv](https://ar5iv.labs.arxiv.org/html/2102.09690)

**C2. Surface Form Competition: Why the Highest Probability Answer Isn't Always Right（PMI_DC）**。Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, Luke Zettlemoyer；EMNLP 2021 main【Anth 2021.emnlp-main.564】；arXiv 2104.08315；引用 363。**必引**
- 每个选项按 "a term that is proportional to its a priori likelihood within the context of the specific zero-shot task" 重新加权，即用选项在领域前提下的先验做除法。在 GPT-2 / GPT-3 的多项选择任务上，"consistent gains ... over both calibrated (Zhao et al., 2021) and uncalibrated scoring functions"。— [arXiv](https://arxiv.org/abs/2104.08315)
- R1 调研已记录 PMI_DC 在 ARC-Easy 上输给原始 LM 打分，原因是正确答案本身就是先验上的常见答案。— `/data/REN/streamkv/reports/R1 答案端弃答 设计参考.md`

**C3. Mitigating Label Biases for In-context Learning（领域上下文校准）**。Yu Fei, Yifan Hou, Zeming Chen, Antoine Bosselut；ACL 2023 long【Anth 2023.acl-long.783】；arXiv 2305.19148；引用 127。**必引**
- 把 ICL 标签偏置分为 vanilla-label、context-label、domain-label 三类，"prior label bias calibration methods fall short of addressing all three"。改用 "random in-domain words from the task corpus" 估计偏置，在领域标签偏置大的任务上 Macro-F1 最多提升 37%。— [arXiv](https://arxiv.org/abs/2305.19148)

**C4. Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering**。Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu 等 7 人；ICLR 2024【arXiv注 + DBLP】；arXiv 2309.17249；引用 120。**必引**
- 给出现有校准方法的统一视角，并揭示失败案例。BC 从批量输入中估计并去除上下文偏置，"zero-shot, inference-only"，在 PaLM 2 和 CLIP 上验证。— [arXiv](https://arxiv.org/abs/2309.17249)
- R1 调研记录了它报告的失败案例：在 QNLI 上，"Question: N/A, Sentence: N/A" 被理解为蕴含，内容无关 token 反而引入偏置。— `/data/REN/streamkv/reports/R1 答案端弃答 设计参考.md`

**C5. Debiasing Multimodal Large Language Models via Penalization of Language Priors（VDD）**。YiFan Zhang, Yang Shi, Weichen Yu, Qingsong Wen 等 9 人；ACM MM 2025【S2/DBLP conf/mm/…25】；arXiv 2403.05262；引用 42。**必引（多模态版的内容无关校准）**
- MLLM "often provide confident answers even in the absence of relevant images"。对分类和多选题提出 "Post-Hoc Debias"："an affine calibration step ... ensures uniform answer scores when the image is absent"。对开放生成提出 Visual Debias Decoding，即与无意义图像做对比。— [arXiv](https://arxiv.org/abs/2403.05262)
- 内容无关输入有 "None"（无图）和 "Noise"（噪声图）两种。— [arXiv HTML](https://arxiv.org/html/2403.05262v3)（经搜索摘要转述）

**C6. Prototypical Calibration for Few-shot Learning of Language Models**。Zhixiong Han, Yaru Hao, Li Dong, Yutao Sun, Furu Wei；ICLR 2023【S2/DBLP conf/iclr/HanH0SW23】；arXiv 2205.10183；引用 73。**可选**
- 用高斯混合估计各类原型簇，再用加权二分匹配把簇对应到标签。需要一批无标注输出，对类别不平衡更稳健。— [arXiv](https://arxiv.org/abs/2205.10183)

**C7. Generative Calibration for In-context Learning**。Zhongtao Jiang, Yuanzhe Zhang, Cao Liu, Jun Zhao 等；Findings of EMNLP 2023【arXiv注】；arXiv 2310.10266；引用 28。**可选**
- 把 ICL 的不稳定归因于标签边际 p(y) 偏移，用模型生成做蒙特卡洛估计 p(y) 后校正，macro-F1 最多提升 27%。— [arXiv](https://arxiv.org/abs/2310.10266)

**C8. Large Language Models Are Not Robust Multiple Choice Selectors（PriDe）**。Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang；ICLR 2024 Spotlight【arXiv注】；arXiv 2309.03882；引用 703。**可选（与 d 共用）**
- 选择偏差主要来自对选项 ID token 的先验。PriDe 是无标签的推理期去偏方法：在少量测试题上置换选项内容来估计先验，再用于其余题目。— [arXiv](https://arxiv.org/abs/2309.03882)

**C9. Noisy Channel Language Model Prompting for Few-Shot Text Classification**。Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer；ACL 2022 long【arXiv注】；arXiv 2108.04106。**可选**
- 用 P(x|y) 代替 P(y|x)，作者把优势归因于更低的方差。这是另一条对抗标签偏置的路线，不属于校准。— [arXiv](https://arxiv.org/abs/2108.04106)

**C10（横跨 b、c）. Language Models (Mostly) Know What They Know**。Saurav Kadavath 等；只有 arXiv 2207.05221【未见正式 venue】。**可选**
- "larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format"。— [arXiv](https://arxiv.org/abs/2207.05221)

### Inferences
- **等价关系（建议在方法节写成一行公式）**：记 s(q) = log P("past" | T(q)) − log P("now" | T(q))，T 是固定的路由模板。
  - 对两个标签取 W = diag(p_cf)⁻¹、b = 0，判 "past" 当且仅当 P(past|q)/p_cf(past) > P(now|q)/p_cf(now)，即 s(q) > τ，τ = log p_cf(past) − log p_cf(now)。
  - 两类重归一化不改变对数几率，所以 ReadPast 的门**就是** Zhao 式上下文校准的二类情形。
  - 同一式子也可以读成 PMI_DC：领域前提是"问题被置空的路由模板"。
  - 写法示例："For two classes, contextual calibration [Zhao et al.] reduces to shifting the log-odds threshold to that of content-free inputs; equivalently, the decision is the sign of a domain-conditional PMI difference [Holtzman et al.]."
- **实现细节需与 Zhao 对齐**：Zhao 是先对三个内容无关输入的（重归一化）**概率**取平均，再求 τ。如果 ReadPast 是对三者的对数几率取平均，两者会略有不同，论文里要如实写，或改成与 Zhao 一致。
- **"我们的用法差在哪"，按重要性排序**：
  1. **校准对象**：Zhao、Fei、BC 校准的是任务答案的类别先验，目的是提高分类准确率。ReadPast 校准的是控制决策，即要不要把文本时间线放进上下文。路由"标签"在任何基准上都不存在，好坏只能通过下游问答间接看到。
  2. **偏置来源**：Zhao 的主要偏置来自示例（多数标签、近因效应）和常见 token。ReadPast 是零样本、无示例，偏置主要来自两个词（"past" / "now"）的先验和对话模板。RELATED.md 的内部观察支持这一点：两个骨干的 p_cf 相差 4 个数量级，校准后路由比例却几乎相同（65/7 vs 63/6）。
  3. **代价不对称**：上下文校准默认"内容无关输入应当对两类无偏好"，这是对称假设。路由的两类错误代价不同：误判 past 只多读一段文本，误判 now 会丢掉记忆。τ 是一个中性点，不是代价最优点；如果要加偏移量，需要另行论证。
  4. **只用文本做 VLM 的校准**：门只做纯文本前向，内容无关的是**问题**。VDD 的内容无关对象是**图像**，它校准的是答案分布。
- **为什么不用 Batch Calibration 或原型校准**：BC 用测试批的平均输出估计偏置，等于假设批内各类大致均衡；原型校准需要一批无标注输出。路由场景里"过去题"的比例随基准和任务变化很大（BT 与 RT 截然不同），用批均值会把路由比例往 50/50 拉，还要看测试分布。内容无关校准每个骨干和模板只需离线跑 3 次前向，不看任何测试题，这是选它的理由。可以作为一句设计说明，最好配一个小消融。
- **建议的廉价消融**：门只是对问题文本的单次前向，每种变体几分钟就能跑完。
  - Zhao 的三输入集成 vs 单个 "N/A"、"[MASK]"、空串（Zhao Table 3 显示差异可达数个点）。
  - Fei 式随机领域词：从问题语料随机抽词组成伪问题。
  - BC 式测试批均值。
  - 未校准。
  - 报告路由比例和下游 OVO / SB 分数。
- **定位句（英文草稿）**：
  - C1 Zhao："We set the gate threshold with contextual calibration [Zhao et al.], averaging the model's output on 'N/A', '' and '[MASK]', so that no labeled routing data is required."
  - C3/C4 Fei, BC："Because content-free inputs can themselves carry bias [Fei et al.; Zhou et al.], we report the sensitivity of the routing rate to the choice of content-free input."
  - C5 VDD："VDD calibrates an MLLM's answer distribution against an image-free input; we instead calibrate a text-only routing decision against question-free inputs."

### Gaps
- 没有找到把内容无关校准用于**二元路由 / 门控决策**的已发表工作，搜索见 (b) 的 Gaps。也没有专门检索 LLM-as-judge 的 yes/no 偏置校准，这一支可能有可引的先例。
- VDD 的内容无关构造（None / Noise）来自搜索摘要转述，没有逐字读正文。
- BC 的"批内类别均衡"假设是从其"用批均值估计上下文偏置"推出来的，本轮没有找到原文的逐字表述。

## (d) 弃答 / 不可答问题：R1 调研之外，为模块 (3) 还需要补哪些引用

### Takeaway
R1 调研已经覆盖答案端弃答的大部分文献：Reliable VQA、MM-UPD 的 Dual 指标与 AO/AI 设定、Tam 等的 NOTA、Kadavath 的 NOTA 偏差、Ling 等的 "Unknown" 注入、UVQA、AbstentionBench、FOLIO 附录 K.3 的 "(when available)"。模块 (3) 只需补四篇奠基或正式 venue 的引用：SQuAD 2.0、Kamath 等的选择性 QA、Whitehead 等的 Reliable VQA、Miyai 等的 UPD（ACL 2025），另有 2–3 篇可选。最关键的不是补文献，而是一个写作风险：**如果 OVO 里只有 HLD 题的选项含 "unable to answer"，那么"只在选项含弃答项时才加指令"在效果上等于"只对 HLD 加指令"。**这必须用数据核实并在文中说明。

### Cited Findings

**D1. Know What You Don't Know: Unanswerable Questions for SQuAD（SQuAD 2.0）**。Pranav Rajpurkar, Robin Jia, Percy Liang；ACL 2018 short【Anth P18-2124】；arXiv 1806.03822；引用 3586。**必引（不可答问题的奠基工作）**
- 加入 5 万多道众包对抗写成的不可答题，系统必须 "determine when no answer is supported by the paragraph and abstain from answering"。SQuAD 1.1 上 86% F1 的强系统在 2.0 上只有 66% F1。— [arXiv](https://arxiv.org/abs/1806.03822)

**D2. Selective Question Answering under Domain Shift**。Amita Kamath, Robin Jia, Percy Liang；ACL 2020【Anth 2020.acl-main.503】；arXiv 2006.09462；引用 350。**必引（选择性 QA 的奠基工作）**
- "Abstention policies based solely on the model's softmax probabilities fare poorly, since models are overconfident on out-of-domain inputs"。训练一个校准器预测出错并据此弃答：在 80% 准确率下作答 56% 的题，直接用概率只能作答 48%。— [arXiv](https://arxiv.org/abs/2006.09462)

**D3. Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly**。Spencer Whitehead, Suzanne Petryk, Vedaad Shakib, Joseph Gonzalez 等 7 人；ECCV 2022【arXiv注 + S2】；arXiv 2204.13631；引用 131。**必引（R1 调研已用其指标，这里补正式条目）**
- 直接用 softmax 分数弃答时，要把风险压到 1%，模型只能作答不到 7.5% 的题。多模态选择函数把 1% 风险下的覆盖率从 6.8% 提到 15.6%。— [arXiv](https://arxiv.org/abs/2204.13631)

**D4. Unsolvable Problem Detection: Robust Understanding Evaluation for Large Multimodal Models（UPD / MM-UPD）**。Atsuyuki Miyai, Jingkang Yang, Jingyang Zhang, Yifei Ming 等 10 人；**ACL 2025 main**【arXiv注 + DBLP conf/acl/MiyaiYZMYILLLA25】；arXiv 2403.20331。**必引（R1 调研只引了 arXiv，需要更新为 ACL 2025）**
- 三类不可解问题：AAD（缺正确选项）、IASD（选项集不兼容）、IVQD（图文不匹配）。— [arXiv](https://arxiv.org/abs/2403.20331)
- 两种附加设定（§4.3）：
  - AI 设定的指令原文是 "If all the options are incorrect, answer F. None of the above."；IVQD 用 "If the given image is irrelevant to the question, answer F. The image and question are irrelevant."
  - AO 设定加的是选项。
  - "these additional options and instructions are also added to the questions in standard scenarios to make a fair comparison"。
- 效果因模型而异：
  - LLaVA-NeXT-34B 在 AAD 上的 Dual 准确率依次为 50.5（Base）、29.9（AO）、55.1（AI）；IASD 上为 48.9、22.6、61.8。
  - LLaVA-OV-7B 和 InternVL2-8B 从 Option 设定换到 Instruction 设定后，UPD 准确率反而下降。— [arXiv HTML](https://arxiv.org/html/2403.20331v4)

**D5. Do Large Language Models Know What They Don't Know?（SelfAware）**。Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu 等 6 人；Findings of ACL 2023【arXiv注】；arXiv 2305.18153；引用 321。**可选**
- SelfAware 数据集包含不可答题和对应的可答题。ICL 和指令微调能增强自我认知，但与人类仍有明显差距。— [arXiv](https://arxiv.org/abs/2305.18153)

**D6. R-Tuning: Instructing Large Language Models to Say 'I Don't Know'**。Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung 等 9 人；NAACL 2024 long【Anth 2024.naacl-long.394】；arXiv 2311.09677；引用 147。**可选（训练式拒答的对照）**
- 构造拒答感知数据做指令微调。拒答能力 "could be generalized to other tasks"，作者称之为一种元技能。— [arXiv](https://arxiv.org/abs/2311.09677)

**D7. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions**。Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, Samuel J. Bell；NeurIPS 2025【researchr 列表；是否 D&B track 未核实】；arXiv 2506.09038。**可选**
- 覆盖 20 个数据集、6 类弃答场景；推理微调会损害弃答能力。— [arXiv](https://arxiv.org/abs/2506.09038)；venue 见 [researchr](https://researchr.org/publication/KirichenkoICB25)

**R1 调研已引用、这里只补 venue 的**：
- Srinivasan 等 "Selective 'Selective Prediction'"：Findings of ACL 2024【arXiv注】，记录 VLM 选择性预测的过度弃答。— [arXiv](https://arxiv.org/abs/2402.15610)
- Feng 等 "Don't Hallucinate, Abstain"：ACL 2024【arXiv注】。— [arXiv](https://arxiv.org/abs/2402.00367)
- Wen 等 "Know Your Limits" 综述：TACL【arXiv 注记为 "TACL 2024"，卷期未核】。— [arXiv](https://arxiv.org/abs/2407.18418)
- Tam 等 "None of the Above, Less of the Right"：只有 arXiv 2503.01550【未核实】，28 个 LLM 在 NA 为正确答案时一致下降 30–50%。— [arXiv](https://arxiv.org/abs/2503.01550)
- Kadavath 等：只有 arXiv。PriDe 见 C8：弃答项通常排在最后一个选项，选项 ID 先验会影响它被选中的概率。

### Inferences
- **模块 (3) 的借用和新意**：
  - 指令文本取自 D-HSM（arXiv 2608.30294），必须注明出处。
  - "只在选项里已有弃答项时才加指令"在先例上最接近 FOLIO 附录 K.3 的 "prefer 'Unable to answer' (when available)"，属于流式文献，见 R1 调研。
  - MM-UPD 的评测协议则是把指令和选项**统一加到所有题上**。
  - 所以模块 (3) 的新意只是一条部署规则，不是方法贡献。
- **定位句（英文草稿）**："Abstention is classically framed as selective prediction that trades coverage for risk [Kamath et al.; Whitehead et al.]. ReadPast does not learn an abstention policy; it only exposes an 'unable to answer' instruction [D-HSM] when the option set already contains an abstention choice. Adding such options or instructions to every question can lower accuracy on answerable ones [MM-UPD; Ling et al.], and we observe the same effect."
- **关键风险，必须用数据核实**：
  - 如果 OVO 里只有 HLD 题带 "unable to answer" 选项，"选项里有弃答项"这个条件就等于任务标识，模块 (3) 实际上就是"只对 HLD 加指令"。
  - 审稿人会把它读成利用基准结构。RELATED.md 里 SAVEMem "门控阈值按 RT/BT 类别分别设，用了类别标签" 正是同类问题。
  - 建议报告各任务中含弃答选项的题目比例。如果确实只有 HLD 有，就在正文直说："the rule only fires on HLD in OVO-Bench"，并配合 R1 调研的建议：headline 用不含 HLD 的分数，HLD 另报。
  - 能找到含弃答选项的非 OVO 数据最好，例如 MM-UPD 式注入，用来说明规则在非 HLD 题上不伤分。
- **与 R1 调研的一致性**：R1 调研把"只靠弃答 prompt（FOLIO K.3 / D-HSM 式措辞）"列为 R1 必须打败的伪基线，而模块 (3) 正是这条伪基线。如果论文同时保留 R1 和模块 (3)，要说明两者的关系：(3) 是 prompt 规则，R1 是 logit 判据。
- **"对所有题都加指令会掉分"的外部旁证**：
  - MM-UPD 的 AO 设定让 LLaVA-NeXT-34B 的 Dual 准确率从 50.5 跌到 29.9。
  - Ling 等在可答的四选一题上注入 "Unknown" 后，弃答率为 0.2–7.4%（R1 调研）。
  - Srinivasan 等记录了选择性预测的过度弃答。
  
  我们自己的"全加会掉分"可以写成与这些发现一致。

### Gaps
- MM-UPD 正文只给 Dual 准确率，没有逐设定的 Standard-only 准确率，所以无法直接引用"加指令后可答题准确率下降 X 点"。如果需要，要去它的附录表格里找。
- 除 FOLIO K.3 外，没有找到专门研究"按选项集条件化弃答指令"的文献。
- OVO 中含弃答选项的题目分布需要从数据本身统计，不属于文献问题，但决定模块 (3) 怎么写。
