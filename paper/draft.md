# ReadPast: See the Present, Read the Past for Training-Free Streaming Video Question Answering

> 草稿 v0（2026-11-04）。由 `paper/main.tex` 和 `paper/sections/*.tex` 转写成 md，方便阅读；正文用英文，写作计划用中文。目标期刊是 IEEE TCSVT，篇幅按 11 页规划。
> **【待写】**表示还没动笔；**【注】**是写框架之后，根据阅读清单（`reports/TCSVT 论文写作 必读文献.md`）需要改的地方。
> 数字口径：OVO 标准总分 = (BT + RT) / 2，不含 Forward；只报点数增益。
> **【REN 决定 11-05】**投稿目标已改为 ACL 2027（ARR 1 月轮），本稿结构仍按 TCSVT，待换模板。弃答指令**不再是方法组件**，改为所有方法共用的作答提示（做法同 D-HSM：消融行全部带着它，只拆记忆），在 IV-A 用一句话交代、附录给原文；不进贡献、不进消融。主表以"同一作答提示的 SimpleStream"为基线报增益，官方提示的 SimpleStream 保留一行作参照。数字来自 `kfs/analyze_protocol_tables.py`（离线组合，第十七轮已验证与真跑逐题一致）。

---

## Abstract（草稿，约 180 词；TCSVT 要求 150–200 词）

Streaming video question answering requires a model to answer about what is happening now and about what happened earlier, while the video keeps arriving. A strong recent baseline, SimpleStream, feeds only the latest few frames to an off-the-shelf video large language model and already rivals memory-based systems on real-time questions, but it cannot answer questions whose evidence has left the window. We present ReadPast, a training-free, plug-in extension that keeps the visual input at the recent window and carries the past as text. The same frozen model periodically writes a short caption of the stream into a timestamped timeline, and a present/past gate reads only the question and decides whether the timeline is needed, with its threshold set without any labels by content-free calibration. On OVO-Bench, with the same answer prompt for all methods, ReadPast improves SimpleStream by 1.9, 1.5, 1.1, and 1.3 points on Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3-VL-4B, and LLaVA-OneVision-7B while reading the timeline for only 30–56% of the questions and keeping real-time accuracy within 0.7 points, and it outperforms the structured-memory method D-HSM by 3.1 and 2.2 points under the same protocol **【待补跑：D-HSM 用同一作答提示后确认】**. **【待写：StreamingBench 一句】**

**Index Terms**: Streaming video understanding, video large language models, video question answering, training-free, memory, calibration.

**【待写】**：在 ScholarOne 上选 EDICS 类别。

---

## I. Introduction（约 1.5 页，含图 1）

![Fig. 1](figs/fig1_teaser.png)

*Fig. 1. 【待写：图注】*

写作计划：

- **P1 任务**：流式视频问答的问题可以在任意时刻到来，有的问当前画面（实时题），有的问已经过去的内容（回看题）。基准用 OVO-Bench 和 StreamingBench。视频流不会停，所以成本也重要。
- **P2 动机**：SimpleStream 只把最近 4 帧给冻结的 VLM，就是很强的基线，实时题上常常比记忆方法还好；但证据一旦移出窗口，回看题就答不了。再加历史**帧**会损害对当前画面的感知（Qwen3-VL-8B：加 4 张历史帧，RT −4.03；加 16 张低分辨率帧，RT −5.86），文字时间线只有 −0.81。所以：现在用帧看，过去用文字读。
- **P3 设计问题**：不用标签、不用任务名，怎么判断什么时候该读过去。
- **P4 方法概览**（对应图 1）：自写时间线 + 校准的现在/过去门。不提弃答指令（它属于作答提示，见 IV-A）。
- **P5 结果**：所有方法用同一作答提示时，OVO 比 SimpleStream 高 +1.90 / +1.47 / +1.12 / +1.28，只读 30–56% 的题，RT 变化在 ±0.7 以内；同协议下比 D-HSM 高 +3.05 / +2.20（待 D-HSM 补跑同一提示后确认）。

> **【注】**引言按"三个问题"（过去用什么载体 / 文字记忆要不要结构化 / 什么时候读）改写的方案已提给 REN，待确认后再动这一节。

**Contributions**（框架里的版本）：

1. A training-free, plug-in extension of recent-frame streaming baselines that keeps the visual input at the recent window and carries the past as a self-written text timeline; we show that text history preserves real-time accuracy whereas history frames lower it (Sec. IV-E).
2. A question-only present/past gate whose threshold is set without labels by content-free calibration [Zhao et al.], performing on par with gates calibrated on labeled synthetic queries (Sec. III-C).
3. Experiments on four backbones and two benchmarks, with same-protocol reruns of public training-free methods (Sec. IV-B).

> **【注】**贡献 1 要改：StreamScout 免训练版的写入设计和我们完全相同（同一骨干每 16 秒写一段字幕，作答时看最近 4 帧，见其 §4.1）。写入端应写成 "we adopt"，贡献 1 只保留"文字补过去、帧看现在，并且有实验支撑"这一点。门的那条建议改称 "adaptive memory access"，强调阈值不用任何标签：阅读清单的调研里，没找到别人这样设门阈值的先例。

---

## II. Related Work（约 1.5 页）

规则：放进对比表的方法必须有公开代码，其余只引用。

- **A. Streaming Video Large Language Models**：在线 / 流式 VideoLLM（KV 缓存、记忆库、检索）和基准（OVO-Bench、StreamingBench）。SimpleStream 作为近帧窗口基线。训练类方法（SelectStream、SPARE 等）只引用。**【待写】**
- **B. Memory for Long and Streaming Video**：视觉 token 记忆 vs 文本记忆。离线长视频的字幕记忆（LLoVi、VideoAgent 等）；流式的文本记忆（D-HSM、FOLIO、StreamScout、DeltaStreamer）。我们的位置：时间线由同一个冻结模型写，贡献不在记忆结构。**【待写】**
- **C. Deciding When to Read the Past**：RAG 里的自适应检索（Self-RAG、Adaptive-RAG、FLARE、SKR 等）；流式里的门：ShallowStream（只看问题的 logit 门，阈值用 200 道带标签的合成题定）、FlashBack、WRWS、StreamScout。我们只看问题，阈值用空问题定、不用标签。**【待写】**
- **D. Calibration**：上下文校准（Zhao et al.）、PMI / 领域上下文 / batch 校准；我们校准的是"读不读"这个二元决策的阈值，不是答案的先验。**【待写】**

> **【注】**同刊论文的 Related Work 一般只有 2–3 个子节，阅读清单建议合并成三节：II-A 流式视频理解与记忆；II-B 基于文本的长视频理解（Socratic Models → LangRepo，加上同刊的 FDVS、EventLens）；II-C 自适应检索、校准与弃答。

---

## III. Method（约 2 页，含图 2）

![Fig. 2](figs/fig2_method.png)

*Fig. 2. Overview of ReadPast. 【待写：图注】*

- **A. Problem Setup and the Recent-Window Baseline**：视频流 V；问题 q 在 t 时刻到来，选项为 O。SimpleStream 以 1 fps 采样，用最近 N = 4 帧作答：f(F_t, q, O)。**【待写】**
- **B. Self-Written Timeline**：每 16 秒，同一个冻结 VLM 对这段里均匀采样的 8 帧写一条字幕，得到 (t0, t1, text)。时间线 M_t 是所有 t1 ≤ t 的字幕，放在近帧之前（第十轮实验里，放在前面比放在后面好，三个骨干一致）。成本是每 16 秒一次 8 帧生成；OVO 上时间线平均 1264 token。写字幕的提示放附录。**【待写】**
- **C. Present/Past Gate with Content-Free Calibration**：用固定的路由提示，只输入问题文本（不给帧，也不给选项，因为"Unable to answer"这类选项会泄露任务），得到 p_past = P("past") / (P("past") + P("now"))。阈值 p_cf 是三个空问题 {"N/A", "", "[MASK]"} 上 p_past 的平均，p_past ≥ p_cf 时读时间线。每题只多一次纯文本前向。配图 3。要如实写：LLaVA-OV 上 p_cf 偏低，实时题有 40% 会去读时间线。**【待写】**
- **D. Cost**：写入成本摊到每秒视频；每题多出的成本是门的一次前向加上时间线的 prefill。**【待写】**

> 原"D. Conditional Abstention Instruction"已移出方法（REN 11-05），改为 IV-A 的作答提示。

---

## IV. Experiments（约 4.5 页）

### A. Setup

- **基准**：OVO-Bench 回看题（EPM、ASI、HLD）和实时题（OCR、ACR、ATR、STU、FPD、OJR），共 1468 道选择题（BT 631 + RT 837），avg = (BT + RT)/2，不用 Forward；StreamingBench 实时部分，500 个视频、2500 题。
- **骨干**：Qwen2.5-VL-7B、Qwen3-VL-8B（开发用）；Qwen3-VL-4B（留出验证）；LLaVA-OneVision-7B（只报告）。
- **实现**：1 fps，最近 4 帧；每 16 秒用 8 帧写一段字幕；贪心解码；评分看生成的选项字母。
- **作答提示（所有方法相同）**：选项里有弃答选项时，在问题前加 D-HSM 公开代码里的作答指令（去掉"检索未找到匹配节点"半句，我们没有检索）；没有弃答选项时用基准原始提示。正文一句话交代，原文放附录。覆盖面：OVO 的 187 题（186 道 HLD + 1 道 EPM）；StreamingBench 没有这种选项，提示与原始完全相同。
  - 拟写："All methods we run use the same answer prompt: when the options include an abstention choice, we prepend the answer instruction from D-HSM's public implementation [D-HSM]; otherwise the benchmark's original prompt is used (Appendix X)."
- **基线**：
  - SimpleStream：我们复现，同一作答提示（主表基线）；另列一行官方提示（Qwen2.5 上 64.75，论文值 65.13）作参照；
  - D-HSM：公开代码，同协议重跑（第十八轮，用它发布的默认提示，即每道选择题都加指令）。**【待补跑】**D-HSM 按同一条件提示的结果：第十五轮已有 Qwen2.5 回看题的"不加指令"运行，缺实时题和 Qwen3，补跑后离线拼；
  - ShallowStream：用论文值，标注"其自身协议"。

**【待写】**：正文。

### B. Main Results

**Table I. OVO-Bench（标准总分；括号内为相对"同一作答提示的 SimpleStream"的点数增益）**

| Backbone | Method | EPM | ASI | HLD | RT | Avg |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | SimpleStream（原始提示，参照） | 53.5 | 61.5 | 40.3 | 77.7 | 64.75 |
| | SimpleStream | 53.5 | 61.5 | 65.1 | 77.7 | 68.88 |
| | **ReadPast** | 53.9 | 64.2 | 73.1 | 77.8 | **70.78 (+1.90)** |
| Qwen3-VL-8B | SimpleStream（原始提示，参照） | 52.2 | 58.1 | 51.6 | 81.4 | 67.66 |
| | SimpleStream | 52.2 | 58.1 | 76.9 | 81.4 | 71.87 |
| | **ReadPast** | 51.9 | 69.6 | 73.7 | 81.7 | **73.34 (+1.47)** |
| Qwen3-VL-4B | SimpleStream（原始提示，参照） | 54.6 | 57.4 | 51.6 | 80.3 | 67.42 |
| | SimpleStream | 54.2 | 57.4 | 76.3 | 80.3 | 71.48 |
| | **ReadPast** | 54.2 | 62.8 | 78.0 | 80.2 | **72.60 (+1.12)** |
| LLaVA-OV-7B | SimpleStream（原始提示，参照） | 55.6 | 54.7 | 20.4 | 74.7 | 59.15 |
| | SimpleStream | 55.6 | 54.7 | 27.4 | 74.7 | 60.32 |
| | **ReadPast** | 58.2 | 57.4 | 31.7 | 74.1 | **61.60 (+1.28)** |

**【待写】**：RT 按 6 个任务拆成分列（数据现成，从 `kfs/analyze_round17.py` 取）。

**StreamingBench 实时部分**（没有弃答选项，弃答指令不触发）：

| | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3-VL-4B | LLaVA-OV-7B |
|---|---|---|---|---|
| SimpleStream | 79.76 | 80.28 | 79.92 | 74.64 |
| ReadPast | 79.56 (−0.20) | 81.64 (+1.36) | 81.40 (+1.48) | 74.92 (+0.28) |

### C. Comparison with Training-Free Methods

- **D-HSM**：同协议重跑（第十八轮，`kfs/analyze_round18_dhsm.py`），同时列论文值。OVO：D-HSM 67.73 / 71.14 vs ReadPast 70.78 / 73.34（Qwen2.5 / Qwen3-8B）；SB-RT：D-HSM 80.51 / 81.91 高于 ReadPast 79.56 / 81.64，如实报。D-HSM 目前用的是它自带的"每题都加"提示，**【待补跑】**同一条件提示后替换。
- **ShallowStream**：论文值，标"其自身协议"。Qwen3-8B OVO 69.5 / SB-RT 78.2；LLaVA-OV OVO 62.2 / SB-RT 75.5。

> **【注】**阅读清单新找到四个有代码、已录用的免训练方法：OASIS（CVPR 2026）、FluxMem（CVPR 2026）、MEMO（ACM MM 2026，只报了 RT）、HERMES（ACL 2026）。它们放论文值还是重跑，等你定。

### D. Ablation Study

**Table II. 时间线与门的消融（OVO 标准总分；所有行用同一作答提示；括号内为读时间线的题目比例）**，数据已有（离线组合）：

| | Qwen2.5 | Qwen3-8B | Qwen3-4B | LLaVA-OV |
|---|---|---|---|---|
| SimpleStream（不读） | 68.88 | 71.87 | 71.48 | 60.32 |
| 总是读时间线 | 70.96 (100%) | 72.95 (100%) | 72.90 (100%) | 61.47 (100%) |
| 门：固定阈值 0.5 | 70.97 (36%) | 72.91 (22%) | 72.76 (44%) | 60.77 (5%) |
| 门：ShallowStream 10-shot + 我们的空问题校准 | 70.89 (87%) | 73.37 (70%) | 73.02 (73%) | 61.27 (92%) |
| **门：空问题校准（ReadPast）** | 70.78 (32%) | 73.34 (30%) | 72.60 (43%) | 61.60 (56%) |

RT（同一顺序）：

| | Qwen2.5 | Qwen3-8B | Qwen3-4B | LLaVA-OV |
|---|---|---|---|---|
| SimpleStream | 77.73 | 81.36 | 80.30 | 74.73 |
| 总是读 | 76.92 | 80.54 | 79.64 | 73.17 |
| ReadPast | 77.84 | 81.65 | 80.19 | 74.06 |

- 读法：门在总分上和"总是读"持平（−0.18 / +0.39 / −0.30 / +0.13），它的作用是 RT 不掉（总是读 RT −0.7 到 −1.6）且只读 30–56% 的题。这一节按"读得少、实时不掉"写，不写成"门提高准确率"。固定 0.5 在 LLaVA 上只读 5%，说明阈值必须校准。
- 弃答指令"按选项触发 vs 每题都加"的对比**不进论文**（REN 11-05），数据留在笔记本第十八轮。
- 正文形式：按 REN 11-05 的讨论，消融用图（柱上标数字），完整表放附录。

还要放进来的消融，数据都已有：

- 时间线放在近帧前还是后（第十轮；需在同一作答提示下重算，离线即可）。

还缺的消融：

- **字幕间隔扫描**：8 / 16 / 32 秒。**【待跑】**
- **空问题输入的组合**：单用 "N/A"、""、"[MASK]" 中的一个 vs 三个平均。离线就能算，不用 GPU。**【待做】**

### E. Analysis

1. **文字 vs 历史帧**（Qwen3-8B）：文字时间线 RT −0.81；4 张历史帧 −4.03；16 张低分辨率帧 −5.86。Qwen2.5 上结论相同。
2. **时间线内容的对照**：自己的时间线减去别的视频的时间线，EPM +6.7（Qwen2.5）/ +5.4（Qwen3）。说明模型确实在用内容，不是"多了一段文字"带来的。
3. **记忆能覆盖多远**：增益集中在证据距提问约 90 秒以内的题。这一条写进局限。
4. **门的校准**：见图 3。

![Fig. 3](figs/fig3_calibration.png)

*Fig. 3. Content-free calibration of the present/past gate on four backbones. 【待写：图注】*

**【待写】**：正文。

### F. Efficiency

暂缓（REN 11-04：暂时不需要效率表）。

### G. Qualitative Results and Failure Cases

**【待写】**：选 2–3 个例子，都从现有结果里挑：

- 一道靠时间线答对的回看题；
- 一道留在帧上作答的实时题；
- 一个失败例子：证据太久远，或者根本没写进时间线。

---

## V. Limitations

- 记忆覆盖距离短：增益集中在证据出现后约 1.5 分钟以内。
- 写入覆盖是瓶颈：没写进时间线的信息读不出来。
- 校准迁移到 LLaVA-OV 不如 Qwen 系列：实时题有 40% 会去读时间线。

**【待写】**

## VI. Conclusion

**【待写】**

---

## References（`refs.bib` 现状）

- SimpleStream, arXiv:2604.02317 （标题、作者待填）
- D-HSM: Dynamic Hub-and-Spoke Memory for Streaming Video Understanding, Findings of EMNLP 2026, arXiv:2608.30294 （作者待填）
- ShallowStream, arXiv:2609.02780 （标题、作者待填）
- T. Z. Zhao, E. Wallace, S. Feng, D. Klein, S. Singh. Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML 2021.
- 待补：OVO-Bench、StreamingBench、Qwen2.5-VL、Qwen3-VL、LLaVA-OneVision，以及阅读清单里的文献。
