# 流式 / 在线视频理解：ReadPast（TCSVT）Related Work 与对比的必读文献（2026-11-03）

> 范围：在 /data/REN/streamkv/RELATED.md（10-24、11-01、11-03 三次调研）之上补充，不重复其内容。
> 来源：Semantic Scholar 引用 API（2026-11-03 拉取）；arXiv API、abs 页、HTML 全文（表格用脚本从 HTML 表格原样抽出）；GitHub API 文件树与 README（2026-11-03 抓取）；对 7 个仓库做了浅克隆并 grep 代码。
> 统一口径：OVO = (RT + BT) / 2，不含 Forward。下面的 OVO 都按论文表里的 RT、BT 重新算过；论文自报的 overall 口径不同（如含 Forward、加权），不直接用。
> 我们的 OVO：Qwen2.5-VL-7B 70.78 / Qwen3-VL-8B 73.34 / Qwen3-VL-4B 72.60 / LLaVA-OV-7B 61.60。
> "代码"一栏：只有 GitHub 文件树里确实有推理代码（.py/.sh）的才算有；只有 README、项目页或"代码将发布"的都记为 no code。

## 分级阅读清单（A 必读必比 / B 必引定位 / C 可选）

### Takeaway
RELATED.md 漏掉了一批**同设定、免训练、有代码、已录用、用的是我们骨干**的方法，必须补进对比表：OASIS（CVPR 2026）、FluxMem（CVPR 2026）、MEMO（ACM MM 2026）、HERMES（ACL 2026）。**StreamScout 免训练版和 ReadPast 的设计最接近**（同一骨干每 16 秒写一条字幕 + 近 4 帧），但没有代码，只能引用并用论文数字对比。在同一"近 4 帧"接口下，训练方法 ViSS（Qwen3-VL-8B，OVO 73.20）已经和我们的 73.34 持平，审稿人可能会问。

### Cited Findings

**A 级：必读、必比（免训练、冻结视频 LLM、有公开推理代码、有 OVO / SB 数字、骨干与我们重合）**

| 方法 | arXiv / 会议 | 代码 | 骨干 | OVO RT / BT → (RT+BT)/2 | SB 实时 | 机制（一句） | ReadPast 的区别（一句） |
|---|---|---|---|---|---|---|---|
| SimpleStream | [2604.02317](https://arxiv.org/abs/2604.02317)；arXiv 注释里只有项目页，未见录用 | [EvolvingLMMs-Lab/SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream)（有 OVO / SB 评测脚本） | Qwen2.5-VL-7B、Qwen3-VL-8B | 论文值 78.4/51.9 → 65.13；81.4/54.0 → 67.70（RELATED.md 记录） | 80.59（Qwen3，[README](https://github.com/EvolvingLMMs-Lab/SimpleStream)） | 只喂最近 4 帧 | 我们的基座；在它上面加文本时间线、门控和弃答指令 |
| D-HSM | [2608.30294](https://arxiv.org/abs/2608.30294)，EMNLP 2026 Findings | [Oshikaka/D-HSM](https://github.com/Oshikaka/D-HSM)（experiments/evaluate_ovo.py、evaluate_streamingbench.py） | Qwen2.5-VL-7B、Qwen3-VL-8B | 79.0/(52.9+63.5+72.0)/3 → 70.9；81.6/… → 71.45（表 2，RELATED.md 已核） | 82.5 / 83.1 | 同一 VLM 写六字段结构化观察，组成实体 hub-spoke 图，检索后与近 4 帧一起作答，带弃答提示 | 我们只借用它的弃答指令，并且只在选项里有弃答项时启用；记忆是线性时间线，不建图；门控只看问题 |
| ShallowStream | [2609.02780](https://arxiv.org/abs/2609.02780)，未见录用 | [CURRENTF/ShallowStream](https://github.com/CURRENTF/ShallowStream)（只有 smoke 推理，无 OVO/SB 脚本） | Qwen3-VL-8B、LLaVA-OV-7B | 80.9/58.1 → 69.5；75.4/49.0 → 62.2（表 1，RELATED.md） | 78.2 / 75.5 | 浅层 KV 建索引 + 只看问题文本的 logit 门 + 浅层注意力选帧 | 我们取回的是文本、不是帧；阈值用空问题标定，不用合成问题；用户已决定按论文值引用 |
| **OASIS（新）** | [2604.17052](https://arxiv.org/abs/2604.17052)，CVPR 2026 | [Solus-sano/OASIS](https://github.com/Solus-sano/OASIS)（src/scripts/eval.py；metadata 里有 eval_ovo_bench.json、StreamingBench） | Qwen2.5-VL-7B、Qwen3-VL-8B | 67.26/52.61 → **59.94**；78.14/57.21 → **67.67**（[表 1](https://arxiv.org/html/2604.17052)，基线为作者以 0.5 fps 重跑） | 70.59 / 78.22（表 2） | 按 2 秒块切分，MLLM 写事件摘要，组织成事件森林 + 中期缓冲 + 近窗；作答分粗到细两阶段，先粗看，再调 rag_retrieval 工具取片段 | 我们不建层级，也不让模型自己调工具，门控是一次前向；在同骨干上分数高出约 +5.7（Qwen3）和 +10.8（Qwen2.5） |
| **FluxMem（新）** | [2603.02096](https://arxiv.org/abs/2603.02096)，CVPR 2026 | 仓库已迁到 [ShareLab-SII/FluxMem](https://github.com/ShareLab-SII/FluxMem)（由 YiwengXie/FluxMem 重定向；GitHub API 描述为 "[CVPR 2026] FluxMem"；文件树未逐项核对） | Qwen2.5-VL-7B | RT 67.2（[表 1](https://arxiv.org/html/2603.02096)）；BT 47.2 → 57.2（BT 引自 [MemEvo 主表](https://arxiv.org/html/2609.36581)，属二手） | 76.4（表 1） | 自适应分层记忆（免训练；另有 +SFT 版本） | 视觉 token 记忆，对比我们的文本记忆 |
| **MEMO（新）** | [2609.38900](https://arxiv.org/abs/2609.38900)，ACM MM 2026 | [Cecilia384/MEMO](https://github.com/Cecilia384/MEMO)（configs/baselines 下有 ovobench_qwen25/qwen3/llava_7b 配置） | Qwen2.5-VL-7B、Qwen3-VL-8B、LLaVA-OV-7B、LLaVA-OV-0.5B | **只报 RT**：66.9 / 76.0 / 66.6（[表 1](https://arxiv.org/html/2609.38900)）；没有 BT，算不出 (RT+BT)/2 | 78.5 / 83.7 / 73.2（表 1） | 多层感知切出语义块，全局和实体级特征作索引，高分辨率帧另存，按问题检索后回看 | 它的基线是 1 fps 全量帧（Qwen3 RT 70.1），比 SimpleStream 近 4 帧（81.4）低很多，所以 RT 增益不能直接比；有代码，可以用来补跑 BT |
| HERMES | [2601.14724](https://arxiv.org/abs/2601.14724)，ACL 2026 Main | [haowei-freesky/HERMES](https://github.com/haowei-freesky/HERMES)（data/ovobench/ovobench_realtime_backeward.json、streamingbench_realtime.json，与我们的协议一致） | Qwen2.5-VL-7B、Qwen3-VL-8B、LLaVA-OV-7B | 官方值（经 MemEvo 表 5 转引）：69.0/49.4 → 59.2；73.3/49.3 → 61.3；ShallowStream 重跑的 LLaVA-OV：66.3/49.9 → 58.1 | 79.4 / 81.3（官方值，经 MemEvo 转引） | 把 KV cache 当作分层记忆 | 训练-free 的 KV 路线代表，我们不动 KV |
| SAVEMem | [2605.07897](https://arxiv.org/abs/2605.07897)，未见录用 | [wuhang03/savemem](https://github.com/wuhang03/savemem)（evaluation/ovobench、streamingbench、odvbench） | Qwen2.5-VL-7B | 62.7（RELATED.md） | — | 三级视觉记忆 + 伪问题库定显著性 + 门控 | 它的门控阈值按 RT/BT 类别分别设，用到了类别标签；我们的门不用标签 |
| ObjectStream | [2607.28312](https://arxiv.org/abs/2607.28312)，未见录用 | [DMK041218/ObjectStream](https://github.com/DMK041218/ObjectStream)（qwen2.5-vl-7b/evaluation/ovobench、streamingbench） | Qwen2.5-VL-7B | 61.4（RELATED.md） | — | 潜在物体作记忆锚点 | 视觉潜在记忆 |

**A- 级：必读、必须用论文值对比，但没有代码（按用户规则不进主表，可放引用或附表）**

| 方法 | arXiv / 会议 | 代码 | 骨干 | OVO RT/BT → (RT+BT)/2 | SB 实时 | 机制 | ReadPast 的区别 |
|---|---|---|---|---|---|---|---|
| **StreamScout（免训练版）** | [2609.00291](https://arxiv.org/abs/2609.00291)，未见录用 | no code（2026-11-03 GitHub 搜索无匹配） | Qwen2.5-VL-7B / Qwen3-VL-8B / LLaVA-OV-2-8B | 75.27/50.08 → 62.67；77.66/59.11 → 68.38；76.46/61.65 → 69.05（[表 1](https://arxiv.org/html/2609.00291)） | 77.7（Qwen2.5，表 2） | **同一骨干每 Δ=16 秒写一条字幕**作为时间线；提问时分三级逐步加视觉：近 4 帧 → 回看 8 帧 → 检索 12 帧，由模型自己发 token 决定是否升级（§3.2、§4.1） | **和我们最像**。区别：我们不做视觉升级，门控只看问题、阈值用空问题标定，只决定读不读时间线；在 Qwen2.5 上高 +8.1，Qwen3 上高 +5.0 |
| FOLIO | [2607.13298](https://arxiv.org/abs/2607.13298) | no code | Qwen2.5-VL-7B / Qwen3-VL-8B | 77.0/57.2 → 67.1；82.0/69.1 → 75.55（RELATED.md） | 75.3 / 81.2（多轮完整设置，协议不同） | 每 8 秒写实体中心的 JSON 记录 + 关键帧缓存 | 我们是线性自由文本，没有实体合并；FOLIO 的 Qwen3 分数高于我们，但领先主要来自 HLD（RELATED.md 11-02） |
| WRWS | [2609.37345](https://arxiv.org/abs/2609.37345) | no code | Qwen2.5-7B / Qwen3-8B | 63.86 / 65.37（RELATED.md） | — | 检索熵决定检索还是只看近帧 | 我们取回文本，靠门控决定 |
| **MemEvo（新）** | [2609.36581](https://arxiv.org/abs/2609.36581) | no code（HTML 中无仓库链接，GitHub 搜索无匹配） | Qwen2.5-VL-7B / Qwen3-VL-8B | 73.8/52.0 → 62.9；78.2/59.3 → 68.75（[主结果表](https://arxiv.org/html/2609.36581)，1 fps 基线 59.9/44.7、68.2/46.4） | 77.5 / 81.4 | 用 LLM 自动搜索"记忆程序"（一个 DSL，表达表示、准入、保留、合并、预算、检索），VLM 冻结 | 我们是人工设计的极简方案；同骨干 OVO 高 +7.9 / +4.6 |
| **Foresight（新）** | [2610.03123](https://arxiv.org/abs/2610.03123) | no code（摘要写 "will be made publicly available"） | Qwen3-VL-8B（冻结） | 75.02/55.10 → 65.06（[表 3](https://arxiv.org/html/2610.03123)，基线 61.6/41.0） | 79.30（表 2，RTVU） | 两个共享权重的孪生 LLM，一个跑在流前面做预测和计算规划 | 方向不同（规划未来计算）；我们高 +8.3 |

**B 级：必引，用于定位（训练方法、基础流式模型、检索门控 / 弃答相关工作）**

| 方法 | arXiv / 会议 | 代码 | 是否训练 | 骨干 | OVO (RT+BT)/2 | SB 实时 | 机制 / 和我们的关系 |
|---|---|---|---|---|---|---|---|
| **ViSS（新，引用 SimpleStream）** | [2610.34926](https://arxiv.org/abs/2610.34926) | [yihaohu0118/ViSS](https://github.com/yihaohu0118/ViSS)（experiments/evaluate_qwen3vl_recent4_full.py；权重在 [HF collection](https://huggingface.co/collections/mis2/viss-6adc8caf041ae0c501c9a11b)） | 训练（SFT） | Qwen3-VL-8B | 近 4 帧：82.20/64.21 → **73.20**；+TVS：81.27/69.37 → 75.32；同表基座近 4 帧 81.38/51.45 → 66.41（[表 1](https://arxiv.org/html/2610.34926)） | 84.77 / 86.33（表 2） | 用"可见状态"探针做监督（事件身份、顺序、累计、关系变化、答案是否充分），HLD 涨得最多（37.6 → 75.3）。在同样的近 4 帧接口下和我们持平，但需要训练 |
| SelectStream | [2606.16353](https://arxiv.org/abs/2606.16353)，**NeurIPS 2026**（arXiv v2 注释，2026-09-26） | [Haonan-Ge/selectstream](https://github.com/Haonan-Ge/selectstream)（main/cli/infer.py、eval.py） | 训练记忆模块，VLM 冻结 | Qwen2.5-7B / Qwen3-8B | 70.95 / 72.48（RELATED.md） | — | 训练的潜在记忆图；我们免训练，分数和它相当（Qwen2.5 −0.2，Qwen3 +0.9） |
| WTI（Watch-Think-Interact，新） | [2609.37035](https://arxiv.org/abs/2609.37035) | no code | 训练（SFT + Stream-GDPO 强化学习） | Qwen3-VL-8B 底座（表 1 名为 WTI-8B） | 76.9/73.4 → 75.15（[表 1](https://arxiv.org/html/2609.37035)） | 83.3 | 自然语言记忆条目 + 源视频时间段，可以回看；"答 / 等 / 回看"由策略学出 |
| EGSD（新） | [2609.36803](https://arxiv.org/abs/2609.36803) | no code | 训练（自蒸馏 + 强化学习） | Qwen2.5-VL-7B / Qwen3-VL-8B | RT 70.0 / 73.4（[表 1](https://arxiv.org/html/2609.36803)）；BT 在表 2 中，未抽取 | 78.4 / 79.8 | 把记忆训练成可核验的 Event 增量更新 |
| EventMemAgent（新） | [2602.15329](https://arxiv.org/abs/2602.15329)，**ECCV 2026**（[README](https://github.com/lingcco/EventMemAgent)） | [lingcco/EventMemAgent](https://github.com/lingcco/EventMemAgent)（权重 EventMemAgent-8B 已放出） | 训练（多轮 GRPO） | Qwen3-VL-8B（策略）+ Qwen3-VL-4B（写字幕） | 68.29/58.03 → 63.16（引自 [ViSS 表 1](https://arxiv.org/html/2610.34926)）；README 自报 60.75 | 77.00（README） | 事件中心的长期记忆 + OCR / 检测工具；学会何时用哪种工具 |
| SPARE | [2610.14671](https://arxiv.org/abs/2610.14671) | no code | 训练 | Qwen3-8B | 77.78（RELATED.md） | — | 训练出来的"近帧够不够"门 + 学会弃答 |
| StreamScout-S / -R | [2609.00291](https://arxiv.org/abs/2609.00291) | no code | 训练（LoRA 蒸馏 / GRPO） | Qwen2.5 / Qwen3 | -R：75.87/67.35 → 71.61；79.45/72.90 → 76.18（表 1） | — | 升级策略靠训练 |
| StreamTTT | [2608.13416](https://arxiv.org/abs/2608.13416) | [zeyun-zhong/StreamTTT](https://github.com/zeyun-zhong/StreamTTT) | 训练 | **Qwen3-VL-4B** | 69.0（SS-4B 66.06）（RELATED.md） | — | 是 Qwen3-VL-4B 上唯一给出 SimpleStream-4B 数字的参照 |
| MaLoW | [2610.22070](https://arxiv.org/abs/2610.22070) | [dragonlzm/MaLoW](https://github.com/dragonlzm/MaLoW)（仓库 2026-10-30 仍在更新） | 训练 | Qwen3-VL-4B、Qwen3.5 | 69.3（RELATED.md） | — | 用超网络把历史写成 LoRA；另带一个与问题无关的文本摘要变体（见下文"文本记忆"一节） |
| OneStreamer | [2610.01762](https://arxiv.org/abs/2610.01762) | [MCG-NJU/OneStreamer](https://github.com/MCG-NJU/OneStreamer)（Eval/Proactive_Eval/OneStreamer/scripts/run_ovobench.sh） | 训练（SFT） | Qwen3-VL-4B | 81.4/77.9 → 79.65（引自 [TwinPast 首表](https://arxiv.org/html/2610.41983)） | 86.9（同上） | 训练模型自己写带时间的字幕记忆，HLD 99.5 |
| StreamOPD | [2608.16320](https://arxiv.org/abs/2608.16320) | [UniX-AI-Lab/StreamOPD](https://github.com/UniX-AI-Lab/StreamOPD) | 训练 | Qwen3.5-4B | 见 RELATED.md | — | 推理仍是近 4 帧 |
| StreamClarify（新） | [2611.01263](https://arxiv.org/abs/2611.01263)，**WACV 2027** | no code（[zhuchens-uiuc/StreamClarify-page](https://github.com/zhuchens-uiuc/StreamClarify-page) 只是静态项目页） | 免训练 | LLaVA-OV-1.5、InternVL-3.5、Qwen3-VL | 不可比：只用 OVO 的 281 题和 StreamBench 的 1,556 题自建 StreamClarify-QA（[表 2](https://arxiv.org/html/2611.01263)） | — | **作答前先做一次二元判断，用其熵作不确定性门**，决定直接答还是请求澄清；含不可答问题。与我们的门控和弃答都相关，必须在 related work 里区分 |
| FlashBack | [2610.01192](https://arxiv.org/abs/2610.01192) | no code | 免训练 | StreamingVLM-7B / Mage-VL-4B | 57.9 / 65.6 | — | 只看问题文本做结构化路由（RELATED.md 11-01） |

**C 级：可选**
- 效率类：StreamingTOM（[2510.18269](https://arxiv.org/abs/2510.18269)，CVPR 2026，代码 [YIGE24/StreamingTOM](https://github.com/YIGE24/StreamingTOM)）；Video-MemComp（[2610.32784](https://arxiv.org/abs/2610.32784)，ACM MM 2026，免训练 KV 压缩，评测用 MLVU / VideoMME / EgoSchema，代码 [hhnk666/videomemorycompress](https://github.com/hhnk666/videomemorycompress)）；StreamAhead（[2610.25551](https://arxiv.org/abs/2610.25551)，自适应预计算，TTFT 最多降 10.4 倍）；StreamFlow（[2608.10949](https://arxiv.org/abs/2608.10949)，骨干 Qwen3.5-9B，SB 总分 67.73，同骨干 RTVU 81.55，同表 SimpleStream 79.02，[HTML 对比表](https://arxiv.org/html/2608.10949)；未发现代码；没有 OVO）；免训练加速综述 [2610.36327](https://arxiv.org/abs/2610.36327)（引用 SimpleStream）。
- 主动 / 前向类：PACE（[2610.28323](https://arxiv.org/abs/2610.28323)，MMM 2027，免训练，只做 OVO Forward 和 ProactiveVideoQA）；ProV-Mem（[2610.23998](https://arxiv.org/abs/2610.23998)，免训练，ProactiveVideoQA）；AskStream（[2610.38678](https://arxiv.org/abs/2610.38678)，代码 [wanglu-cs/AskStream](https://github.com/wanglu-cs/AskStream)）；SyncPro（[2610.32132](https://arxiv.org/abs/2610.32132)）；Live-OPSD（[2610.36959](https://arxiv.org/abs/2610.36959)，Qwen3-Omni-30B-A3B，OVO 65.84）。
- 名字冲突：已有一篇方法叫 **StreamKV**（[2511.07278](https://arxiv.org/abs/2511.07278)，代码 [sou1p0wer/StreamKV](https://github.com/sou1p0wer/StreamKV)，LLaVA-OV-7B，SB 实时 68.8，见 [MEMO 表 1](https://arxiv.org/html/2609.38900)），和我们的工作目录同名；论文里只用 ReadPast 这个名字。
- 文本记忆但不是流式基准：CAM（[2609.06504](https://arxiv.org/abs/2609.06504)，代码 [Jake-Tian/CAM](https://github.com/Jake-Tian/CAM)，M3-Bench）；Retrieval-Augmented Online Textual Memory（[2610.31284](https://arxiv.org/abs/2610.31284)，QAEgo4D：检索预算越大，时间覆盖越好，但精度可能反而下降）。

### Inferences
- 主对比表最少应补：OASIS（Qwen2.5 / Qwen3）、HERMES（三个骨干）、FluxMem（Qwen2.5）、MEMO（三个骨干，但需要用它的代码补跑 BT）。这四篇都是已录用、免训练、有代码，审稿人最容易问"为什么不比"。
- 这四篇加上 D-HSM，在同骨干上都低于我们：Qwen3 上 OASIS 67.67、HERMES 61.3、D-HSM 71.45，我们 73.34；Qwen2.5 上 OASIS 59.94、FluxMem 57.2、D-HSM 70.9，我们 70.78，**和 D-HSM 只差 0.1，等于持平**。
- LLaVA-OV-7B 上 ShallowStream 62.2 高于我们 61.60。但两者的 SimpleStream 基线不同（它重跑的 60.3，我们复现的 59.15），按"相对各自基线"算，我们 +2.45，它 +1.9（RELATED.md 11-02）。
- **StreamScout 免训练版必须在方法节和 related work 里明确区分**：写入设计（同一骨干、16 秒一段、近 4 帧）和我们几乎一样。我们能主张的差别只剩读取端：门控只看问题、空问题标定、不做视觉升级；以及分数（同骨干 +5 到 +8）。
- ViSS 说明：只训练、不加记忆，近 4 帧在 Qwen3-VL-8B 上就能到 73.2，而且主要靠 HLD。我们可以写成"免训练达到训练方法的水平"；但按用户规则，HLD 不单独分析。

### Gaps
- MEMO 没有报 OVO Backward，FluxMem 的 BT 来自 MemEvo 的转引；两者都要用各自代码补跑，或从原文表 2 逐项核对。
- HERMES 的官方 OVO / SB 数字是从 MemEvo 表 5 转引的，没从 HERMES 原文表格逐字核对（HERMES 的 arXiv HTML 没有抽到表格）。
- 没核对 FluxMem 新仓库（ShareLab-SII/FluxMem）的文件树，也没核对 apple-aiml-research/ml-streambridge 的内容。
- OASIS 的基线是作者用 0.5 fps 全量帧重跑的，不是 SimpleStream；它相对 SimpleStream 的增益未知。

## 2026-10-24 以来新出现的、引用或对比 SimpleStream / D-HSM / ShallowStream 的论文

### Takeaway
Semantic Scholar 上引用 SimpleStream 的论文从 36 篇增加到 41 篇。新增的里只有 **ViSS** 是同设定（近 4 帧接口、OVO + SB），但它需要训练。D-HSM 和 ShallowStream 的引用者都不是流式 QA 方法论文。另外有一批不引用 SimpleStream、但完全同设定的工作（MEMO、MemEvo、OASIS、FluxMem、Foresight），只靠引用链找不到。

### Cited Findings
- SimpleStream 当前被引 41 篇。10-24 之后新增 5 篇：ChronoReplay（2610.34283）、Supervising the Visible State / ViSS（2610.34926）、Training-Free Inference Acceleration for Video LLMs（2610.36327）、AskStream（2610.38678）、TwinPast（2610.41983）—— [S2 citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2604.02317/citations?fields=title,externalIds,publicationDate)
- 10-24 之前就已在引用列表、但 RELATED.md 没收录的：StreamFlow（2608.10949）、LLaVA-OneVision-2（2605.25979）、Q-GeoMem（2605.27318）、Harnessing Streaming Video in the Wild（2606.08615）、Towards Spatial Supersensing in the Wild（2607.13681，ECCV 2026）、SimpleMemVLA（2609.05533）、Simple Agentic Memory for Generalist Robot Policies（2609.36595）—— [S2 citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2604.02317/citations?fields=title,externalIds,publicationDate)
- D-HSM 只被 4 篇引用：Simple Agentic Memory for Generalist Robot Policies、APM-Bench、When Can Text Replace Vision?（图表推理）、LEAP（音视频长时检索），都不是流式 QA 方法 —— [S2 citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2608.30294/citations?fields=title,externalIds,publicationDate)
- ShallowStream 只被 2 篇引用：WRWS、FlashBack（RELATED.md 已收）—— [S2 citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2609.02780/citations?fields=title,externalIds,publicationDate)
- ViSS：在"同样的近帧接口"下，用 source-grounded 探针做监督（事件身份、顺序、累计、关系变化、答案是否充分）；推理时用 TVS，从近帧按需扩展到稀疏事件视图和紧凑历史视图；"supervision particularly benefits hallucination detection" —— [arXiv 2610.34926](https://arxiv.org/abs/2610.34926)
- ViSS 三个随机种子的配对差：OVO +1.72、HLD +7.71、SB −0.17（相对严格匹配的通用时间监督，表 11）—— [arXiv HTML 2610.34926](https://arxiv.org/html/2610.34926)
- ChronoReplay：同一组作者，训练方法，针对流式空间智能（OVO-S-Bench、OST-Bench、UCS-Bench），OST-Bench +13.41 —— [arXiv 2610.34283](https://arxiv.org/abs/2610.34283)
- TwinPast：在 OVO-Bench 和 StreamingBench 抽样的历史题中，32% 只靠近帧就够；模型只看问题和选项也能答对很多题。新基准含 350 对人工审核的视频对，每对改变关键历史、保持近帧不变；最强的流式系统 PairAcc 49.14% —— [arXiv 2610.41983](https://arxiv.org/abs/2610.41983)
- TwinPast 首表：Qwen3-VL-8B 近 4 帧 OVO RT 82.55 / BT 51.31；"Query only"（不给视频）HLD 69.89、BT 52.09；DeepSeek-Flash Query only HLD 88.71 —— [arXiv HTML 2610.41983](https://arxiv.org/html/2610.41983)；代码 [Yh1zzzzzzz/TwinPast_code](https://github.com/Yh1zzzzzzz/TwinPast_code)
- AskStream：在不确定时主动提问（何时问、问什么），有代码和数据集 —— [arXiv 2610.38678](https://arxiv.org/abs/2610.38678)
- 用 arXiv 全文检索（OVO-Bench / StreamingBench / streaming video understanding，按提交时间倒序）补到的同设定新作，S2 引用链里没有：MEMO（09-30）、MemEvo（09-29）、Foresight（10-02）、EGSD（09-29）、WTI（09-29）、StreamClarify（10-31）、Video-MemComp（10-23）、PACE（10-20）、ProV-Mem（10-15）、StreamAhead（10-18）、TRACE（09-25）—— [arXiv API 查询](https://export.arxiv.org/api/query?search_query=abs:%22OVO-Bench%22+OR+abs:StreamingBench&sortBy=submittedDate&sortOrder=descending)

### Inferences
- 10-24 之后没有出现新的"免训练、近帧 + 文本记忆"方法。真正的新压力来自两处：已录用会议的免训练方法（OASIS、FluxMem、MEMO、HERMES），以及近帧接口下的训练方法（ViSS）。
- TwinPast 是写引言和动机的有力引用：现有基准里很多"历史题"其实不需要历史。如果我们声称文本时间线带来了"真正的历史利用"，审稿人可能要求在 TwinPast 上验证（有代码，350 对，规模小）。

### Gaps
- S2 收录有延迟：10-30 以后的论文可能还没进引用列表。arXiv 检索只覆盖了摘要里出现关键词的论文。
- ChronoReplay、AskStream、Harnessing Streaming Video in the Wild 只读了摘要。

## RELATED.md 中 A / B 组论文：现在是否有代码、是否已录用

### Takeaway
与 RELATED.md 11-03 的核查结果相比，代码情况**没有变化**：WRWS、FOLIO、StreamScout、FlashBack、LatentStream、ReMem、FrameBlend、SPARE 仍然没有代码。LatentStream 有一个同名仓库，但里面只有 README。录用方面，唯一的新变化是 **SelectStream 被 NeurIPS 2026 接收**；D-HSM 仍是 EMNLP 2026 Findings；其余都只有 arXiv 版本。

### Cited Findings
- arXiv API 返回的 comment / journal_ref 字段（2026-11-03）：
  - SelectStream v2（2026-09-26）注释 "NeurIPS 2026" —— [arXiv 2606.16353](https://arxiv.org/abs/2606.16353)
  - D-HSM 注释 "Accepted to Findings of EMNLP 2026" —— [arXiv 2608.30294](https://arxiv.org/abs/2608.30294)
  - 以下论文的注释为空或只有页数 / 项目页：WRWS、ShallowStream（v2，09-30）、FOLIO（"28 pages"）、StreamScout、FlashBack、SAVEMem、LatentStream、ReMem、ObjectStream（只有代码链接）、DeltaStreamer、FrameBlend、SPARE、StreamTTT、StreamOPD（项目页）、MaLoW、OneStreamer（项目页）、SVMemAgent，以及 SimpleStream 本身（项目页）—— 见各 abs 页，如 [2609.37345](https://arxiv.org/abs/2609.37345)、[2607.13298](https://arxiv.org/abs/2607.13298)、[2609.00291](https://arxiv.org/abs/2609.00291)、[2610.01192](https://arxiv.org/abs/2610.01192)、[2604.02317](https://arxiv.org/abs/2604.02317)
- 已确认的仓库（GitHub API 文件树，2026-11-03）：
  - [Oshikaka/D-HSM](https://github.com/Oshikaka/D-HSM)：25 个文件，含 experiments/evaluate_ovo.py、evaluate_streamingbench.py
  - [CURRENTF/ShallowStream](https://github.com/CURRENTF/ShallowStream)：46 个 py/sh，含 qwen3vl_v3_ovobench.json 配置和 examples/smoke_inference.py
  - [wuhang03/savemem](https://github.com/wuhang03/savemem)：evaluation 下有 ovobench、streamingbench、odvbench
  - [DMK041218/ObjectStream](https://github.com/DMK041218/ObjectStream)：qwen2.5-vl-7b/evaluation/ovobench、streamingbench
  - [BBBBchan/DeltaStreamer](https://github.com/BBBBchan/DeltaStreamer)：configs/ovobench.json、streamingbench.json；README 写明 "The release contains the inference implementation, preparation scripts and scoring tools for OVO-Bench and StreamingBench"
  - [Haonan-Ge/selectstream](https://github.com/Haonan-Ge/selectstream)：main/cli/infer.py、eval.py
  - [zeyun-zhong/StreamTTT](https://github.com/zeyun-zhong/StreamTTT)：docs/EVALUATION.md
  - [UniX-AI-Lab/StreamOPD](https://github.com/UniX-AI-Lab/StreamOPD)：streamopd/eval/ovo_bench.py、streamingbench.py
  - [dragonlzm/MaLoW](https://github.com/dragonlzm/MaLoW)：docs/video_evaluation.md
  - [MCG-NJU/OneStreamer](https://github.com/MCG-NJU/OneStreamer)：Eval/Proactive_Eval/OneStreamer/scripts/run_ovobench.sh
- 未找到代码（2026-11-03 用 GitHub 仓库搜索 + arXiv HTML 链接核查）：WRWS、FOLIO、FlashBack、SPARE、FrameBlend 搜索结果为 0 或都是无关仓库；StreamScout、ReMem 的同名仓库都是无关项目（如 ncareau/StreamScout 2023 年的、omkar-kangutkar/REMem-Streaming 是一篇关于对话记忆的硕士论文）；[quhongyu/LatentStream](https://github.com/quhongyu/LatentStream) 只有一个 103 字节的 README —— [GitHub search API](https://api.github.com/search/repositories?q=LatentStream)

### Inferences
- 按"只比有代码的方法"这条规则，A 组可进主表的仍是 D-HSM、ShallowStream（按论文值引用）、SAVEMem、ObjectStream、DeltaStreamer（但骨干是 27B 智能体，不同）。加上新发现的 OASIS、FluxMem、MEMO、HERMES，主表在 Qwen2.5 和 Qwen3 上会比较充实。
- SelectStream 已被 NeurIPS 录用、有代码、骨干相同，适合作为"训练方法参照"放在表中（我们 Qwen3 高 +0.9，Qwen2.5 低 0.2）。

### Gaps
- 没有检查 OpenReview 上 ICLR 2027 / AAAI 2027 的匿名投稿。这些投稿匿名，无法确认作者对应关系；AAAI-27 的录用结果截至今天未查到。
- GitHub 搜索 API 未认证时有限速：StreamFlow、WeaveTime、StreamAgent 的搜索被限速打断，这三篇的代码情况未确认。

## 基础流式视频 LLM 与记忆方法（审稿人默认要看到的引用）

### Takeaway
基础工作都已录用，大多有代码。但**在我们的骨干上、按 OVO RT/BT + SB-RT 报数的只有 HERMES（三个骨干）、InfiniPot-V（Qwen2.5）、ReKV / LiveVLM（LLaVA-OV）**。其余要么需要训练，要么骨干是 Qwen2-VL / Qwen2 / Llama-3；它们的 OVO 数字（多由后续论文转引）明显低于 SimpleStream，适合放在表的"训练型流式模型"分组里作背景。

### Cited Findings
| 方法 | arXiv | 会议（来源） | 代码 | 骨干 | OVO RT / BT → (RT+BT)/2 | SB 实时 |
|---|---|---|---|---|---|---|
| VideoLLM-online | [2406.11816](https://arxiv.org/abs/2406.11816) | CVPR 2024（arXiv 注释） | [showlab/videollm-online](https://github.com/showlab/videollm-online)（demo/cli.py） | Llama-3-8B | 20.8 / 17.7 → 19.25（[MemEvo 主表](https://arxiv.org/html/2609.36581)转引） | 36.0 |
| Flash-VStream | [2406.08085](https://arxiv.org/abs/2406.08085) | ICCV 2025（[README](https://github.com/IVGSZ/Flash-VStream)） | [IVGSZ/Flash-VStream](https://github.com/IVGSZ/Flash-VStream) | Qwen2-VL-7B（ICCV 版，见 MemEvo 表 5） | 28.4 / 27.4 → 27.9（MemEvo）；[MEMO 表 1](https://arxiv.org/html/2609.38900) 给 RT 29.9（两处略有出入） | 23.2 |
| Dispider | [2501.03218](https://arxiv.org/abs/2501.03218) | CVPR 2025（[EGSD 表 1](https://arxiv.org/html/2609.36803)、MemEvo 表 5 标注；arXiv 注释为空） | [Mark12Ding/Dispider](https://github.com/Mark12Ding/Dispider)（scripts/evaluate_ovobench.py） | Qwen2-7B | 54.6 / 36.1 → 45.35（MemEvo） | 67.6 |
| ReKV | [2503.00540](https://arxiv.org/abs/2503.00540) | ICLR 2025（arXiv 注释） | [Becomebright/ReKV](https://github.com/Becomebright/ReKV) | LLaVA-OV-7B / 0.5B | 64.4 / 46.7 → 55.55（[TwinPast 首表](https://arxiv.org/html/2610.41983)）；MEMO 表 1 给 RT 57.3（不一致）；ShallowStream 在 Qwen3-8B 上重跑为 70.8 / 50.9 | 69.1（[FluxMem 表 1](https://arxiv.org/html/2603.02096)） |
| StreamBridge | [2505.05467](https://arxiv.org/abs/2505.05467) | NeurIPS 2025（arXiv 注释） | apple/ml-streambridge，已迁到 apple-aiml-research/ml-streambridge（GitHub API，最近 push 2026-09-11；内容未核对） | Qwen2-VL-7B、LLaVA-OV-7B、Oryx | Qwen2-VL + Stream-IT：71.30 / 68.05 → 69.68（[Foresight 表 3](https://arxiv.org/html/2610.03123)）；MemEvo 给 63.4 / 59.5（不一致） | 77.04（Foresight 表 2） |
| LiveVLM | [2505.15269](https://arxiv.org/abs/2505.15269) | DAC 2026（arXiv journal_ref） | [sjtu-zhao-lab/LiveVLM](https://github.com/sjtu-zhao-lab/LiveVLM)（文件树里只有 benchmarks/mlvu 的评测脚本） | LLaVA-OV-7B | 未报 OVO | 72.9（MEMO 表 1） |
| InfiniPot-V | [2506.15745](https://arxiv.org/abs/2506.15745) | NeurIPS 2025（arXiv 注释） | [aiha-lab/InfiniPot-V](https://github.com/aiha-lab/InfiniPot-V)（qwen_inference_ovu.py） | Qwen2.5-VL-7B | 65.9 / 47.6 → 56.75（MemEvo 转引"官方值"） | 76.4 |
| StreamForest | [2509.24871](https://arxiv.org/abs/2509.24871) | NeurIPS 2025 Spotlight（arXiv 注释） | [MCG-NJU/StreamForest](https://github.com/MCG-NJU/StreamForest)（模型、数据、代码已放出） | Qwen2-7B LLM（MemEvo 表 5） | 61.2 / 52.0 → 56.6（MemEvo） | 77.3 |
| TimeChat-Online | [2504.17343](https://arxiv.org/abs/2504.17343) | ACM MM 2025（[README](https://github.com/yaolinli/TimeChat-Online)） | [yaolinli/TimeChat-Online](https://github.com/yaolinli/TimeChat-Online) | Qwen2.5-VL-7B（训练） | 61.9 / 41.7 → 51.8（MemEvo） | 75.4（MemEvo）/ 75.3（MEMO） |
| StreamMem | [2508.15717](https://arxiv.org/abs/2508.15717) | 未见录用（arXiv 注释为 "15 pages"） | 未核实 | 未核实 | 摘要说评测了 "three long video … and two streaming video QA benchmarks"；StreamFlow 表中只列了 MLVU / VideoMME | — |
| StreamingVLM | [2510.09608](https://arxiv.org/abs/2510.09608) | ICLR 2026（arXiv 注释） | [mit-han-lab/streaming-vlm](https://github.com/mit-han-lab/streaming-vlm) | 未逐项核实（StreamClarify 将其作为 LLaVA-OV / InternVL / Qwen3-VL 上的基线） | 未抽取 | — |
| VideoChat-Online | [2501.00584](https://arxiv.org/abs/2501.00584) | CVPR 2025（arXiv 注释） | [MCG-NJU/VideoChat-Online](https://github.com/MCG-NJU/VideoChat-Online)（eval/evaluate_online_stream_single.py） | 未核实 | 只报自建的 OVBench | — |
| MMDuet | [2411.17991](https://arxiv.org/abs/2411.17991) | 未确认（arXiv 注释为 "9 pages"，README 中未找到会议名） | [yellow-binary-tree/MMDuet](https://github.com/yellow-binary-tree/MMDuet)（demo/app.py） | 未核实 | 未报 | — |
| HERMES | [2601.14724](https://arxiv.org/abs/2601.14724) | ACL 2026 Main（arXiv 注释 + 本地 README） | [haowei-freesky/HERMES](https://github.com/haowei-freesky/HERMES) | Qwen2.5-VL-7B、Qwen3-VL-8B、LLaVA-OV-7B | 见 A 级表 | 见 A 级表 |
| StreamChat | [2412.08646](https://arxiv.org/abs/2412.08646) | 未见录用 | 只有项目页（jihaonew.github.io） | — | 15 fps：50.14 / 47.38 → 48.76（[StreamScout 表 1](https://arxiv.org/html/2609.00291)） | 60.1 |
| LiveCC | [2504.16030](https://arxiv.org/abs/2504.16030) | CVPR 2025（arXiv 注释） | 未核实 | — | — | — |
| WeaveTime | [2602.22142](https://arxiv.org/abs/2602.22142) | CVPR 2026（arXiv 注释） | 未核实 | 训练（RELATED.md） | — | — |
| StreamAgent | [2508.01875](https://arxiv.org/abs/2508.01875) | 未见录用 | 未核实（搜索被限速） | 8B | 74.28 / 36.26 / 34.62 是 SB 的三个分组，不是 OVO（Foresight 表 2） | 74.28 |

- MemEvo 表 5 列出了基线的出处和会议：FluxMem、OASIS 为 CVPR 2026；HERMES 为 ACL 2026 Long；InfiniPot-V、StreamBridge 为 NeurIPS 2025；StreamForest 为 NeurIPS 2025 Spotlight；TimeChat-Online 为 ACM MM 2025；Flash-VStream 为 ICCV 2025；Dispider 为 CVPR 2025；VideoLLM-Online 为 CVPR 2024 —— [arXiv HTML 2609.36581](https://arxiv.org/html/2609.36581)

### Inferences
- 相关工作建议按四组写：
  1. 训练型流式视频 LLM：VideoLLM-online、Flash-VStream、Dispider、VideoChat-Online、MMDuet、StreamForest、TimeChat-Online、StreamingVLM、StreamBridge、LiveCC；
  2. 免训练 KV / 视觉记忆：ReKV、LiveVLM、InfiniPot-V、StreamMem、HERMES、FluxMem、StreamingTOM、MEMO、ShallowStream；
  3. 文本 / 事件记忆：D-HSM、FOLIO、StreamScout、OASIS、DeltaStreamer、EventMemAgent、OneStreamer、WTI；
  4. 近帧基线与何时取历史：SimpleStream、WRWS、FlashBack、SPARE、StreamClarify、ViSS。
- 表里"训练型流式模型"分组的数字会彼此冲突（StreamBridge 63.4 vs 71.3，ReKV RT 57.3 vs 64.4）。论文表注应写明"数字取自原文或某转引来源"，每格只取一个来源。

### Gaps
- StreamMem、MMDuet、StreamingBench 的会议，以及 StreamMem、LiveCC、WeaveTime、StreamAgent 的代码，都未确认。
- 大部分基础工作的 OVO 数字是从后续论文（MemEvo、MEMO、TwinPast、Foresight）转引的，不是从原文表格读的。

## 基准：必须引用哪些，审稿人可能要求加哪些

### Takeaway
OVO-Bench 和 StreamingBench 是这一方向的标准组合，几乎所有对手都用。审稿人最可能要求加的是：
- OVO Forward（我们的口径不含 Forward，要在正文说明理由）；
- StreamingBench 的 Omni-Source / Contextual 分组，或完整 SB；
- OVBench；
- 新出的 TwinPast（直接检验"有没有真正用到历史"）。
ODV-Bench、SVBench、RTV-Bench、StreamBench 只作为引用即可。

### Cited Findings
- OVO-Bench：CVPR 2025，代码 [JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench)（README："OVO-Bench is accepted by CVPR 2025"）。分 Real-Time、Backward、Forward 三组；人类 RT 93.20 / BT 92.33 / FW 92.90 —— [arXiv 2501.05510](https://arxiv.org/abs/2501.05510)；人类分数见 [ViSS 表 1](https://arxiv.org/html/2610.34926)
- StreamingBench：[2411.03628](https://arxiv.org/abs/2411.03628)，代码 [THUNLP-MT/StreamingBench](https://github.com/THUNLP-MT/StreamingBench)；arXiv 注释里没有会议。分组为 Real-Time Visual Understanding（10 个任务）、Omni-Source、Contextual（[Foresight 表 2/7](https://arxiv.org/html/2610.03123)）。FOLIO、Foresight 报的是完整基准，和只报实时分组的协议不同
- OVBench（VideoChat-Online 提出）：CVPR 2025 —— [arXiv 2501.00584](https://arxiv.org/abs/2501.00584)。StreamForest 也报了 OVBench 60.5 —— [arXiv 2509.24871](https://arxiv.org/abs/2509.24871)
- ODV-Bench：StreamForest 提出，自动驾驶场景的实时流式理解 —— [arXiv 2509.24871](https://arxiv.org/abs/2509.24871)。SAVEMem 代码里有 evaluation/odvbench —— [wuhang03/savemem](https://github.com/wuhang03/savemem)
- SVBench：ICLR 2025 Spotlight，时间维度的多轮对话 —— [arXiv 2502.10810](https://arxiv.org/abs/2502.10810)
- RTV-Bench：NeurIPS 2025 Datasets & Benchmarks —— [arXiv 2505.02064](https://arxiv.org/abs/2505.02064)
- OmniMMI：CVPR 2025，流式多模态交互 —— [arXiv 2503.22952](https://arxiv.org/abs/2503.22952)
- StreamBench：StreamScout 表 2 与 StreamingBench 并列报告（Qwen2.5 + StreamScout 49.7，低于基座 51.1）—— [arXiv HTML 2609.00291](https://arxiv.org/html/2609.00291)；StreamClarify 用 StreamBench 的 1,556 题做诊断 —— [arXiv 2611.01263](https://arxiv.org/abs/2611.01263)
- TwinPast（新，10-30）：350 对视频对，5 类历史任务，用 PairAcc 衡量，有代码 —— [arXiv 2610.41983](https://arxiv.org/abs/2610.41983)、[Yh1zzzzzzz/TwinPast_code](https://github.com/Yh1zzzzzzz/TwinPast_code)
- TRACE（技术报告）：条件感知的流式评测，记录历史处理与响应事件；1,240 条记录，517 个视频 —— [arXiv 2609.30670](https://arxiv.org/abs/2609.30670)、[om-ai-lab/trace-bench](https://github.com/om-ai-lab/trace-bench)
- 主动响应类：ProactiveVideoQA（PACE、ProV-Mem 使用）—— [arXiv 2610.28323](https://arxiv.org/abs/2610.28323)
- 同方向论文通常再加离线长视频基准：VideoMME、MLVU、LongVideoBench（[StreamFlow](https://arxiv.org/html/2608.10949)、[EGSD](https://arxiv.org/html/2609.36803)、[WTI 表 4](https://arxiv.org/html/2609.37035)）

### Inferences
- OVO-Bench 本身就分 RT / BT / FW，StreamingBench 本身就分 RT / Omni / Contextual，这两个最容易补齐。OVBench 按名字有过去 / 现在 / 未来的划分，但未核实。
- 不报 Forward 的理由可以引用 SimpleStream / ShallowStream / D-HSM 等主表都只用 RT + BT 的惯例（RELATED.md 已记录 ShallowStream、FOLIO 只报 RT / BT）。
- 如果加一个新基准，TwinPast 性价比最高：它直接回应"近帧就够、文本先验就能答"的质疑，规模小，有代码。

### Gaps
- 没有核实 OVBench、RTV-Bench、SVBench、StreamBench 的任务划分细节和评测成本，也没核实 StreamBench 出自哪篇论文。
- StreamingBench 是否已被会议录用，未查到。

## 文本 / 字幕记忆方法：写什么、谁写、多久写一次、是否看问题、评测时在线还是预计算、是否计入写入延迟（coordinator 追加）

### Takeaway
"同一个冻结 VLM 定时写字幕 + 近帧作答"已经是主流写法：StreamScout 16 秒一条、FOLIO 8 秒、OASIS 2 秒块、D-HSM 1 秒块。几乎都是**与问题无关地写入**。在 OVO 评测里，大多数实现要么按题逐题重放视频（DeltaStreamer、OASIS），要么把字幕**预计算并缓存**（OneStreamer、EventMemAgent、MaLoW，D-HSM 在 SB 上也是）；D-HSM 在 OVO 上甚至是**提问时才建记忆**。只有 FOLIO 明确报告了写入开销（每 8 秒一段，每段 5.88 秒，约 0.74 倍实时）。

### Cited Findings
| 方法 | (1) 写什么 / 谁写 / 间隔 | (2) 是否看问题 | (3) OVO / SB 评测中如何生成 | (4) 写入延迟 / 实时率 |
|---|---|---|---|---|
| **StreamScout**（免训练版，无代码） | 自由字幕，按窗口均匀取帧，由**骨干模型自己**写；**Δ = 16 秒**一条；另维护 QA 摘要 h_t 和有界视觉缓存（§3.2 式 (1)(3)、§4.1 Implementation Details）—— [arXiv HTML 2609.00291](https://arxiv.org/html/2609.00291) | 与问题无关（§3.2 "query-agnostic stream maintenance"）；但 QA 摘要记录了之前的问答 | 论文写 "As the stream unfolds"；无代码，无法核实是否预计算 | 图 1 报的是作答延迟和 token 数；**没有找到字幕写入开销** |
| **FOLIO**（无代码） | JSON 实体中心记录 + 事件记录，由**同一 VLM** 写（Qwen2.5-VL-7B / Qwen3-VL-8B，vLLM）；**固定 8 秒一段**，自适应取关键帧（平均 3.83 帧）（RELATED.md 11-02；附录 H："The writer VLM processes one fixed 8-second video chunk at a time"）—— [arXiv HTML 2607.13298](https://arxiv.org/html/2607.13298) | OVO 是单轮问答（§4.1），写入与问题无关；多轮的 SB 中，"previous turns can update the same focus state for later stream segments"（§2），即之前的问题会影响后续写入 | 论文说 "FOLIO writes memory online from the observed prefix"（§2）；附录 H 的统计来自 "saved memory-construction logs"；没写明是按视频预计算还是按题重放 | **报告了**：写入 7.5 次/分钟，每段 5.88 秒（Qwen3-8B，RTX 6000 Ada，§4.3 表 4）→ 写入实时率约 0.74（我们按 5.88/8 计算）；首 token 6.58 秒 |
| **D-HSM**（有代码） | 六字段结构化观察，由同一 VLM 写；OVO 上用 ≤20 个均匀采样的 1 秒历史块（dhsm/hub_and_spoke.py build_memory_from_chunks） | 提示词通用，但**计数题换成针对问题的提示词**（_caption_prompt_for）→ 部分依赖问题 | **OVO：提问时才建记忆**；SB：按视频块缓存，跨题复用（experiments/evaluate_streamingbench.py ensure_captions）（coordinator 本地核实）—— [Oshikaka/D-HSM](https://github.com/Oshikaka/D-HSM) | 论文：每片段写观察 1.55 秒、每题作答 0.74 秒（A100，RELATED.md 11-02） |
| **DeltaStreamer**（有代码） | 每个采样帧（基础 1 fps，静止时最低 0.25 fps）写结构化场景描述 D_t 和相对上一帧的变化 Δ_t，由同一 VLM（Qwen3.6-27B）写，输入为上一帧、当前帧和记忆文本（RELATED.md 10-24） | 提问前**看不到问题**：core/agent.py 约 L5137–5142 的注释写 "Before that the model must observe and accumulate memory without knowing the question"；问题在提问那一轮才钉进记忆（pin_fact "[Q] …"）。提问前的检索用最近观察拼成的隐式查询（agent.py L6159） | **按题、按流顺序逐轮重放**：core/video_processor.py get_next_request（L241–266）每轮一帧，到 question_time 才作答；另有 cached 模式，可从保存的轨迹恢复提问前的记忆（agent.py L2934 merge_cached_trajectory_rounds、约 L3690–3730 restore_prequery_memory）—— [BBBBchan/DeltaStreamer](https://github.com/BBBBchan/DeltaStreamer) | 代码按样本记录 elapsed_ms（agent.py L9208）；论文里的实时率未核实 |
| **OneStreamer**（训练，有代码） | Observe 记录（物体、动作、场景、状态变化）+ Summary 记录（RELATED.md），由**训练后的 OneStreamer 自己**写；评测配置为每秒 1 次更新，4 fps 取帧，上下文 128 帧（Eval/Proactive_Eval/OneStreamer/configs/ovobench.json："updates_per_second": 1, "fps": 4, "input_mode": "video_per_second"） | 与问题无关：prompts.py L80 "A causal textual memory produced while the video was playing, without access to the current question" | **预计算并缓存**：先把流字幕化到题目时间点，存到 captions/<model>/<sample_id>.json（src/onestreamer/cache.py L73），也可以按流共享时间线（cache.py L106–118，"shared_timelines"，以各题结束时间为边界），再用记忆 + 近帧作答 —— [MCG-NJU/OneStreamer](https://github.com/MCG-NJU/OneStreamer) | 未核实 |
| **OASIS**（有代码） | STM 窗口的事件摘要（文本，可附粗帧），由 MLLM 写（src/oasis/model.py L75 "event summarizer"，summarize_event L190，输入含 ASR 文本）；chunk_seconds = 2.0（src/oasis/config.py L25）；相邻摘要会合并（event/forest.py L17） | 摘要与问题无关；另有 QA 历史摘要，会把之前的问答并进去（event/forest.py L32–44） | **按视频、按流顺序在线处理**：src/scripts/eval.py L160–204 对每个视频先 reset，逐个数据包调用 process_stream_input，到断点时调用 process_query —— [Solus-sano/OASIS](https://github.com/Solus-sano/OASIS) | 论文报告了每题 token 消耗（图 4）和峰值显存（表 13）；写入延迟未核实 |
| **EventMemAgent**（训练，有代码） | 事件字幕，由**另一个冻结的 Qwen3-VL-4B** 写（README："The 4B model is used only to generate event captions during memory preprocessing"）；用直方图相似度切事件边界，每条字幕最多取 32 帧（memory/preprocess.py L92、L206–229），另写变化描述（L374） | 未核实 | **离线预处理**：OVO 评测直接读 HF 上已处理好的 memory parquet（README："no additional caption generation is needed"）；仓库另有在线构建器 memory/builder.py —— [lingcco/EventMemAgent](https://github.com/lingcco/EventMemAgent) | 未核实 |
| **MaLoW 文本摘要变体**（训练，有代码） | 带时间戳的结构化笔记（ENTITIES、EVENTS、STATE CHANGES、COUNTS、VISIBLE TEXT、UNCERTAINTIES），由与作答者共享的基座写，写时临时去掉生成的 LoRA（src/malow/video/text_summary.py 文档字符串与 CAPTION_PROMPT）；输出目录名为 "text-summary-30s" | 与问题无关（"Opt-in, question-independent visual memory"） | 顺序回放视频生成字幕，再压缩成摘要；文档提供了**预计算字幕与摘要**的下载 —— [docs/malow_w_text_summary_evaluation.md](https://github.com/dragonlzm/MaLoW) | 文档承认 "time-consuming"，没有实时率 |
| **WTI**（训练，无代码） | 自然语言记忆条目 + 源视频时间段，由策略自己写；"controller-triggered memory updates … compress elapsed observations into a bounded temporal index" —— [arXiv HTML 2609.37035](https://arxiv.org/html/2609.37035) | 在未来问题未知时写入（摘要） | 未说明 | 未说明 |
| **EGSD**（训练，无代码） | 记忆是可核验 Event（关键实体、动作、细节）的增量更新，由模型自己写 —— [arXiv 2609.36803](https://arxiv.org/abs/2609.36803) | "memory is written before the question arrives"（§1） | 未说明 | 未说明 |
| CAM（不在 OVO / SB 上评测） | 每 30 秒由 MLLM 写行为、对白、外观，再由 LLM 转成三元组（GPT-5.2）（RELATED.md 11-02）—— [arXiv 2609.06504](https://arxiv.org/abs/2609.06504) | 与问题无关 | 未说明 | 每题 47.3 秒（作答端） |
| 不是文本记忆（作区分） | SAVEMem：三级视觉记忆（[savemem.py](https://github.com/wuhang03/savemem)）；MEMO：全局 / 实体级特征作索引，外加高分辨率帧（[arXiv 2609.38900](https://arxiv.org/abs/2609.38900)，式 (10)）；ShallowStream、HERMES、ReKV：KV 记忆 | — | — | — |

### Inferences
- ReadPast 的写入设定（同一 VLM、16 秒一条、与问题无关）和 StreamScout 完全一样，和 FOLIO / D-HSM 是同一家族，**不能当作创新点**。论文里写入应作为"沿用"的设计，并引用 StreamScout 和 D-HSM。
- 审稿人很可能追问"字幕写入的开销是否计入实时"。对手里只有 FOLIO 报了写入实时率（约 0.74），D-HSM 报了每段 1.55 秒。我们最好也报一个"每 16 秒一条字幕的写入耗时 / 16 秒"的实时率，并写清评测时字幕是在线按流顺序生成、按视频缓存，还是提问时生成；同时注明 D-HSM 在 OVO 上是提问时建记忆，和"流式写入"并不一致。
- "预计算并缓存字幕"在这一方向很常见（OneStreamer、EventMemAgent、MaLoW、D-HSM 在 SB 上），只要写入与问题无关、只用题目时间点之前的帧，就和在线写入等价，可以在论文里如实说明。

### Gaps
- StreamScout、FOLIO、WTI、EGSD 没有代码，评测时是在线生成还是预计算无法核实。
- DeltaStreamer、OneStreamer、OASIS、EventMemAgent 论文中关于写入延迟的段落没读到；EventMemAgent 写入时是否用到问题类型（preprocess.py 头注释提到 "question_type"）未核实。
- OASIS 的 config 有三个 model_path 字段，写摘要的 MLLM 是否就是作答模型，未逐行确认。
