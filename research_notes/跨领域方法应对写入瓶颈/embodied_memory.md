# 具身/机器人记忆系统：开放词表、与问题无关的场景记忆（逐物体视觉证据 + 属性聚合 + 查询回答）

> 范围说明：本笔记只覆盖 embodied / egocentric 的"物体中心记忆"路线（ReMEmbR、OpenEQA 基线、ConceptGraphs、HOV-SG、OpenMask3D、3D-Mem、GraphPad、VideoAgent、Embodied VideoAgent、AMEGO、RELOCATE、Ledger、R4DSG、EgoMemory 等）。Embodied-RAG 和 HOV-SG 的细节没有核实到全文（见各节 Gaps）。各项写明年份/会议；最新一项是 2026-10 的 arXiv。
> 对照我们的现象：caption 写入 + 4 张近期帧 + 选择题；~44% 题"记忆中无答案信息"，~44%"正确与干扰词并存（歧义）"，换 writer、分辨率、粒度、prompt、实体图都不变；7B 在查询时回看会编造证据；整帧 CLIP 检索不准；喂整张历史帧会损害当前感知。

## Q1：这些系统存的是物体 crop / 逐物体 embedding 而不是文本吗？相对纯 caption 记忆，报告了多大的准确率差异？

### Takeaway
存储方式分三类：(a) 纯文本，即 caption 加时间和位置（ReMEmbR）；(b) 每个物体存特征加文本标签/描述（ConceptGraphs、VideoAgent、E-VideoAgent、GraphPad、AMEGO，存的是特征，原始 crop 通常不保留）；(c) 存少量精选整帧并配物体清单（3D-Mem）。公开对比里有一个很一致的规律：只把物体变成文本（scene-graph captions）不如逐帧 caption，逐帧 caption 又明显不如直接给 VLM 少量精选图像。在"物体在哪、被移动过几次"这类物体中心问题上，纯 caption 记忆（ReMEmbR）几乎等于 blind，物体中心的结构化记忆则带来约 +10～13 点。几乎所有大增益都出自强 API 模型（GPT-4V/4o、Gemini、GPT-5.4）。7B/9B 级别的增益较小，个别类别还会变差。

### Cited Findings
**纯 caption 记忆（基线 / 对照）**
- ReMEmbR（NVIDIA，arXiv 2024-09）：记忆是向量库，每段存 caption 的文本 embedding、位置和时间，不存图像。captioner 是 VILA1.5-13b，每 3 s 输入 6 帧；embedding 用 mxbai-embed-large-v1。检索由 LLM agent 调用文本、位置、时间三类检索函数，最多 3 轮。— [ReMEmbR arXiv 2409.13682](https://arxiv.org/html/2409.13682)
- ReMEmbR NaVQA descriptive 准确率（short/med/long）：ReMEmbR(GPT-4o) 0.62/0.58/0.65；"LLM with Caption"（整条 caption 全塞给 GPT-4o）0.57/0.66/0.55；Multi-Frame VLM 只能处理 short，得 0.55。也就是说，在 caption 记忆内部，用检索还是全读，差距不大，而且互有胜负。— [ReMEmbR](https://arxiv.org/html/2409.13682)
- ReMEmbR 消融（Overall correctness S/M/L）：完整版 0.72/0.56/0.61。caption 粒度从 3 s 改到 12 s（0.5 FPS）后降到 0.54/0.50/0.38。换 VILA1.5-3b 作 captioner 得 0.60/0.58/0.50。报告的失败案例：饮水机被描述成 "silver machine"，导致 "soda machine" 查询走错。重复 caption 堆积会稀释记忆。— [ReMEmbR](https://arxiv.org/html/2409.13682)
- 在 2026 年的 HD-EPIC 物体中心问答上，Ledger 作者重建了 ReMEmbR 记忆并用 GPT-5.4 作答，得 29.8，blind 是 29.7，等于没有增益。同表 AMEGO 34.7、OSNOM 30.2、DirectMe 30.7。— [Ledger arXiv 2610.10538](https://arxiv.org/html/2610.10538)
- AMEGO（ECCV 2024）的 Active Memories Benchmark（EPIC-KITCHENS，5 选 1，随机 20.0）：caption + LLaMA-2-7B 的 S-QA 只有 22.4–23.6，LLoVi 20.8–22.6，CLIP 帧检索 SF-QA 22.0，都接近随机。物体中心的 AMEGO-L 得 36.3（+12.7）。— [AMEGO arXiv 2409.10917](https://arxiv.org/html/2409.10917v1)

**OpenEQA（CVPR 2024）EM-EQA：文本物体图 < 帧 caption < 原始帧**
- EM-EQA（LLM-Match）：GPT-4 blind 33.5；GPT-4 + LLaVA-1.5 帧 caption 43.6；GPT-4 + ConceptGraphs 场景图 caption 36.5；GPT-4 + SVM 场景图 caption 38.9；GPT-4V（多帧图像）55.3；人类 86.8。— OpenEQA 原表数字转引自 [GraphPad arXiv 2506.01174](https://arxiv.org/html/2506.01174) 和 [Embodied VideoAgent arXiv 2501.00358](https://arxiv.org/html/2501.00358)，两处一致（原 PDF 抓取失败，见 Gaps）。
- 由此可算：把信息组织成"物体 + 文本"（CG）比逐帧 caption 低 7.1 点，比 blind 只高 3.0 点；直接给图像比帧 caption 高 11.7 点。— 同上来源
- OpenEQA 项目页的定性结论：有视觉输入的 agent 在物体/属性识别、定位和世界知识类上表现好，其他类别"接近 blind GPT-4"。— [OpenEQA project page](https://open-eqa.github.io/)

**3D-Mem（CVPR 2025）：精选少量整帧 + 物体清单**
- 存储内容：每个 snapshot 是一张整 RGB 帧（360×360），附带该视角内的物体集合（类别、置信度、3D 位置）。物体用 YOLOv8x-World（ScanNet 200 类）检测，跨视角合并沿用 ConceptGraphs 的做法。QA 时不存 crop，crop 只在 GOAT-Bench 物体导航中使用。— [3D-Mem arXiv 2411.17735](https://arxiv.org/html/2411.17735v2)
- A-EQA：CG 场景图 caption 34.4；LLaVA-1.5 帧 caption 38.1；Multi-Frame（75 帧）41.8；Explore-EQA 46.9；3D-Mem 52.6；人类 85.1。EM-EQA：Frame Captions 38.1；Multi-Frame 48.1（平均 3.0 帧）；3D-Mem 57.2（平均 3.1 帧）；人类 86.8。在几乎相同的帧数下，精选帧比均匀/普通多帧高 9.1 点。— [3D-Mem](https://arxiv.org/html/2411.17735v2)
- 每个 episode 约 39.76 次观测，生成 10.94 个 snapshot，预筛后保留 3.26 个（约 30%）。— [3D-Mem](https://arxiv.org/html/2411.17735v2)

**GraphPad（arXiv 2025-06）：物体节点（CLIP/BGE 特征 + caption + 关键帧）+ 查询时由 VLM 回看并改写记忆**
- 节点存物体 track 的点云、pooled CLIP 视觉 embedding、pooled BGE 文本 embedding、caption、楼层/房间 ID、可见关键帧。论文没有说存了 crop。— [GraphPad](https://arxiv.org/html/2506.01174)
- EM-EQA：GraphPad（Gemini 2.0 Flash，初始 5 帧，最多 20 次 API 调用）55.3，纯图像 Gemini 2.0 Flash（25 帧）52.3。分类对比（GraphPad vs 纯图像）：属性识别 66.8 vs 46.5（+20.3）；物体状态 69.6 vs 66.5；功能推理 59.2 vs 53.5；物体识别 58.4 vs 62.2（−3.8）；空间理解 47.7 vs 52.4（−4.7）；物体定位 31.3 vs 34.3（−3.0）。— [GraphPad](https://arxiv.org/html/2506.01174)
- 消融（184 题子集）：只有 Frame Memory 32.9；加场景图 34.6（+1.7）；加导航日志 42.5；场景图加导航日志 46.9；再加帧级回看 API 50.5。可见场景图本身只贡献 +1.7，大头来自查询时用强 VLM 回看并记笔记。— [GraphPad](https://arxiv.org/html/2506.01174)

**VideoAgent（ECCV 2024）/ Embodied VideoAgent（ICCV 2025）：caption 记忆 + 物体特征记忆**
- VideoAgent 消融（NExT-QA 600 题子集，平均准确率）：只有 caption 40.7；加带 re-ID 的物体记忆 56.0（+15.3）；物体记忆不做 re-ID 48.7。物体记忆的做法：RT-DETR + ByteTrack 跟踪，每个 track 随机取 10 帧 crop，对 CLIP 特征求平均，配合类别和出现时段存入 SQL。— [VideoAgent arXiv 2403.11481](https://arxiv.org/html/2403.11481)
- VideoAgent 在 Ego4D NLQ val 上 zero-shot R1@0.3 为 17.39、R1@0.5 为 7.47，有监督的 GroundNLQ 是 27.20/18.91。EgoSchema 500 题子集 62.8（LLoVi 51.8）。— [VideoAgent](https://arxiv.org/html/2403.11481)
- Embodied VideoAgent 每个物体存：ID、类别、STATE（open/close/in hand/normal）、关系、3D box、OBJ Feat（物体 crop 的 CLIP 特征）、CTX Feat（所在整帧的 CLIP 特征）。另外沿用 VideoAgent 的 2 s caption 记忆。没有明确说存原始 crop。— [Embodied VideoAgent](https://arxiv.org/html/2501.00358)
- E-VideoAgent 结果：Ego4D-VQ3D Succ 85.37%（图像查询），EgoLoc 80.49%。用文本查询的版本只有 53.05%。OpenEQA EM-EQA 1/5 子集：VideoAgent 36.3；E-VideoAgent（InternVL2-8B）41.2；E-VideoAgent（GPT-4o）47.0。没有"去掉物体记忆"的消融。依赖深度和位姿：GT 位姿 41.2，换成 DUSt3R 噪声位姿 40.0。— [Embodied VideoAgent](https://arxiv.org/html/2501.00358)

**Ledger（arXiv 2026-10，"Never Look Back"）：逐物体位置历史 + 每个静止段的文本描述，回答器只读文本**
- 每个物体存：label、带时间戳的观测（时间、图像 box、3D 位置）、"静止段"（时间区间、3D 位置、文本描述），另存佩戴者轨迹。摘要所示部分没有提到存 crop 或 CLIP/DINO 特征。回答器单次调用，只读文本，不看图。— [Ledger](https://arxiv.org/html/2610.10538)
- HD-EPIC（2,400 题）：GPT-5.4 blind 29.7，加 Ledger 42.6（+12.9）；Qwen3.5-9B blind 22.8，加 Ledger 33.6（+10.8）。分原型（GPT）：fixture location 13.8→56.0；object location 38.6→48.0；contents retrieval 16.5→33.5；movement counting 31.0→45.0。— [Ledger](https://arxiv.org/html/2610.10538)
- UCS-Bench（2,771 题，随机 25.7）：blind 33.8；48 帧 44.4；只用 Ledger 38.5；帧加 Ledger 46.3。在这个更"通用"的 egocentric 基准上，纯文本物体记忆不如直接看 48 帧，两者叠加只多 +1.9。— [Ledger](https://arxiv.org/html/2610.10538)
- Qwen3.5-9B 在 stationary localization 上用记忆反而更差：blind 26.0，有记忆 14.5。— [Ledger](https://arxiv.org/html/2610.10538)

**其它 2025–2026 物体中心 egocentric 记忆**
- R4DSG（ACM MM 2026，arXiv 2026-08）把视频转成按时间、地点、持久物体、相对锚点变化索引的记忆条目。在 EgoLifeQA 的 255 道物体相关题上，比 EgoRAG-Text（caption RAG）总体高 +6.7 点，"when" 类高 +12.5 点。摘要没有给绝对分数。— [R4DSG arXiv 2608.11017](https://arxiv.org/abs/2608.11017)
- EgoMemory（Findings of ACL 2026）是基于 Ego4D 的个人物品检索基准：245 个视频、45 名参与者、165,795 条用户专属物体标注、639 条人工校验的查询。配套方法 EgoRetriever 是 training-free 的（MLLM 加反思式 CoT）。具体数字没取到。— [EgoMemory ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.362/)

### Inferences
- 对我们最关键的对照是 OpenEQA 的 CG 36.5 < 帧 caption 43.6 < 图像 55.3，以及 3D-Mem 的 EM-EQA 57.2 vs Multi-Frame 48.1（同为约 3 帧）。这说明"实体化/结构化的文本"并不能补回 caption 丢掉的信息，与我们 D-HSM 实体图无效的观察一致。真正的增益来自保留像素，并在问答时只给少量正确的图。
- 纯 caption 记忆在物体中心问题上接近 blind：ReMEmbR 在 HD-EPIC 上 29.8 vs 29.7，AMEGO 的 caption 基线接近随机。这和我们 44% 的"无信息"桶同构。caption 写入器不知道未来问题，就不会写下"钥匙放在哪个台面"这种细节。
- 多数系统存的是平均后的特征，不是 crop 本身。特征适合检索，属性判断仍需看像素：3D-Mem 存整帧、GraphPad 回看关键帧、E-VideoAgent 的 vqa 工具看帧或 crop。若想在我们的设置里修复属性歧义，需要存小 crop 或 crop 坐标加降采样帧，不能只存特征。
- 以上增益几乎都靠强闭源模型。7–9B 的证据只有三条：Ledger 的 Qwen3.5-9B（+10.8，但个别类别 −11.5）、E-VideoAgent 的 InternVL2-8B（41.2，比 GPT-4o 低 5.8）、AMEGO（规则式回答，不靠 LLM）。对我们 7B 的可迁移性要打折扣。

### Gaps
- OpenEQA 原 PDF 抓取失败（403 / 超过 10MB），CG 基线和 GPT-4V 的分类别数字、GPT-4V 是否只在子集上评测，都没有核实。55.3/43.6/36.5 来自两篇后续论文的转引。
- 3D-Mem 表 2 的 "Frame Captions 38.1" 和 OpenEQA 原文的帧 caption 43.6 不一致，可能是子集或评审模型不同（另一来源指出不同论文的 LLM-Match 评审模型不同，分数不可直接比较，见 [Spatiotemporal KG arXiv 2510.01483 搜索摘要](https://arxiv.org/pdf/2510.01483)）。原因未核实。
- 没有找到任何工作在相同回答模型下做"存 crop vs 存 caption"的严格 A/B。E-VideoAgent 和 3D-Mem 都没有这个消融。
- ReMEmbR 只报告 caption 内部的消融，没有和存图像特征的版本对比。
- Embodied-RAG（2024）和 HOV-SG（RSS 2024）的存储与检索细节没取到全文。HOV-SG 只确认了"楼层/房间/物体三级开放词表特征，表示体积比稠密开放词表地图小 75%"。— [HOV-SG arXiv 2403.17846](https://arxiv.org/abs/2403.17846)

## Q2：它们如何处理同一物体跨时间的不一致属性观测（跟踪 + 多视角投票）？能否 training-free 地修复我们的"歧义"桶？

### Takeaway
这些系统的"多次观测聚合"主要作用在身份和位置上：几何加外观关联、特征滑动平均、位置变化的迟滞阈值。对文本属性做显式投票的只有 ConceptGraphs 一例：约 10 个视角各自出 LLaVA caption，再由 GPT-4 总结，并"以重复出现为准"。但它在 OpenEQA 上的分数反而低于帧 caption。没有任何论文报告属性一致性指标，也没有"多视角属性投票提升属性题准确率"的直接证据。对我们的歧义桶，文本投票只在"正确属性在多数 caption 中出现过"时有用。更可能有效的是：靠跟踪/re-ID 把同一实体的多张 crop 聚到一起，问答时让 VLM 在选项约束下看这几张 crop 做闭集判断。这一点没有直接证据，需要实测。

### Cited Findings
- ConceptGraphs（ICRA 2024；arXiv 2023-09）的物体特征融合：f_o = (n·f_o + f_new)/(n+1)，即 CLIP（或 DINO）特征的 running average。关联分数 = 语义余弦（缩放到 [0,1]）+ 几何重叠（邻近点比例），贪心归属，超过 δ_sim=1.1 才合并，否则新建物体。作者说 Hungarian 匹配更慢、增益"极小"。— [ConceptGraphs arXiv 2309.16650](https://arxiv.org/html/2309.16650)
- ConceptGraphs 的多视角文本属性"投票"：每个物体按 3D 点贡献数选最多 10 个视角，逐视角用 LLaVA-7B 生成 "describe the central object"，再由 GPT-4（gpt-4-0613）汇总。"被反复提到的视为可能准确，不重复的无关 caption 标为 invalid"，最终输出 summary、tags 和 object_tag。— [ConceptGraphs](https://arxiv.org/html/2309.16650)
- ConceptGraphs 节点标签精度平均 0.71（CG-D 为 0.61，各场景 0.65–0.78），作者把主要错误归因于 LLaVA，例如小物体被误认成牙刷、剪刀。每个场景有 0–5 个重复物体。— [ConceptGraphs](https://arxiv.org/html/2309.16650)
- OpenMask3D（NeurIPS 2023）：每个 3D 实例按可见度选 top-k 视角，每个视角取 L 个多尺度 crop，对 k·L 个 crop 的 CLIP 特征求平均。第三方综述称多尺度 crop 与 SAM 精修的组合最好（15.4 AP），top-k 的具体消融没有核实。— [OpenMask3D arXiv 2306.13631](https://arxiv.org/pdf/2306.13631)；[第三方综述](https://liner.com/review/openmask3d-openvocabulary-3d-instance-segmentation)
- VideoAgent 的 re-ID：相似度 = 0.15·CLIP + 0.85·DINOv2。一个 track 要与组内每个成员相似度 > 0.5、且至少与一个成员 > 0.62，才能并入该组。权重是在 EgoObjects 上网格搜索得到的，所以不算严格零调参。re-ID 带来 +7.3（56.0 vs 48.7）。— [VideoAgent](https://arxiv.org/html/2403.11481)
- Embodied VideoAgent：静态物体在 Spatial IoU > 0.2 时匹配，或者 MaxIoS > 0.2 且同类别时匹配。动态物体要求体积相似度 > 0.7 且视觉相似度 > 0.45。匹配后做滑动平均，窗口在静态物体上取 10，动态物体取 2，即对会变的物体刻意缩短平滑窗口。状态由 VLM 看画了 3D box 的帧来判断和更新。— [Embodied VideoAgent](https://arxiv.org/html/2501.00358)
- Ledger 的"重复证据"规则：要连续 k=3 次观测都离当前静止位置超过 τ=0.3 m，才开新的静止段；物体返回原位的观测仍记在当前段。描述按段存，每个物体最多描述 6 段，用 Qwen3.5-9B 生成。跟踪碎片用 SAM 3 视频跟踪器合并（mask IoU 0.3，3D 距离超过 1.0 m 则否决合并）。代价有两点：短暂移动会被抑制而漏记；关联把一个物体拆成多条 track，或漏掉物体，导致计数题不涨。— [Ledger](https://arxiv.org/html/2610.10538)
- Ledger 消融（2,317 题）：去掉描述，总体 −4.9（object location −8.4）；去掉持久性/迟滞，总体 −5.5（movement counting −15.5）。规则阈值是在单独的校准视频上调的。— [Ledger](https://arxiv.org/html/2610.10538)
- AMEGO 的 track 稳定化：30 帧窗口内有 20 次一致检测才开始一条 tracklet，手还在视野里但连续 20 帧没有匹配检测就结束；跨时间 re-ID 用 DINOv2 平均特征的余弦相似度。去掉 tracker 或基于手的终止规则后，片段数比 GT 多出 3,000 多个。— [AMEGO](https://arxiv.org/html/2409.10917v1)
- 错误会固化：GraphPad 指出检测错误没有校验步骤，会一直留在图里误导后续推理；浅层回看（m=1/2）有时还不如不回看（45.8/46.3 vs 46.9）。3D-Mem 也报告检测器误标或漏检会误导 VLM，且 360×360 分辨率下小物体难以识别。— [GraphPad](https://arxiv.org/html/2506.01174)；[3D-Mem](https://arxiv.org/html/2411.17735v2)

### Inferences
- 这些"投票"的本质，是用几何和外观把多次观测确定为同一实体，再对特征做平均或对位置做迟滞。属性层面只有 CG 的"LLM 汇总多条 caption"。按我们的数据推测，这个机制修不了"灰衬衫/白衬衫/答案是蓝衬衫"这类题：若所有 caption 都没写 "blue"，文本多数票只会把错误巩固下来；若 "blue" 只出现在少数 caption 里，多数票还会把它当成噪声丢掉。
- 歧义桶里可能有两种子情形，应分开统计：(i) 记忆里出现过正确属性，但与其他描述冲突，可用跨观测的一致性或计数来修复；(ii) 正确属性从未写入，只是名词（shirt）重叠被 word-overlap 判成歧义，本质上属于"无信息"，只能靠回到像素解决。建议在 OVO-Bench 上把歧义桶按"正确选项的区分性属性词是否出现在任一 caption 中"再细分一次，零成本就能得到可修复的上限。
- 颜色在不同光照和视角下本来就会漂移（egocentric 视频尤其明显）。逐视角让 7B 自由描述颜色，等于对一个本就带噪的信号多次采样，真正需要的是闭集判定。更合理的 training-free 版本是：同一实体的 K 张 crop 加上选项中的候选属性，让 VLM 逐张做闭集选择，再取多数或平均对数概率。这是"有选项约束的感知"，比"自由回看"更不容易编造，但没有任何论文在 7B 上验证过。
- 跨 16 s 块的人物 re-ID 是最大的不确定因素。上面的 re-ID 都针对物体（DINOv2 + 几何/3D），对人只能靠外观，而 egocentric 视频里人常被截断、背光、只露部分身体。DINOv2 余弦阈值（0.5/0.62 这一类）要在我们的数据上重新看分布，否则错并会把"蓝衬衫的人"和"灰衬衫的人"混在一起，反而制造歧义。

### Gaps
- 没有找到报告"属性一致性"或"多视角属性投票对属性类 QA 的消融"的论文。ConceptGraphs 只给了标签精度，没有属性准确率。
- 没有找到专门针对人物（而非物体）的 embodied 记忆跨时间 re-ID 与属性聚合方法及数字。
- OpenMask3D 的 top-k 视角数和多尺度 crop 消融的具体数字没有核实。

## Q3：它们如何为查询检索证据（crop 上的开放词表特征相似度 vs 文本）？相比整帧 CLIP 检索精度如何？

### Takeaway
检索有四种做法：(1) 文本查询与物体 crop 平均特征做 CLIP 相似度（ConceptGraphs、E-VideoAgent、VideoAgent 的候选 ID）；(2) 让 LLM/VLM 读物体清单或 caption 来挑选（ConceptGraphs 的 LLM 检索、3D-Mem 的类别预筛）；(3) 几何/时间加词重叠的规则检索（Ledger）；(4) 图像查询与区域特征匹配（AMEGO、RELOCATE）。已有证据显示：CLIP 对描述性查询尚可，对功能性和否定查询很差，否定查询的 R@1 可以低到 0.00–0.26；LLM 读文本在这些情况下好得多。在"图像查询"场景，区域级特征明显优于先前方法（RELOCATE stAP 相对 +49%）。但在文本查询下，"crop 级 CLIP vs 整帧 CLIP"在 egocentric 视频上的直接对比，我没有找到。对我们最可借鉴的是 3D-Mem 的思路：先用 VLM 从物体类别清单里挑出与问题相关的类别（文本），再取含这些物体的少量帧。它不依赖 CLIP 对整帧排序。

### Cited Findings
- ConceptGraphs 检索 R@1（Replica）：描述性查询 CLIP 0.59 vs LLM 0.61（但 CLIP 的 R@2/R@3 更高：0.82/0.86 vs 0.64/0.64）；功能/可供性查询 0.43 vs 0.57；否定查询 0.26 vs 0.80。实验室场景：描述性查询两者都是 1.00，可供性 0.40 vs 1.00，否定 0.00 vs 1.00。作者原话："CLIP excels with descriptive queries but struggles with complex affordance and negation queries"。— [ConceptGraphs](https://arxiv.org/html/2309.16650)
- 3D-Mem 预筛：把问题和场景内所有物体类别交给 VLM，返回按重要性排序的相关类别，取 top K=10，丢弃不含这些类别的 snapshot。最终 A-EQA 平均保留 3.26 个 snapshot（总数的 29.8%），GOAT-Bench 保留 4.66 个（28.1%）。K 变小时性能下降不多。— [3D-Mem](https://arxiv.org/html/2411.17735v2)
- Embodied VideoAgent 的 query_db 用 SQL 加 CLIP 文本-图像相似度（同时比 OBJ Feat 和 CTX Feat），返回 top-10 物体条目；temporal_loc 按文本-图像相似度返回 top-5 帧。用文本查询定位物体（VQ3D 文本版）Succ 只有 53.05%，用图像查询是 85.37%。— [Embodied VideoAgent](https://arxiv.org/html/2501.00358)
- VideoAgent 的片段定位联合使用 ViCLIP 视频特征和 caption 文本相似度，返回 top-5 片段；物体查询先用 CLIP 找候选 ID，再由 LLM 写 SQL。— [VideoAgent](https://arxiv.org/html/2403.11481)
- Ledger 完全不用 embedding 检索：HD-EPIC 的 box 查询先提升到 3D，取 ±6 s 内最近的 4 个实例；UCS-Bench 从问题中推断目标物体，按词重叠和时间邻近排序。同一份 Ledger 记录换成 ReMEmbR 的 agent 式检索阅读，只得 34.8，单次全读的读法是 42.6。— [Ledger](https://arxiv.org/html/2610.10538)
- AMEGO 的查询本身是物体或地点的图像 crop（每题 5 个候选 crop），用 DINOv2 特征做最近邻匹配到已存实例，再按题型规则作答；CLIP 帧检索基线 SF-QA 只有 22.0，接近随机 20.0。— [AMEGO](https://arxiv.org/html/2409.10917v1)
- RELOCATE（CVPR 2025）：training-free，对视频做 SAM 区域分割，在每个区域上池化 DINOv2 特征，先检索候选，再精修候选表示并过滤误检。视频编码与查询无关，可被多个查询复用。在 Ego4D VQ2D 上，stAP 相对先前 SOTA 提升 49%，tAP 提升 33%。— [RELOCATE arXiv 2412.01826](https://arxiv.org/abs/2412.01826v1)

### Inferences
- 我们是选择题，选项本身就提供了"查询词"。按上述证据，"选项文本 → crop CLIP 相似度"适合"蓝衬衫的人 / 红杯子"这类描述性检索，但会受 CLIP 属性绑定能力的限制。CLIP 对颜色-物体绑定较弱是社区共识，但本次没有找到量化来源。对于"做 X 时和谁说话"这类需要事件或否定语义的问题，CLIP 检索预计会很差，应先由文本记忆（caption 时间线）定位时间窗，再在该窗口的 crop 中做选项判别。也就是"文本定时间，crop 定属性"的两级检索。
- 3D-Mem 的"VLM 选类别 → 取含该类别的帧/crop"本质上是符号级检索，精度取决于写入时检测器的类别覆盖。它绕开了整帧 CLIP 排序不准的问题，与我们"整帧 CLIP 检索不准"的观察相符：问题在于整帧特征被背景主导，而不在于 CLIP 本身。
- E-VideoAgent 的文本查询定位 53% vs 图像查询 85%，提示"文本 → 视觉"跨模态检索本身就是瓶颈。即使存了 crop，用选项文本去检索 crop 的召回也可能只有中等水平。

### Gaps
- 没有找到在 egocentric 长视频上、以文本为查询、直接对比"物体 crop 级 CLIP/SigLIP 检索"与"整帧 CLIP 检索"精度的论文数字。
- 没有找到 CLIP/SigLIP 在小 crop 上做颜色/服饰属性检索的量化精度来源（只有 ConceptGraphs 的查询类型对比）。
- EgoMemory/EgoRetriever 的检索机制和数字没有取到全文。

## Q4：在我们设置下可行的 training-free 改造（写入时开放词表检测/跟踪、每个实体存小 crop、问答时只展示与选项相关的 crop），成本与诚实的风险

### Takeaway
最有依据的可迁移设计有三条：(a) 写入时在每个 16 s 块内用开放词表检测（YOLO-World / Grounding-DINO 一类）加视频分割跟踪（SAM2/SAM3），每条 track 留 2–4 张小 crop 和平均 DINOv2/CLIP 特征，跨块用 DINOv2 余弦做 re-ID（对应 VideoAgent / E-VideoAgent / AMEGO / Ledger）；(b) 问答时"文本定时间、选项定实体"，只给 VLM 少量（约 3 张）与选项相关的 crop 或精选帧（对应 3D-Mem 约 3 帧即 +9 点）；(c) 对同一实体的多张 crop 做"选项约束的闭集投票"，代替自由文本描述。风险也很实在：已发表的增益大多来自强 API 模型，并依赖深度/位姿和静态场景；检测词表同样与问题无关，"无信息"桶只能部分缓解；事件和交互类问题，物体记忆本身就弱。建议先做一个"查询时检测上限"实验：在问答时对存档帧按选项名词检测、裁剪、再展示，以此判断 7B 能否用好 crop。上限都不涨，就不要投入写入时系统的工程。

### Cited Findings
- 写入时开放词表管线的现成范例：Ledger 先用 VLM（Qwen3.5-9B）每 4 s 一帧给出可见物体名称，同义词合并后得到不超过 80 个标签的词表，再交给 YOLO-World v8x-worldv2 检测（1408 px，置信度 0.05，top-40），最后用 SAM 3 合并跟踪碎片。— [Ledger](https://arxiv.org/html/2610.10538)
- E-VideoAgent 的管线：YOLO-World 检测，SAM-2 生成 mask，借助深度和位姿把 2D 提升到 3D；动作描述用 LaViLa 每 2 s 生成一条。— [Embodied VideoAgent](https://arxiv.org/html/2501.00358)
- 2D-only 的选择：VideoAgent 用 RT-DETR + ByteTrack 加 CLIP/DINOv2 re-ID，完全不需要深度；AMEGO 用手-物交互检测器加 EgoSTARK 单目标跟踪器和 DINOv2，也是 2D 且 training-free。— [VideoAgent](https://arxiv.org/html/2403.11481)；[AMEGO](https://arxiv.org/html/2409.10917v1)
- 少量精选图像优于多帧：3D-Mem 在 EM-EQA 上用平均 3.1 帧得 57.2，Multi-Frame 用 3.0 帧得 48.1。GraphPad 初始帧数从 2 增到 6，得分是 48.2/47.4/49.9/50.3/50.5，增加帧数收益很小。— [3D-Mem](https://arxiv.org/html/2411.17735v2)；[GraphPad](https://arxiv.org/html/2506.01174)
- 小模型的证据：Ledger 用 Qwen3.5-9B 作答时 22.8→33.6，但 stationary localization 26.0→14.5；E-VideoAgent 换成 InternVL2-8B 后在 EM-EQA 子集上得 41.2，GPT-4o 是 47.0。— [Ledger](https://arxiv.org/html/2610.10538)；[Embodied VideoAgent](https://arxiv.org/html/2501.00358)
- 把检测结果写成文本也是一条已发表的 training-free 路线：Video-RAG（NeurIPS 2025）用开源工具提取 OCR、ASR 和目标检测信息，以文本形式输入 LVLM，自称 training-free、即插即用，在 Video-MME、MLVU、LongVideoBench 上"显著且一致"地提升。摘要没有给出 7B 的具体数字，也没有检测分量的消融。— [Video-RAG arXiv 2411.13093](https://arxiv.org/abs/2411.13093)
- 失败风险的出处：
  - 交互/事件类问题缺少事件证据，AMEGO 的手-物 tracklet 在交互计数上更好；Ledger 在物体最密集的长时录像上无法完成构建；场景切换会同时破坏检索和构建；单次运行每个原型的结果波动约 ±7–10 点。— [Ledger](https://arxiv.org/html/2610.10538)
  - 检测错误固化；每次 API 调用约 2–3 s；只在静态场景验证过。— [GraphPad](https://arxiv.org/html/2506.01174)
  - VLM 难以跨视角匹配同一物体的图像；360×360 下小物体识别困难。— [3D-Mem](https://arxiv.org/html/2411.17735v2)
- 辅助场景的早期先例：GO-Finder（ACM IUI 2021）自动检测手持物体并分组，生成"物体视觉时间线"，用户通过浏览 crop 找到物体最后一次出现的位置。用户研究称搜索准确率提高、心理负担降低，但页面没有给出具体数字。— [GO-Finder arXiv 2101.07314](https://arxiv.org/abs/2101.07314)

### Inferences
**方案 A（零工程诊断，先做）：查询时检测上限。** 问答时取选项里的名词短语（如 "blue shirt person" → person；"keys"），在存档的历史帧（如 1 fps 降采样）上跑开放词表检测器，裁出 top-N crop。按时间窗过滤，时间窗由 caption 时间线定位。然后给 VLM 每个选项 1–2 张 crop 加时间戳，以及最近的 4 帧，做闭集选择。这一步违反了"写入时不知道问题"，所以只作上限：若 7B 在歧义桶上看到正确 crop 也不涨，说明瓶颈在 7B 的细粒度属性感知或证据整合，写入时 crop 存储就不值得做；若涨，再做方案 B 把检测前移到写入时。额外价值：拿 crop 而不是整帧去测，可以直接检验"整帧伤害当前感知"是否由 token 量和背景干扰造成。

**方案 B（真正与问题无关的写入）：**
- 每个 16 s 块：先用 VLM 或 caption 本身抽取可见实体名词作为检测词表（Ledger 的做法），再加一个固定的通用类集合（person、hand-held object、container、furniture 等）。用 YOLO-World / Grounding-DINO 检测，SAM2 在块内传播跟踪。每条 track 按检测置信度×面积×清晰度选 2–4 张 crop，以约 112–224 px 的 JPEG 存储，同时存 DINOv2 平均特征（用于 re-ID）和 SigLIP/CLIP 平均特征（用于文本检索）。
- 跨块 re-ID：DINOv2 余弦加时间邻近；人物单独设阈值。阈值不要照搬 VideoAgent 的 0.5/0.62，先在我们的数据上看正负对的分布。
- 问答时：caption 时间线和问题先定位时间窗；选项文本对窗口内的实体 crop 做 SigLIP 打分；每个选项展示 top-1/2 crop（标注时间），总数控制在约 6 张以内。对同一实体的 K 张 crop 做"选项闭集投票"，取多数或 log-prob 平均，以此处理颜色等跨观测不一致的属性。
- 成本估计（推断，未测）：检测和跟踪在 1–2 fps 下对每块增加的 GPU 时间可能与写一条 7B caption 同量级；存储方面，每条 track 只有几张小 JPEG 加两个特征向量，远小于存整帧。按 Qwen2.5-VL 每 28×28 px 对应 1 个视觉 token 推算，一张 112² crop 约 16 token，224² 约 64 token，6 张 crop 远少于 4 张整帧，因此对当前感知的干扰预期较小。这是基于架构的推断，需实测。

**方案 C（最便宜的变体）：** 不存 crop，只把检测和跟踪结果按 Video-RAG 的做法写成结构化文本附在 caption 后（如 "person#3: blue shirt (3/4 views), gray (1/4)"）。属性分布由写入时对 crop 做的闭集或自由 VLM 判定统计得到。它能修复"正确属性出现过但被冲突描述淹没"的子桶，修复不了"从未写入"的子桶。按 OpenEQA 中 CG 文本低于帧 caption 的结果，纯文本版本的上限有限。

**诚实的风险：**
1. 写入时的检测词表同样与问题无关，"44% 无信息"桶只有在相关实体恰好被检测并保存时才能补救。动作、事件、对话类问题（"切菜时和谁说话"）不是物体属性问题，物体记忆帮助有限（Ledger 和 AMEGO 都报告事件证据不足）。
2. 所有大增益都来自 GPT-4V/4o、Gemini、GPT-5.4。在 7–9B 上增益缩小，个别类别还会变差；我们已经观察到 7B 回看会编造，crop 的闭集判定能否避免这个问题完全未知。
3. 大多数系统依赖深度、位姿和静态室内场景（ConceptGraphs、3D-Mem、E-VideoAgent、Ledger）。OVO-Bench 这类视频没有这些信号，只能走 2D 版本（VideoAgent、AMEGO），而 2D re-ID 更脆弱。
4. re-ID 错并会人为制造新的歧义；检测错误会固化（GraphPad）。
5. 属性颜色漂移可能是真实的光照效应，"多数票"未必等于标注者看到的颜色。
6. 工程量与延迟：检测、分割跟踪加 re-ID 三套模型常驻 GPU，与我们只用 GPU 2,3 的资源约束有冲突，需要评估。
7. 这些系统的阈值或权重往往在别的数据上调过（VideoAgent 在 EgoObjects 上网格搜索，Ledger 用校准视频），严格的"零调参"并不成立。

### Gaps
- 没有找到在 OVO-Bench、Ego4D NLQ 等流式 egocentric 选择题上，用写入时 crop 存储、配合 7B 回答模型的已发表结果。
- 检测器/跟踪器的实际吞吐（在我们的 GPU 上按每 16 s 块计）没有可引用的数字。上面的成本只是推断。
- 没有核实 Video-RAG 中检测分量的单独贡献和 7B 上的具体增益。
- LightMem-Ego（浙大，面向 AI 眼镜的分层流式记忆）只在新闻聚合页看到，日期和结果都无法核实，未纳入正文。
