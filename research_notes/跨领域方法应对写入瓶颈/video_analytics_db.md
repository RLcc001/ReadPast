# 视频分析数据库/系统：查询未知时的 ingest 索引 + query-time 验证（及其向 StreamKV 写入瓶颈的 training-free 迁移评估）

> 范围：video analytics / VDBMS 领域（OSDI/NSDI/SIGMOD/VLDB/ICDE 等），重点是"ingest 时不知道查询"和"ingest 模型便宜、会出错"这两个问题。2026-11-02 检索。
> 背景（来自任务描述，不是外部来源）：我们的 7B writer 每 16 s 写一条通用 caption；OVO-Bench EPM/ASI 上，按词重叠代理统计：约 44% 的题在记忆里没有答案信息（bucket a），约 44% 同时含正确项与干扰项的词（bucket b），约 6% 干净。换 writer、换分辨率、换粒度、换 prompt、换 D-HSM，这三个比例都不变。读取侧没有提升空间；用 7B 在 query 时重看帧会编造证据；CLIP 检索不准；把历史帧放进上下文会损害对当前画面的感知，放文本则不会。
> 每条 Cited Finding 都附来源。未在本次会话中核实的背景知识，一律放进 Inferences 或 Gaps 并标明。

## Q1：这些系统在 ingest 时怎样建 query-agnostic 索引？相对 top-1 标签能拿回多少 recall？

### Takeaway
该领域在 ingest 时保住 recall 的办法有四类：(1) 存 cheap model 的 top-K 候选而非 top-1（Focus）；(2) 存与模型无关的低层结构，如 blob、轨迹、关键点（Boggart）；(3) 存 embedding，再加少量代表样本（TASTI、LOVO、Panorama）；(4) 存不确定分布而非点估计（Everest）。它们给出的硬数字说明，便宜的通用模型 top-1 与"真值模型"差距很大：Focus 中通用 cheap CNN 要 K 达到 60–200（1000 类）才有 90% recall。Boggart 显示，预处理模型与查询模型一旦不一致，语义索引最多损失 94% 的精度。代价是这些办法几乎都依赖闭集标签空间、静态摄像头或训练过的 proxy。

### Cited Findings
**Focus（OSDI 2018；Hsieh et al.）：top-K 索引 + 物体级聚类**
- ingest 时，每个物体按 cheap CNN 输出的 top-K 类建索引，不只用 top-1。依据是经验观察："GT-CNN 的 top-1 往往落在 cheap CNN 的 top-K 之内，K 相对 1000 类很小" — [Focus paper (arXiv 1801.03493)](https://arxiv.org/pdf/1801.03493)
- 具体 recall 数字（lausanne 视频）：三个 cheap CNN（ResNet18 输入 224；ResNet18 去掉 3 层、输入 112；去掉 5 层、输入 56）达到 90% recall 分别需要 K≥60、K≥100、K≥200。越便宜的模型，同一 K 下 recall 越低 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- 代价：每个物体挂 K 个类，所以索引的平均 precision 只有 1/K。precision 要靠 query 时调用昂贵的 GT-CNN 来恢复 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- 针对单条视频流重训的 specialized cheap CNN（类别更少，并设一个 "OTHER" 类）只需 K=2 或 4 就能达到目标 recall，通用 cheap CNN 则要 K=60–200。specialized 模型还比通用 cheap CNN 便宜 10 倍，但需要按流周期性重训 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- ingest 时用 cheap CNN 倒数第二层特征做单遍聚类（阈值 T），按物体聚类而不是按整帧，索引里只存聚类中心。T 同时影响 precision 和 recall：中心被判成别的类时，簇里的真目标就漏掉了 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- 头条结果（示例交通视频）：Focus-Balance 的 ingest GPU 开销比 "Ingest-all"（全帧跑 ResNet152）低 86×，查询延迟比 "Query-all" 快 56×，precision 和 recall 均 ≥95%。作为对比，用 ResNet152 处理一个月的交通视频要 280 GPU 小时，约 $250 — [Focus paper](https://arxiv.org/pdf/1801.03493)

**Boggart（Agarwal & Netravali；arXiv 2106.15315，v2 2022-05）：模型无关的低层索引**
- 动机实验：用 A 模型做预处理、用 B 模型做查询（两者只差架构，或只差权重），中位精度下降为：二分类 0–32%，计数 8–84%，检测 46–94%。分开看，只差架构时为 0–84%，只差权重时为 2–94%，两者都差时为 1–90% — [Boggart paper](https://arxiv.org/pdf/2106.15315)
- ingest 只用传统 CV，全在 CPU 上跑：先做背景估计，提取运动区域（blob），再用 SIFT 等关键点跟踪把 blob 连成轨迹。原话是索引"imprecise, but fundamentally comprehensive across different CNNs/queries"；建索引比基于压缩 CNN 的方案快 58% — [Boggart paper](https://arxiv.org/pdf/2106.15315)
- **限制**：预处理目前只支持静态摄像头，移动摄像头留作 future work — [Boggart paper](https://arxiv.org/pdf/2106.15315)
- 存储：平均每小时视频产生 306 MB 索引（同一视频 H.264 编码约 1 GB/h），其中 98% 是关键点 — [Boggart paper](https://arxiv.org/pdf/2106.15315)

**TASTI（SIGMOD 2022；Kang et al.）：embedding 语义索引**
- 为每条记录生成 embedding，使 embedding 相近的记录在 target labeler 下输出也相近；不需要为每个查询训练 proxy。流程：预训练 embedding → 用 FPF（furthest point first）挑训练样本 → 用 triplet loss 微调 → 再用 FPF 选 cluster representatives → 只对代表样本调用 target DNN，其余记录按距离从 k 个最近代表传播结果 — [TASTI paper](https://arxiv.org/pdf/2009.04540)
- 可以 training-free（"TASTI-PT"，直接用预训练 embedding），但作者说明 "TASTI-T generally outperforms TASTI-PT"。lesion study 显示 triplet 训练对聚合查询帮助明显，FPF 聚类对 limit（稀有事件）查询至关重要 — [TASTI paper](https://arxiv.org/pdf/2009.04540)
- 头条：索引构建比现有 proxy 方法的标注开销便宜约 10×，查询最多加速 24× — [TASTI abstract](https://arxiv.org/abs/2009.04540)

**LOVO（arXiv 2507.14301，2025-07；镜像站给的 DOI 指向 ICDE 2025，未核实）：物体级 patch embedding 索引**
- 一次性用预训练视觉编码器（ViT-B/32 patch encoder，定位头仿照 OWL-ViT）在关键帧上抽取 patch 级 embedding，每个 patch 配一个 box。数据存入 Milvus，使用 PQ + inverted multi-index，作者称 "agnostic to user queries" — [LOVO HTML](https://arxiv.org/html/2507.14301)；[LOVO search abstract](https://arxiv.org/abs/2507.14301v1)
- 关键帧按运动向量变化选取；消融中存储从约 7976 MB 降到 2453 MB（抽取文本有乱码，数据集规模不明，需核对原文）— [LOVO HTML](https://arxiv.org/html/2507.14301)
- 抽取文本没有说明 box/MLP head 和 rerank transformer 是否训练过，training-free 与否不明 — [LOVO HTML](https://arxiv.org/html/2507.14301)

**Everest（Lai et al.；arXiv 2003.00773）：不确定索引**
- Phase 1：训练一个小型 CMDN（convolutional mixture density network），给每帧输出分数**分布**而非点值，得到一张"不确定关系表"。训练数据由 oracle 标注的少量样本构成，训练只需几分钟，并按 holdout NLL 选模型 — [Everest paper](https://arxiv.org/pdf/2003.00773)
- 采用 possible-world semantics：Top-K 答案的置信度等于它成为 Top-K 的所有可能世界的概率之和；用户为每个查询设阈值 thres，要求 Pr(R̂=R) ≥ thres — [Everest paper](https://arxiv.org/pdf/2003.00773)
- 头条：比 baseline 高 14.3×–20.6× 效率 — [Everest abstract](https://arxiv.org/abs/2003.00773)

**Panorama（PVLDB 13(4)，2020；Zhang & Kumar）：unbounded vocabulary**
- 把"查询目标词表无界"定为瓶颈，用多任务 PanoramaNet 抽 embedding，配合 content-based image retrieval，不重训就能处理新目标。in-vocabulary 查询效率比 baseline 高 2–20×，并能泛化到 out-of-vocabulary 查询 — [VLDB page](https://mail.vldb.org/pvldb/volumes/13/paper/Panorama%3A%20A%20Data%20System%20for%20Unbounded%20Vocabulary%20Querying%20over%20Video)；[survey arXiv 2105.14195](https://arxiv.org/pdf/2105.14195)
- 综述的对比说法：NoScope 只能对给定物体类型做 yes/no 判断，Panorama 的词表开放 — [survey arXiv 2105.14195](https://arxiv.org/pdf/2105.14195)

**AVA（NSDI 2026；Yan et al.，MSR + ZJU）：VLM 时代的 query-agnostic 事件知识图谱**
- ingest：Qwen2.5-VL-7B 每 3 s 一块写描述；相邻块 BERTScore > 0.65 就合并，合并后由 VLM 再总结。每个事件抽取实体，用 JinaCLIP embedding 做 K-means 聚类，以聚类中心作代表实体，以此解决 "raccoon" 与 "procyon lotor" 这类同物异名。原始帧用 JinaCLIP 向量化后挂到事件上。整个 EKG 包括事件-事件、实体-实体、实体-事件三种关系，存成 5 张表 — [AVA arXiv HTML v3](https://arxiv.org/html/2505.00254v3)；[NSDI'26 page](https://www.usenix.org/conference/nsdi26/presentation/yan)
- 吞吐（输入固定 2 FPS）：2×A100 上 6.7 FPS，单张 4090 上 4.4 FPS，单张 3090 上 2.5 FPS — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)
- 索引对比（305 道 LVBench 题，约 1.2 h 视频）：AVA EKG 39.7%，LightRAG 30.6%，MiniRAG 28.1% — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)

**STEG-OVR（arXiv 2609.23393，2026-09，cs.DB）**
- ingest 时把持续存在的物体存成 tracklet 节点，时间上相容的物体对存成关系边，证据包括外观、运动、场景占据、相对几何、速度相容性和符号关系 — [STEG-OVR abstract](https://arxiv.org/abs/2609.23393)
- 绝对 AP 仍低：Beach 从 0.0882 到 0.2073，Shibuya 从 0.0834 到 0.1505；在 LOVO 风格对比中从 0.701 到 0.743 — [STEG-OVR abstract](https://arxiv.org/abs/2609.23393)
- 一篇 2026 论文对 LOVO 的评价：LOVO 把离线视觉计算和在线查询分离得好，但可检索证据仍主要是帧级物体证据，"停住"这类状态和持续关系只能近似 — [WebSearch 摘要转述 arXiv 2609.23393](https://arxiv.org/pdf/2609.23393)

**OTIF / MIRIS（轨迹预计算）**
- MIRIS（SIGMOD 2020）是 query-driven tracking，能低帧率时就用低帧率，查询需要时才提高；在相同精度下平均比 IOU tracker 快约 9× — [MIRIS paper](https://people.csail.mit.edu/arjunvb/pubs/miris-sigmod20-paper.pdf)；[MIT DSpace](https://dspace.mit.edu/handle/1721.1/132289)
- 一篇 2025 综述称 OTIF 基本是 MIRIS 的改进版：ingest 时就物化全部轨迹，查询时直接回答 — [Kang monograph arXiv 2511.03489（检索摘要）](https://arxiv.org/pdf/2511.03489)

### Inferences
- **我们的现象对应 Focus 的 "top-1 recall 不足" 加 Boggart 的 "预处理/查询模型错配"。** 通用 caption 相当于 writer 的 top-1 语义投影；问题要的是另一个投影（某个具体时刻的衬衫颜色）。Boggart 测得错配时最多 94% 的精度损失，说明这类损失是语义索引的结构性问题，不是 prompt 或粒度问题。这和"换 writer、prompt、粒度、D-HSM 比例都不变"是一致的。
- Focus 能靠 top-K 恢复 recall，有三个前提：(1) 标签空间闭合（1000 类），top-K 有明确含义；(2) 真值就是 GT-CNN 的输出，验证器按定义正确；(3) K 很大（60–200）时可以靠聚类摊薄代价。我们的属性空间是开放词表（颜色 × 物体 × 人 × 时刻的组合），7B 口头列出的"top-K 假设"也不是校准过的 softmax。所以 Focus 的 recall 曲线不能直接套到我们这里，只能当作定性警告：便宜模型的 top-1 与正确答案的距离，可能比想象的远。
- Boggart 和 Focus 都依赖静态摄像头（Boggart 明确写了；Focus 的数据集是交通/监控视频）。OVO-Bench 的 EPM/ASI 多为第一人称视角、摄像头在动，基于背景减除的 blob 和轨迹基本无法直接迁移。
- AVA 是领域内和我们最接近的设定：用 7B VLM 写 query-agnostic 文本索引，再挂帧 embedding。它和我们的区别在于：每 3 s 一块（我们是 16 s）、实体做聚类、帧作为第二个检索视图保留；面向的是离线长视频，不是流式。

### Gaps
- Focus 中 K=1 时的具体 recall 数值没从文本里抽到（只有图 5 曲线），所以无法给出"top-1 → top-K 提升多少点"的精确数字。
- 本次未拿到 TASTI-PT 与 TASTI-T 的具体数值差（只有定性表述）。
- 未核实 Boggart（据我所知是 NSDI 2023）、Everest（据我所知是 SIGMOD 2021）、LOVO（DOI 显示 ICDE 2025）的正式 venue。arXiv 页面都没列出。
- 本次未检索 NoScope（VLDB 2017）、BlazeIt（VLDB 2020）、Chameleon（SIGCOMM 2018）、VIVA、EVA、VOCAL 的原文。按背景知识：NoScope/BlazeIt 要求查询已知并训练专用 proxy，Chameleon 做在线配置自适应（分辨率、帧率、模型），都与"查询未知"关系弱。这一点未核实。
- 没有找到任何系统给出"开放词表属性（颜色、手持物）的 ingest recall"数字。

## Q2：怎样把昂贵、精确的处理推迟到查询时？存储与延迟代价多大？

### Takeaway
共同模式是：ingest 时存廉价、高 recall 的候选结构（top-K、中心、轨迹、embedding、分布），query 时只在**少量候选/代表**上跑"更强或被定义为真值"的模型，再把结果**传播**回去。传播的方式有簇成员继承、轨迹传播、embedding 近邻传播和时间连续性。验证预算通常是全部帧的 2–54%。所有方案都有一个隐含前提：query 时用的模型比 ingest 模型更准，或者干脆就被定义为真值。我们的 training-free 7B 设定恰恰缺这个前提。

### Cited Findings
- **Focus**：query 时先查 top-K 索引里与类 X 匹配的簇，只把簇中心送进 GT-CNN，再把中心的标签赋给簇内所有物体；返回中心被判为 X 的所有帧 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- **Boggart**：query 时先快速刻画索引对该查询（CNN、精度目标、视频）的不精确程度，只在代表帧上跑用户 CNN（默认让中心覆盖 2% 的视频），再沿轨迹传播结果；对检测类查询，利用关键点间相对位置稳定来传播 box — [Boggart paper](https://arxiv.org/pdf/2106.15315)
- Boggart 在 80/90/95% 精度目标下都能满足目标，同时只在 3–54% 的帧上跑 CNN；90% 目标时，各模型 GPU 小时的中位占比为：二分类 3–6%，计数 4–11%，检测 8–28%。chunk 大小在 0.2–10 min、中心覆盖率在 0.5–5% 之间变化时，性能变化 < 5% — [Boggart paper](https://arxiv.org/pdf/2106.15315)
- **TASTI**：query 时只在 cluster representatives 上算精确结果，其余记录用 k 个最近代表的结果传播（例如计数取近邻代表的车数）— [TASTI paper](https://arxiv.org/pdf/2009.04540)
- **Seiden（PVLDB 16(9)，2023；Bang, Kakkar, Chunduri, Mitra, Arulraj）**：作者指出 "新 oracle 模型推理时间已与 proxy 相当甚至更低"，于是不用 proxy，直接在间隔较大的 I 帧子集上跑 oracle，建成 query-agnostic 索引；未采样帧靠时间连续性拿到 proxy 标签。查询时用 exploration–exploitation 再追加采样。平均比 SoTA VDBMS 快 6.6×，精度也更高 — [Seiden PVLDB PDF](https://www.vldb.org/pvldb/vol16/p2289-kakkar.pdf)；[Adobe Research page](https://research.adobe.com/publication/seiden-revisiting-query-processing-in-video-database-systems)
- **Everest**：Phase 2 把 Top-K 处理看作"在线不确定数据清洗"，只挑最能改变答案置信度的不确定元组交给 oracle 清洗，直到置信度 ≥ thres；目标是尽量少清洗 — [Everest paper](https://arxiv.org/pdf/2003.00773)
- **Zelda（arXiv 2305.03785，2023）**：直接把自然语言查询交给 VLM 检索，不用每个谓词一个模型。它自动给 prompt 加 discriminator 词和同义词，加入识别低质量帧的词，再用 VLM embedding 去掉相似帧以保证多样性。相比开箱即用的 VLM，mAP 最多提升 1.15×；相比一个 SoTA 视频分析引擎，同等精度下快 7.5×（最高 10.4×）— [Zelda abstract](https://arxiv.org/abs/2305.03785)
- **LOVO**：query 时先把文本编码成向量，在 PQ 中心上做 ANN，对 top-k 精确重算，再用 cross-modal transformer 重排输出 box。search 比 MIRIS 快 99×、比 FiGO 快 85×；总时间 LOVO 约 145–220 s，VISA 约 510–1890 s，ZELDA 约 60.7–218 s。LOVO 的检索阶段比 ZELDA 慢（约 23–27 s 对 3–5 s），但 AveP 更高（Q1.1 0.91，Q2.2 0.29）— [LOVO HTML](https://arxiv.org/html/2507.14301)
- **AVA**：query 时做 tri-view 检索（事件、实体、帧 embedding 三路，各取 top-K，用加权 Borda 融合），再做 agentic tree search（Forward/Backward/Re-query/Summary&Answer 四个动作，深度 3 时 13 条路径，事件列表上限 16）。**"Check Frames and Answer"（CA）只在两个最高候选答案节点不一致时触发**：取回对应事件的帧，让 VLM 依据视觉证据作答；实现中 CA 用 Gemini-1.5-Pro。最后 8 次自一致采样，打分 = 0.3 × 答案一致率 + 0.7 × 推理链 BERTScore — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)
- AVA 的 tree-depth 消融（深度 1/2/3/4；抽取文本未写明是哪个基准）：只用 Qwen2.5-14B（纯文本）为 34.1/36.1/40.9/39.5；加 Qwen2.5-VL-7B CA 为 49.3/52.1/53.8/50.2；加 Gemini-1.5-Pro CA 为 54.2/58.4/61.5/52.7。论文没有严格的有/无 CA 对照，纯文本行与 CA 行之间不止一处不同 — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)
- AVA 各阶段延迟：检索 0.44 s；tree search 101.5 s（14B）/ 174.2 s（32B）；CA 85.2 s（Qwen2.5-VL-7B）/ 50.2 s（Gemini）。深度 1–4 的 tree search 分别为 6.7/27.3/90.1/370.3 s。论文另称只用 EKG 文本的 Qwen2.5-32B/7B 在三个基准上胜过 Qwen2.5-VL-7B baseline，但正文没给数 — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)
- **STEG-OVR**：query 时把自由文本查询分解成实体/状态/场景/时间/关系五类槽位，只激活对应的检索通道做 soft score fusion，最后做一次**固定预算的一致性验证** — [STEG-OVR abstract](https://arxiv.org/abs/2609.23393)
- 存储数据点：Boggart 306 MB/h 视频（约为 H.264 视频的 30%）— [Boggart paper](https://arxiv.org/pdf/2106.15315)；Focus 把 top-K 索引存进 MongoDB，ingest 时聚类、只存中心以压缩索引 — [Focus paper](https://arxiv.org/pdf/1801.03493)

### Inferences
- **关键的不对称性**：Focus 的 GT-CNN 按定义就是真值；Everest/Seiden 的 oracle 是"昂贵但正确"的模型；AVA 最强的配置把 CA 换成了 Gemini-1.5-Pro（比 7B CA 高 7.7 点：61.5 对 53.8）。我们是同一个 7B 既写又验，没有更强的验证器。因此能利用的只有"让验证任务本身更容易"：只验候选、用 crop 提高有效分辨率、问单一的二值或盲问题。指望靠"重看"拿到更强的判断力，不现实。
- Seiden 的论点（proxy 与 oracle 速度差已经消失）与我们一致：writer 和 reader 是同一模型。Seiden 的应对是"在稀疏样本上跑 oracle，再按不确定性追加采样"。我们的 16 s caption 本质上就是 Seiden 式均匀 oracle 采样；它的追加采样对应我们的 query-time 重看，而重看已被证实会编造证据。
- AVA 中 7B CA 相对纯文本有 +12.9 点（深度 3：40.9 → 53.8），这是领域内唯一接近"7B 在 query 时重看帧"的正面数据。但它有三个混淆：离线长视频、14B 推理器加 8 次自一致、只在候选不一致时触发。它不能直接否定我们"7B 重看会编造"的观察，只说明"触发条件 + 候选限定 + 多次采样一致性"可能是让重看变得有用的关键。
- 延迟：AVA 每题约 150–260 s（我把各阶段相加得到，论文未报总数），对流式 QA 不可接受。可借鉴的是它的结构，不是它的预算。
- 存储估算（我的算术，不是来源数据）：每 2 s 存一帧 448 px JPEG（约 30–50 KB），每小时约 1800 帧，即约 55–90 MB/h；只存物体 crop 会更少。这与 Boggart 的 306 MB/h 同一量级或更低。可见存储不是瓶颈，瓶颈在验证的可靠性。

### Gaps
- 没拿到 Focus/Boggart 的 query 时绝对延迟（秒级），只有相对倍数。
- AVA 的 CA 每次用多少帧、用整帧还是 crop，论文抽取文本中都没写。
- Seiden 的 exploration–exploitation 细节（采样预算、怎样选窗口）只从摘要获知，未读全文。

## Q3：哪些思路对应我们的两个失败 bucket？哪些可能降低 query-time 重看时的编造？

### Takeaway
bucket (a)（记忆里没有答案信息）对应的是 ingest 的 recall 问题。领域给的答案是"保留可在 query 时重新解释的原始或低层证据"（Boggart、Seiden、AVA 的帧视图、LOVO 的 patch），而不是"写得更好"。bucket (b)（答案和干扰项都在记忆里）对应低 precision 的 top-K 或不确定索引，领域的答案是"只对冲突候选做定向验证"（Focus 验簇中心、Everest 清洗不确定元组、AVA 只在 top-2 不一致时触发 CA）。降低编造最直接的证据来自 2026 年一项小规模验证器消融：直接问 VLM 某条断言是否成立，捕获率为 0；先盲重描述、再用小型 NLI 模型判定，捕获率为 79%。

### Cited Findings
- **验证器设计（arXiv 2608.15574，2026-08，单作者，n=40）**：RAG 式视频 LLM 先给每条断言附带时间戳引用，再逐帧复核。三种复核方式：(1) 直接问视觉模型断言是否成立，在 40 条断言上捕获率 0%，作者归因于 sycophancy；(2) 先盲重描述（不给断言），再交给通用 LLM 判定，结果随 prompt 措辞在 0% 和 100% 之间摆动；(3) 用小型 NLI 模型判定，在对抗性 false-premise 问题上捕获 79% 的编造断言，且不误伤真断言。摘要没有给出 precision 和假阳性率 — [arXiv 2608.15574](https://arxiv.org/abs/2608.15574)
- **VideoHEDGE（arXiv 2601.08557，2026-01）**：对干净片段和扰动片段分别采样答案，用熵检测视频 QA 幻觉；在 3 个 7B 视频模型上，某个熵变体的 ROC-AUC 最好 — [papers.cool 2601.08557（检索摘要）](https://papers.cool/arxiv/2601.08557)
- **AVA 的触发式验证**：CA 只作用于 top-2 且答案不一致的节点；作者把 CA 的动机写成弥补 "may have been overlooked during the construction phase" 的信息，即承认 ingest 索引有缺失 — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)
- **Focus 验证的是物体 crop，不是整帧**："Focus clusters the objects in the frames and not the frames as a whole"，GT-CNN 的分类对象是物体 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- **Zelda 的 discriminator 词和低质量帧过滤**专门针对 VLM 检索 precision 低的问题 — [Zelda abstract](https://arxiv.org/abs/2305.03785)
- **Everest 的不确定关系表**会显式表示"多个候选并存"，并按置信度阈值决定验证到哪里为止 — [Everest paper](https://arxiv.org/pdf/2003.00773)
- **Boggart 的 "imprecise but comprehensive" 原则**：索引宁可不精确，也不能漏，精确性留到 query 时补 — [Boggart paper](https://arxiv.org/pdf/2106.15315)

### Inferences
- **映射到 bucket (a)，约 44%**：
  - 领域的经验是，对未知查询而言语义索引的 recall 有上限（Boggart 的错配实验），只有保留原始或低层证据才"comprehensive"。对我们来说，bucket (a) 的唯一出路是 query 时能访问存下来的视觉证据。改写 caption（包括 top-K 假设）只能把一部分 (a) 变成 (b)，前提是 writer 至少注意到了那个物体。
  - 我们已观察到：比例对 writer、分辨率、粒度、prompt 都不敏感。这说明缺的信息很可能是 writer 根本没"看"的细节（时刻 × 物体 × 属性的组合），top-K 属性词救不回来。这是推断，可做离线验证：在已存帧上让 writer 用"列出 3 种可能"的格式重写，再用词重叠代理测 (a) 比例的变化。
- **映射到 bucket (b)，约 44%**：
  - 这是与 Focus/Everest/AVA 最贴合的情形：记忆里已经有竞争候选（对应 top-K 索引的低 precision），需要的是对候选做定向验证。领域方法默认验证器可靠，我们的风险集中在这里。
  - 一个可利用的结构：(b) 题里记忆本身提供了时间锚点，即哪条 caption 提到了 "gray shirt"、哪条提到了 "white shirt"。这对应 Focus 的"簇 → 中心"：只需回看这几个时间窗的帧或 crop，不必在全历史上做 CLIP 检索。这样可以绕开"CLIP 检索不准"的问题。
- **降低编造**：
  - (1) **只验候选**：问题从开放的"发生了什么"变成按选项逐一的封闭判断，借鉴 Focus/Everest。
  - (2) **盲描述 + 外部匹配器**：不把选项或断言写进 VLM prompt，先让它描述 crop 里的属性，再用字符串、embedding 或预训练 NLI 与选项匹配，借鉴 2608.15574。直接问 "you took a knife?" 恰好是其中捕获率 0% 的那种设置。
  - (3) **用 crop，不用整帧**：提高物体的有效分辨率，减少无关上下文，借鉴 Focus 与 LOVO。
  - (4) **验证放在单独一次调用里，只把文本判定结果传回 reader**：避开"历史帧稀释当前感知"，也符合我们"放文本不伤感知"的观察。
  - (5) **多次采样一致性或熵**作为拒答信号，借鉴 AVA 和 VideoHEDGE；验证不一致时保留原 caption 的判断。
- **诚实的弱点**：2608.15574 只有 40 条断言、单作者、未经同行评审；AVA 的 CA 收益有混淆。目前没有领域证据表明 7B 在 crop 级定向验证上足够可靠。

### Gaps
- 没有找到 VDBMS 领域专门量化"验证器幻觉或编造率"的论文。该领域的 oracle 默认就是正确的，这本身是领域盲区。
- 2608.15574 的完整设置（用哪个 VLM、看整帧还是 crop、NLI 用哪个模型）摘要没写，需要读全文。

## Q4：每个候选思路在 7B writer/reader 下怎样做 training-free 适配？结合我们的证据，风险多大？

### Takeaway
最值得先做的，是两个成本低、直接针对 bucket (b) 和编造问题的组合：一是"由记忆冲突触发、限定在 MCQ 选项和冲突时间窗内的 crop 级盲验证"，二是把"外部匹配器"换成选项对比式打分。对 bucket (a)，只有"保留物体 crop 或关键帧 + 物体级 embedding"在原理上可行，但它又会落回我们已经失败过的"检索不准 + 7B 重看编造"，风险高。Focus 式 top-K 属性假设、TASTI/Seiden 式代表样本、Boggart/OTIF 式轨迹，在我们的设定里不是缺前提（闭集标签、静态摄像头、可训练 proxy），就是与已证伪的路线重合。

### Cited Findings
（以下每条的方法依据见 Q1–Q3 已引用的来源；本节的"适配方案"和"风险"属于推断，统一放在 Inferences。）
- Focus 的 top-K 在通用 cheap CNN 上需要 K=60–200 才有 90% recall，K=2–4 只在流专用、经过重训的模型上成立 — [Focus paper](https://arxiv.org/pdf/1801.03493)
- TASTI 可以用纯预训练 embedding（TASTI-PT），但通常不如训练版 — [TASTI paper](https://arxiv.org/pdf/2009.04540)
- Boggart 的低层索引目前只支持静态摄像头 — [Boggart paper](https://arxiv.org/pdf/2106.15315)
- Everest 的不确定 proxy（CMDN）需要用 oracle 标注的少量样本训练，耗时几分钟 — [Everest paper](https://arxiv.org/pdf/2003.00773)
- AVA 全流程只用预训练模型（Qwen2.5-VL-7B、JinaCLIP、Qwen2.5-14B/32B、Gemini），属于 training-free 系统；7B CA 在其消融中优于纯文本 — [AVA HTML v3](https://arxiv.org/html/2505.00254v3)
- 直接问 VLM 某条断言是否成立的捕获率为 0%，盲描述 + NLI 为 79%（n=40）— [arXiv 2608.15574](https://arxiv.org/abs/2608.15574)

### Inferences
按优先级排列（风险评估结合我们已有的负面证据）：

1. **冲突触发 + 选项限定 + crop 级盲验证（Focus 验中心 + Everest 清洗 + AVA 的 CA 触发 + 2608.15574 的盲描述/NLI）**：对应 bucket (b)，并降低编造。
   - 做法：
     - (i) ingest 时除 caption 外，按固定间隔（如每 2 s）保存关键帧到磁盘，不进上下文。
     - (ii) query 时用现有词重叠代理识别"记忆同时支持 ≥2 个选项"的题。
     - (iii) 取回提到各竞争选项的 caption 对应的时间窗，每个选项 1–2 帧。
     - (iv) 用 Qwen2.5-VL 的 grounding 能力或预训练开放词表检测器（OWL-ViT、Grounding DINO 这类预训练模型，无需训练）截出相关物体的 crop。
     - (v) 单独一次调用，盲问"这个物体是什么颜色/是什么"，prompt 里不出现选项。
     - (vi) 用字符串、embedding 或预训练 NLI 把回答与选项匹配。
     - (vii) 只有在多次采样一致、且结论与其中一个候选吻合时才覆盖原判断，否则保持 text-only 答案。
   - 预期风险：**中**。
     - 正面：只动 (b)（约 44%）；时间锚点来自记忆本身，绕开了 CLIP 检索；盲问避开了 sycophancy 式编造。
     - 负面：EPM 的"哪个时刻"可能比 caption 粒度（16 s）更细；第一人称视角中物体被遮挡、运动模糊；7B 的颜色和细粒度识别本身可能就是错误来源，因为当初 writer 写 "gray/white" 可能正是感知错误，而不只是没写全。
   - 它需要的门控只是"是否触发验证"。我们已有结论"gate 不是收益来源"，这里的收益必须来自验证本身的准确率。建议先在已知 (b) 题上离线测"盲 crop 验证的准确率"，再决定是否接入。
2. **选项对比式检索/打分（Zelda 的 discriminator 词 + 多样性剪枝）**：对应 bucket (b)，并缓解 CLIP 不准。
   - 做法：用 CLIP/SigLIP 对冲突时间窗内的帧或 crop 计算 sim(选项 i) − max_{j≠i} sim(选项 j)，再加入同义词和低质量帧负词，取差值而不是绝对相似度。
   - 风险：**中**。Zelda 的提升本来就不大（mAP 最多 1.15×）。CLIP 对细粒度属性（衬衫是蓝还是灰）和动作（拿刀）分辨力有限，适合作为方案 1 的廉价预筛或第二票，不宜单独使用。
3. **top-K 属性假设写入（Focus top-K）**：主要把 (a) 转成 (b)。
   - 做法：writer 对显著实体输出 "shirt: blue | gray (unsure)" 这类多候选槽位；或者在属性槽位上做约束解码，读取 top-K token 的 logprob，这是 training-free 的近似校准。
   - 风险：**高**。属性词表开放，组合爆炸；7B 口头给出的不确定性未经校准；我们已证明换 prompt 和结构化记忆不改变比例，说明瓶颈在"没注意到"，而不在"只写了 top-1"。即使有效，也只是把题从 (a) 搬到 (b)，仍要靠方案 1 的验证器收尾。可做一次低成本离线测试，看 (a) 比例是否下降。
4. **物体 crop/embedding 证据库（LOVO patch 索引、Focus 物体聚类、AVA 实体聚类）**：理论上唯一能救 (a) 的方向。
   - 做法：ingest 时用预训练检测器或 VLM grounding 截出人、手持物、关键物体的 crop，存 SigLIP/JinaCLIP embedding、时间戳，并按 embedding 聚类（同一物体跨时间合并，参考 Focus 和 AVA）。query 时按选项文本做 ANN，再走方案 1 的盲验证。
   - 风险：**高**。在没有时间锚点的 (a) 题上，又回到"全历史检索 → 不准"和"7B 重看 → 编造"这条我们已经证伪的路线。第一人称视角下检测器的漏检率也未知。物体级比帧级好的地方只在于检索单元更干净，没有证据表明这足以扭转我们之前的 CLIP 结果。存储不是问题（估算低于 100 MB/h）。
5. **TASTI/Seiden 式"代表样本加密写入 + 传播"**：对应 bucket (a) 的一小部分。
   - 做法：用帧 embedding 做 FPF 或新颖度检测，对语义上新颖的帧追加一条详细描述，常规帧沿用 16 s caption。
   - 风险：**中高**。EPM/ASI 缺的大多是常规场景里的细节，不是新场景；TASTI-PT 本身弱于训练版。这条线对 ASI（动作序列，新动作常伴随视觉变化）也许比 EPM 略有用，属推测。
6. **Everest 式不确定记忆 + 置信度阈值**：作为方案 1 的触发器或拒答机制。
   - 做法：用 7B 在 caption 记忆上对各选项打出的 logprob 作为"可能世界"概率，低于阈值才触发验证。
   - 风险：**低**（实现成本低），但**收益上限也低**，它本身不增加信息。这和我们"gate 不是收益来源"的结论一致。
7. **低层轨迹索引（Boggart、OTIF、MIRIS、STEG-OVR）**：
   - 风险：**高，契合度弱**。Boggart 的 blob 和背景减除要求静态摄像头；轨迹类方法面向计数、检测、轨迹谓词，不面向"属性 × 时刻"的问答。STEG-OVR 的绝对 AP 也很低（约 0.15–0.21）。
8. **NoScope、BlazeIt、Panorama、Chameleon**：
   - **基本不适用**。前两者需要查询已知并训练专用 proxy；Panorama 要用参考样例做 embedding 检索，而我们没有选项对应的参考图；Chameleon 是在线配置自适应（据背景知识，未核实）。

**总体判断**：视频分析系统领域对"查询未知"的核心答案是"ingest 时保证 recall，query 时用更强的模型在候选上恢复 precision"。我们缺少"更强的 query 时模型"。在不训练、不换更大模型的前提下，能迁移的只有"缩小验证问题"这一半：限定候选、限定时间窗、用 crop、盲问、外部匹配、一致性拒答。它只可能改善 bucket (b)，对 bucket (a) 几乎无能为力。如果把这一点写成 findings，可以表述为：查询未知时，语义记忆存在结构性的 recall 上限，Boggart 的错配实验和 Focus 的 top-K 曲线是这一点的跨领域旁证。

### Gaps
- 没有找到 VDBMS 领域在**流式**、**第一人称**、**属性 × 时刻问答**上的任何评测。所有迁移判断都是外推。
- 没有找到用 7B 级 VLM 做 crop 级 oracle 的可靠性数据；AVA 的 7B CA 收益有混淆，2608.15574 样本太小。
- 本次没有检索到 2026 年 VDBMS 顶会（SIGMOD/VLDB/NSDI 2026）里除 AVA 和 STEG-OVR（预印本）之外、专门处理"VLM ingest 错误 + query 时验证"的系统。可能存在但未被检索到。
