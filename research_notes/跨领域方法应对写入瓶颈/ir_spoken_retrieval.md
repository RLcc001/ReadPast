# IR / 口语内容检索中的"有损、与查询无关的索引"方法：document expansion 与不确定识别结果（lattice / N-best / confusion network）的索引，以及向 StreamKV（训练无关流式视频 QA 写入瓶颈）的迁移评估

> 研究日期 2026-11-02。范围：spoken document retrieval (SDR) / spoken term detection (STD)、IR document expansion 与索引单元、OCR 噪声、"保留不确定性能否帮助下游 reader"。
> 本问题的两个桶（来自任务背景）："absent"（约 44%，写下的记忆里没有答案信息）与 "ambiguous"（约 44%，正确选项词和干扰项词同时出现）。
> 引用的数字都来自原文 PDF/HTML（Chelba 2005、Saraclar 2004、Doc2Query--、Lee 2015 综述、PNP-VQA 用 pypdf 抽取全文核对过；Everson 2024 用 arXiv HTML 核对；Singhal & Pereira 1999 的数字来自与论文同方法的 AT&T 专利正文，不是论文本身）。

## Q1：lattice / N-best / posterior 加权索引比 1-best 转写能提升多少检索 recall/MAP（尤其 1-best 错误率高时）？存储代价多大？

### Takeaway
在高 WER（40–55%）下，lattice/PSPL/WCN 索引相对 1-best 的提升确实存在，但幅度不大：在讲座检索上 MAP 相对提升约 20%（0.53→0.62），在 STD 上 maxF 绝对提升 +0.8 到 +2.9，代价是索引变大 3–9 倍（剪枝过松时达到约 194 倍，且没有额外收益）。提升的前提是正确词在假设空间里（lattice oracle WER 22% 对 1-best 45%）。识别器从未产生的词（OOV、被剪掉的弧）lattice 救不回来，语音界对此的办法是回到信号本身（子词索引、查询时插弧再重打分）。

### Cited Findings
**PSPL（Chelba & Acero, ACL 2005）**
- 语料：MIT iCampus 讲座，未针对该场景调过 ASR。第一讲 L01 的 1-best WER 为 44.7%，OOV 3.3%；全部讲座平均 WER 54.8%（44%–74%）— [Chelba & Acero 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/2005-chelba-acl.pdf)
- 检索结果（96 个查询，平均 1.97 词，含 OOV 的查询已剔除；检索权重和链接打分权重都没调）：MAP 为人工转写 0.99、ASR 1-best 0.53、PSPL lattice 0.62（相对 +20%）。R-precision 0.53→0.58。1416 个相关文档中检出 1088→1301 个，但返回总数也从 3206 增到 4971 — [Chelba & Acero 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/2005-chelba-acl.pdf)
- 为什么有效：lattice 的 ORACLE WER 远低于 1-best（3-gram lattice 26.4%，PSPL 21.7%，1-best 约 45%）。作者的启发式估算是：对 2 词查询，在 PSPL 中命中的概率约为 1-best 的 (1−0.22)²/(1−0.45)² ≈ 2 倍。相邻 PSPL bin 的内容"相当随机"，所以一旦命中一个典型 2-gram 多半是真的说过，precision 掉得不多 — [Chelba & Acero 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/2005-chelba-acl.pdf)
- 存储：L01 上 3-gram lattice 占 11.3 MB，PSPL 占 3.2 MB，node density 从 7.4 降到 1.1，1-best WER 只差 0.3 个百分点。文中没有给出 1-best 文本索引的大小 — [Chelba & Acero 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/2005-chelba-acl.pdf)
- 后续 CSL 2007 "Soft indexing of speech content" 报告：用语音内容加纯文本元数据，比只用元数据 MAP 相对提升 302%。这是"语音 vs 无语音"的比较，不是 lattice 和 1-best 的比较；数字只来自检索摘要片段，未读原文 — [MSR 页面](https://www.microsoft.com/en-us/research/?p=155685)

**Lattice-based STD（Saraclar & Sproat, HLT-NAACL 2004）**
- 1-best 基线（WER / Precision / Recall）：Broadcast News 约 20% / 92% / 77%；Switchboard 约 40% / 74% / 47%；Teleconferences 约 50% / 65% / 37%（OOV 依次为 0.6%、6%、12%）— [Saraclar & Sproat 2004](https://aclanthology.org/N04-1017.pdf)
- 词 lattice 对 1-best（maxF，索引大小）：
  - BN：84.0→84.8，29→91 MB（约 3 倍）
  - SWB：57.1→58.4，18→90 MB（约 5 倍）
  - Teleconf：47.4→50.3，16→142 MB（约 9 倍）
  - Teleconf 剪枝放宽到 cost=12：3100 MB，maxF 50.1，没有更多收益
  - 结论：WER 越高收益越大，但收益很快饱和，存储线性甚至超线性增长 — [Saraclar & Sproat 2004](https://aclanthology.org/N04-1017.pdf)
- 摘要里的"高 WER 时 F 提升超过 5 个点"来自词 lattice 加音素 lattice 的 search cascade（先查词索引，没结果再查音素索引）：Teleconf 上 maxF 52.8，比 1-best 的 47.4 高 5.4。音素索引若直接做音素识别，maxF 只有 30.4，从词 lattice 转换得到则是 45.6 — [Saraclar & Sproat 2004](https://aclanthology.org/N04-1017.pdf)

**WCN 索引（Mamou, Carmel, Hoory, SIGIR 2006，IBM 呼叫中心）**
- 呼叫中心数据 WER "30% 或更高"。用带后验的 word confusion network 索引，MAP 高于原始 1-best 转写，并且在极高 WER 下 MAP 仍"合理"。我只拿到摘要，没有具体数字 — [IBM Research 摘要页](https://research.ibm.com/publications/spoken-document-retrieval-from-call-center-conversations)

**综述（Lee, Glass, Lee et al., IEEE/ACM TASLP 2015）**
- TREC SDR 的 WER 为 15–20%，查询和文档都长，所以 SDR 一度被认为"solved"。后来发现级联方法主要在 ASR 准确率高时有效，遇到短查询、短片段、自发语音、OOV 和高 WER 时就不够用了 — [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)
- Chelba 2005 引用 TREC 的结论：1-best 索引的 MAP 在 WER 15%–30% 区间内基本持平。Woodland et al. 2000 把 query-OOV 提到接近 15% 后，MAP 从 44 降到 22（相对降 50%）— [Chelba & Acero 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/2005-chelba-acl.pdf)
- 1-best 准确率低时通常更倾向用 lattice，但 lattice 中错误假设太多"也会造成问题"，因此要用后验或置信度过滤。CN、PSPL 这类 sausage 结构用来降低内存 — [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)
- 识别器词表里没有的词"永远不可能被正确识别"。即使是词表内的罕见词，也会因为 LM 分数低在 lattice 生成时被剪掉。查询扩展时，伪相关文档里有用的词如果是 OOV 或被误识别，就"从未进入转写或 lattice，因此无法帮助" — [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)
- 语音界对"假设空间里没有"的补救是回到信号（retrieval-oriented decoding）：先用子词 lattice 找候选片段，再把 OOV 查询词作为新弧插进词 lattice，在对应时间段上用声学模型和 LM 重打分，不做完整重解码。在 iCampus 上对 OOV 查询比子词检索相对提升 8.7% — [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)

**OCR 噪声（简述，适配度弱）**
- OHRBench（"OCR Hinders RAG", arXiv 2412.02592, ICCV 2025）：8,561 张文档图像、8,498 个 QA。作者测试的 OCR 方案中没有一个足以构建高质量 RAG 知识库。semantic noise（识别错误）始终有显著影响，formatting noise 的影响因 retriever/LLM 而异，噪声程度与 RAG 性能呈趋势关系。这是一篇测量研究，不提供保留不确定性的修复方法 — [Zhang et al. 2024/2025](https://arxiv.org/html/2412.02592v2)

### Inferences
- 收益机制是：正确假设在 lattice 里，只是没排第一（PSPL：oracle 22% 对 1-best 45%）。在我们的设定里，这对应 "ambiguous" 桶的一部分，甚至更窄：只有"另一种描述在写入模型的采样分布里有可观概率"的情况。"absent" 桶（写入时根本没感知或没提）对应 OOV 和被剪枝的弧，lattice 式保留无效。语音界唯一有效的补救是回到原始信号做受约束的重打分（插弧加声学重打分），对我们而言就是查询时访问原始帧或视觉特征。
- 量级预期：语音里最好的情况是 MAP 相对 +20%，或 STD 的 F 值 +1 到 +3（加子词才 +5）。这些都是检索指标，不是 QA 准确率。而且 Saraclar 的数据显示，在低冗余、高 WER 的任务上，收益随剪枝放宽很快饱和。我们的问题是单个具体细节、冗余很低，属于语音里 1-best 最差、lattice 收益也有限的那一类。
- 存储：每段只存 1 条 caption 改为存 N 条采样或属性假设，预计 token 量涨 3–10 倍。对流式 reader 来说，这直接变成上下文长度压力，而不只是磁盘成本（见 Q3 中 PNP-VQA 的 FiE 约 40 条 caption 上限）。PSPL 的经验是：按位置或时间对齐后再合并假设（sausage），可以把 lattice 压到约 28%（3.2/11.3 MB），且不丢 oracle。对 caption 的类比是按实体×属性槽合并同段的多个假设，不要拼接 N 条全文。
- TREC SDR "MAP 对 WER 不敏感"靠的是长查询、长文档带来的冗余。我们的 caption 时间线对同一属性也有跨段冗余（如 "gray shirt"、"white shirt" 在多段出现），但它们是彼此矛盾的 1-best，没有后验。这更像多个独立识别器的 1-best 互相冲突，不是同一识别器的 N-best。

### Gaps
- Mamou et al. 2006 的具体 MAP 数字和 WER 曲线没拿到（只有摘要）。
- 没有找到"lattice 对 1-best 的增益随 WER 变化"的系统曲线（PSPL 只有一个语料，Saraclar 有三个点）。
- Chelba/Silva/Acero 2007 CSL 的 302% 只有检索片段，未核对原文。
- 没有专门检索 2025–2026 年把 lattice 式索引用于 LLM/VLM 记忆的工作；就本次搜索所见，没有直接对应的成果。

## Q2：当索引器感知不到缺失细节时，doc2query 式"在索引时预判可能的问题"有用吗？文献报告了哪些失败模式（幻觉扩展、Doc2Query--）？

### Takeaway
Document expansion 解决的是词汇不匹配，也就是"信息在文档里，但用的词不同"，不能凭空补出索引器没看到的事实。语音领域最早的 document expansion（Singhal & Pereira 1999）收益主要来自对已有词重新加权，且依赖同期的可比外部文本语料。Doc2query 生成的查询有大量幻觉，用独立的相关性模型删掉 70% 后效果反而 +16%，生成器自身的概率不能识别幻觉。Weller et al. 2024 发现，retriever 越强，扩展的收益越小甚至为负。对我们的 "absent" 桶，这类方法基本无效且有制造干扰项的风险；对 "ambiguous" 桶也帮不上忙，因为我们的瓶颈不在词汇匹配（BM25 选择反而更差）。

### Cited Findings
**Document Expansion for Speech Retrieval（Singhal & Pereira, SIGIR 1999）**
- 数字取自同方法的 AT&T 专利 US7395207 正文。语料约 100 小时广播新闻，2,866 个故事，23 个查询。扩展方法：把每份 ASR 转写当 tf-idf 查询，在同期的 NA News 文本库中取 10 个近邻，用 Rocchio 合并 — [US7395207 (Google Patents)](https://patents.google.com/patent/US7395207)；论文元数据见 [SIGIR'99](https://ir.webis.de/anthology/1999.sigirconf_conference-99.6/)
- 平均精度（短查询）：
  - 最差转写 DERASRU-S1（WER 最高约 65%）：0.3139→0.4576（+46%）
  - 人工转写：0.4277→0.5265（+23%），其中只做重加权已达 0.5017
  - 高 WER 转写与人工转写的差距从 22–27% 缩小到约 12–13%
  - 来源：[US7395207](https://patents.google.com/patent/US7395207)
- 失败或边界条件：
  - 换成不相关的 1987–1994 语料后，人工转写短查询只 +12%，长查询反而 −7%
  - 对完美文本，大部分收益来自已有词的重新加权，新增词只贡献约 6%（短查询），长查询时新增词带来小幅损失
  - 来源：[US7395207](https://patents.google.com/patent/US7395207)
- 综述总结：对口语内容，基于 latent topic 的 document expansion 比 query expansion 有效（查询太短，估不准主题）；来自 web 的外部信息有助于缓解 ASR 错误 — [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)

**Doc2Query--（Gospodinov, MacAvaney, Macdonald, ECIR 2023）**
- docTTTTTquery（T5）生成的查询中，很多无法由源段落回答。在作者人工打分的例子里，不少查询得 0–1 分（如段落讲大麦，生成 "what is bare wheat"）— [Doc2Query-- (arXiv 2301.03266)](https://arxiv.org/pdf/2301.03266)
- MS MARCO Dev 的 RR@10：
  - BM25 0.185，索引 0.71 GB
  - Doc2Query n=80：0.279，索引 1.41 GB
  - 加 ELECTRA 过滤（只保留得分前 30% 的生成查询）：0.323（+16%），索引 0.95 GB（−33%），平均响应时间 30→23 ms
  - TREC DL'19 nDCG@10 从 0.627 到 0.670，但不显著
  - 来源：[Doc2Query--](https://arxiv.org/pdf/2301.03266)
- 关键负面结果：作者也试过用生成过程自身的概率做过滤，"发现无效"，所以没有报告。有效的过滤器是独立的相关性模型（ELECTRA、MonoT5 cross-encoder 效果最好，TCT-ColBERT bi-encoder 较弱），都是 MS MARCO 上训练的现成模型 — [Doc2Query--](https://arxiv.org/pdf/2301.03266)
- 作者同时指出 SPLADE 一类 MLM 扩展也会幻觉，例如给大麦段落扩展出 "reed"、"herb"、"troy" — [Doc2Query--](https://arxiv.org/pdf/2301.03266)

**扩展何时失败（Weller et al., Findings of EACL 2024）**
- 覆盖 11 种生成式查询/文档扩展、12 个数据集、24 个 retriever。retriever 性能与扩展收益强负相关：扩展帮助弱模型、伤害强模型。原因是扩展加入的关键词会被模型关注，掩盖原文的相关性信号，即使提高了召回也更难区分 top 相关文档。建议只在弱模型或数据格式差异大时使用；长查询分布偏移是例外 — [Weller et al. 2024](https://arxiv.org/abs/2309.08541v1)；[ACL Anthology](https://preview.aclanthology.org/naacl24-info/2024.findings-eacl.134)

**HyDE（Gao et al., ACL 2023；查询时生成，不是索引时）**
- 用指令 LLM 零样本生成一篇假设文档，再用无监督的 Contriever 编码去检索。作者承认生成的文档可能含错误细节，并称 dense encoder 的瓶颈会过滤掉这些细节。整个流程不需要训练 — [HyDE (arXiv 2212.10496)](https://arxiv.org/abs/2212.10496)

**索引单元：Dense X Retrieval / proposition（Chen et al., EMNLP 2024）**
- 把 Wikipedia 按 100 词段落、句子、proposition（自洽的原子事实）三种粒度建索引（FactoidWiki），测试 6 个 dual-encoder、5 个开放域 QA 数据集。proposition 对无监督 retriever 的 Recall@20 平均约 +10.1，对有监督 retriever 只有 +2.2（v1）到 +2.7（v3）。同等 token 预算下下游 QA 也更好 — [Dense X Retrieval (arXiv 2312.06648)](https://arxiv.org/html/2312.06648v3)
- 拆分器 Propositionizer 是微调过的模型（需要训练组件），可以用 LLM 提示拆分替代 — [Dense X Retrieval](https://arxiv.org/html/2312.06648v3)

### Inferences
- 我们不缺词汇匹配：已知只给 reader 最匹配的 caption 不比全时间线好，BM25 选择比 recency 更差。document expansion、doc2query、HyDE、proposition 解决的都是"信息在、找不到"，我们的问题是"信息不在"（absent）或"信息自相矛盾"（ambiguous），因此这一类 IR 方法基本对不上。
- "在写入时预判问题"（如让 VLM 为每段写 "可能被问的问题＋答案"）在 absent 桶上等价于 doc2query 的幻觉：写入模型没看清颜色，生成的 Q&A 就会编一个颜色，这会把 absent 变成 ambiguous，甚至变成一个看似确定的错答案。Doc2Query-- 说明过滤必须靠独立的验证器，而不是生成器自身的概率。训练无关的替代是用现成的图文匹配模型（CLIP、SigLIP 一类）对每条生成的属性陈述和当时的帧打分后再保留。这一点是推断，文献没有在视频记忆上验证过。
- Singhal & Pereira 的有效来源是同期外部文本库，视频情节细节（某人穿什么颜色）没有外部可比语料可借，所以 expansion 在这里没有信息来源。
- Weller 2024 的"强模型被扩展伤害"对我们是警告：7B/8B reader 已能读完整时间线，额外扩展文本更可能加噪声，而不是加信号。
- proposition 化（把 caption 改写成原子事实）属于读取侧整理，我们已知读取侧没有余量，且 D-HSM 实体图也没改变比例。预期收益很小。

### Gaps
- 没有核实 docTTTTTquery 原报告中"复制词与新词"的比例（印象中约 2/3 来自原文，但没有找到可引用的原文，不予采用）。
- Weller 2024 的具体数值（各 retriever 的增益表）没有抽取。
- 没有找到"索引器在物理上感知不到细节时 doc2query 的定量研究"。相关的只有 Doc2Query-- 的幻觉论证，属间接证据。

## Q3："保留带置信度的多个候选而不是 1-best"如何训练无关地用到 VLM caption 上？对应哪个桶？

### Takeaway
最接近的直接证据来自 PNP-VQA（零训练）：对同一图像采样 100 条与问题无关的 generic caption，相比 5 条，VQAv2 从 53.5 升到 58.6，OK-VQA 从 26.6 升到 31.9；beam search 产生的重复 caption 明显更差（55.3 对 top-k 采样的 62.1）；准确率随 "answer hit rate"（任一 caption 含答案）上升。这说明多次采样可以把一部分 absent 变成"有答案但混有干扰"，即覆盖率（oracle）提高，代价是 token 量和干扰项增多，并且要求 reader 能聚合（FiE 约 40 条 caption 就饱和）。真正解决 ambiguous 桶需要在 reader 外做显式聚合（频率或后验），或在查询时对选项做受约束打分；直接把候选塞进 7B reader 的 prompt，按 Everson 2024 的结果，很可能没用甚至有害。

### Cited Findings
**PNP-VQA（Tiong et al., Findings of EMNLP 2022，零训练）**
- 三个数字依次为 VQAv2 / OK-VQA / GQA 的零样本准确率 — [PNP-VQA (arXiv 2210.08773)](https://arxiv.org/pdf/2210.08773)

  | 设置 | VQAv2 | OK-VQA | GQA |
  |---|---|---|---|
  | 无 caption | 33.4 | 10.3 | 25.9 |
  | 全图 generic caption（与问题无关）× 5 | 53.5 | 26.6 | 36.5 |
  | 全图 generic caption × 100 | 58.6 | 31.9 | 39.8 |
  | 均匀随机 patch × 100 | 58.4 | 32.4 | 40.4 |
  | 问题引导 patch 采样（GradCAM，查询时）× 100 | 62.1 | 34.1 | 42.3 |
  | MS COCO 人工 caption × 5 | 56.9 | 28.1 | – |

- 解码方式（100 条，问题引导）：beam search 55.3/26.8/37.2，top-k(k=50) 62.1/34.1/42.3，nucleus(p=0.9) 61.3/32.9/41.7。作者发现 beam search 生成重复 caption，覆盖不到图像的不同方面 — [PNP-VQA](https://arxiv.org/pdf/2210.08773)
- 问题引导的 answer hit rate 随 caption 数从 20 增到 100，由 71.8% 升到 84.0%。在每种采样方式内，AHR 与 VQA 准确率正相关，但不完全对应（答案的位置和出现次数也有影响）— [PNP-VQA](https://arxiv.org/pdf/2210.08773)
- Reader 容量：Fusion-in-Encoder 在约 40 条 caption 时因输入长度到顶，Fusion-in-Decoder 能继续提升。Reader 的选择影响极大：同样 100 条 caption，GPT-J 6B 只有 28.7（VQAv2），UnifiedQAv2 3B 有 62.1 — [PNP-VQA](https://arxiv.org/pdf/2210.08773)
- 多次采样答案的一致性可作置信度信号（VizWiz-VQA 校准研究），这是采样答案而不是采样 caption — [arXiv 2406.00980](https://arxiv.org/pdf/2406.00980v1)

**语音侧对应物（见 Q1）**
- 后验过滤是必要的，lattice 中错误假设过多会带来问题 — [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)
- 按位置合并成 sausage（CN/PSPL）可以在不丢 oracle 的情况下压缩 — [Chelba & Acero 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/2005-chelba-acl.pdf)
- 查询时插入候选弧再做声学重打分，对 OOV 有效（+8.7% 相对）— [Lee et al. 2015](https://sls.csail.mit.edu/archives/root/publications/2015/Glass_IEEE-15.pdf)

### Inferences（迁移方案，均为训练无关；按与证据的接近程度排序）
1. **每段采样 K 条 caption（温度或 top-k，K≈5–20），离线合并成"属性 sausage"**：例如 "shirt: gray 0.5 / white 0.3 / blue 0.2"。频率充当后验，按 p≥阈值剪枝（类比 Everson 的 p≥0.3）。
   - 证据：PNP-VQA 的 5→100 generic caption 提升，以及 beam 不如采样。
   - 针对的桶：主要是 absent 中"模型其实感知到但 1-best 没写"的那部分（提高 oracle 覆盖），其次给 ambiguous 提供权重。
   - 已知风险：如果 7B 写入模型的感知本身就错（低分辨率、小物体），采样只会在错误分布里打转。我们观察到比例对写入模型和分辨率不敏感，这暗示相当一部分 absent 属于"写入模型根本没看到"，采样救不回来。建议先离线测：K 条采样的 union 能把 answer-overlap proxy 中 absent 的 44% 降多少，即我们自己的 "oracle WER"。这是最便宜的可证伪实验。
2. **在 reader 外做聚合（score fusion），而不是把候选塞进 prompt**：对每个 MC 选项，从属性 sausage 中算支持度（含时间衰减或跨段计数），再与 reader 的选项 logprob 融合。
   - 理由：Everson 2024 显示小模型读 WCN 会受损，只有最大的模型（GPT-3.5）稳定获益。
   - 针对的桶：ambiguous。
   - 风险：我们前几轮发现"gate 不是增益来源"，融合权重本身可能也没有信号；这需要实验确认，文献无法回答。
3. **查询时、选项条件的受约束打分（lattice 插弧重打分的类比）**：如果保留了历史帧或视觉特征，对每个选项构造陈述（如 "the man wears a blue shirt"），用 VLM 的条件 likelihood 或图文匹配分数去判别，而不是让 VLM 自由描述。
   - 理由：这把开放生成（已知会编造证据）变成 4 选 1 的判别，类似语音里不重新解码、只对候选弧重打分。
   - 针对的桶：absent（唯一可能触及它的方向，前提是原始视觉信息还在）。
   - 风险：7B VLM 的选项 likelihood 校准性未知，可能带有偏向长度或常见词的偏差。本次没有找到相关证据。
4. **写入时预判问题（doc2query 式）**：不推荐，理由见 Q2；如果做，必须用独立的图文匹配模型过滤（Doc2Query-- 原则）。
- 需要训练的组件及替代：
  - PNP-VQA 的问题引导需要 BLIP ITM 的 GradCAM（现成模型，但发生在查询时，且需要保留帧）
  - Doc2Query-- 的过滤器是 MS MARCO 训练的 cross-encoder；视觉侧可用 CLIP/SigLIP 现成替代
  - Propositionizer 是微调模型，可用 LLM 提示替代
  - 其余（采样、频率后验、WCN 提示）都不需要训练

### Gaps
- 没有找到在视频流式记忆上做"N-best caption / 属性后验"的直接研究。PNP-VQA 是静态图、单步 QA，reader 是 FiD 而不是 7B VLM 读长时间线。
- 没有关于 7B VLM 采样多样性能否覆盖 1-best 漏掉的属性的定量数据，需要自己测 union-oracle。
- VLM 选项条件 likelihood 的校准与偏差，本次未检索。

## Q4：有没有证据表明保留不确定性能帮助下游 reader/answerer（而不只是检索排序）？

### Takeaway
有，但很弱，而且依赖 reader 的规模。Everson et al.（ICASSP 2024，训练无关 ICL）把 WCN 以文本形式放进 LLM prompt 做口语 QA 和意图分类：零样本下 WCN 对所有模型都不如 1-best；单样本下 GPT-3.5 只多 +1.7 EM（43.0→44.7），ATIS 上 +0.4（84.6→85.0）；小模型（BLOOMZ 560M/3B）读 WCN 通常变差，不剪枝时 ATIS 从 66.8 掉到 51.9；WER 超过约 70% 时 1-best 反而更好。HyPoradise 显示 LLM 能利用 N-best 甚至修正 N-best 中不存在的 token，但最好结果需要微调，而且"补出列表外 token"在我们的设定里正是编造风险。

### Cited Findings
**Everson et al. 2024（arXiv 2401.02921，ICASSP 2024，训练无关）**
- 设置：
  - NMSQA 测试集 1-best WER 19.35%，n-best oracle WER 12.82%
  - ATIS 1-best WER 8.40%，oracle 5.31%
  - WCN 以 "//" 或 "||" 分隔候选并展开成文本，posterior 只用于过滤（p≥0.3，在 dev 上选），不写进 prompt
  - 过滤后平均每词候选数在 0% WER 的分箱为 1.04，在最高 WER 的分箱为 1.12
  - 来源：[Everson et al. 2024](https://arxiv.org/html/2401.02921v1)
- NMSQA 零样本（F1/EM）：GPT-3.5 的 1-best 为 64.5/36.5，最佳 WCN 为 63.2/35.3，人工转写为 71.7/42.5；不加 WCN 说明的 "//" 格式掉到 50.5/21.0。论文称零样本下没有任何 WCN 变体超过 1-best — [Everson et al. 2024](https://arxiv.org/html/2401.02921v1)
- NMSQA 单样本（F1/EM）：GPT-3.5 的 1-best 为 70.2/43.0，WCN 为 70.5/44.7，n-best oracle 为 72.4/45.1，人工转写为 76.2/47.9。WCN 补上了 1-best 到 oracle 之间 EM 差距的大部分（1.7/2.1）。BLOOMZ-3B 则 27.1→24.3 EM，反而下降 — [Everson et al. 2024](https://arxiv.org/html/2401.02921v1)
- ATIS 零样本准确率：GPT-3.5 的 1-best 84.6，WCN（"||"，p≥0.3）85.0，oracle 86.0，人工转写 88.1。BLOOMZ-560M 不剪枝时 "//" 只有 51.9，1-best 是 66.8 — [Everson et al. 2024](https://arxiv.org/html/2401.02921v1)
- 在 dev 集中有 ASR 错误的样本上：EM 1-best 42.3%，WCN 44.1%。WER 超过约 70% 时 1-best 更好。作者推测，从混淆候选中消歧可能是大模型才有的能力 — [Everson et al. 2024](https://arxiv.org/html/2401.02921v1)
- 早期需要训练的路线：Huang & Chen（ICASSP 2019）在 WCN 上微调预训练 LM，使声学易混词表示相近 — [搜索结果汇总](https://astro.paperswithcode.com/paper/learning-asr-robust-contextualized-embeddings)

**HyPoradise（Chen et al., NeurIPS 2023）**
- N-best 到转写的基准（arXiv 版 334K 对，NeurIPS 摘要写 316K，版本间不一致）。LLM 直接从 N-best 生成真实转写，而不是在候选中重排序。微调设置下可以超过 N-best 重排序的 oracle 上界；另有零样本/少样本 ICL 设置，不调参数。作者称合理的 prompt 能修正 N-best 列表里不存在的 token — [HyPoradise (arXiv 2309.15701)](https://arxiv.org/pdf/2309.15701)；[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract.html)

**PNP-VQA（见 Q3）**
- 下游 VQA 准确率随候选 caption 数量和 answer hit rate 上升，这是"保留更多假设 → 下游答题提升"的直接证据，但 reader 是专门的 QA 模型（UnifiedQAv2 + FiD）。换成 GPT-J 6B，同样输入只有 28.7 — [PNP-VQA](https://arxiv.org/pdf/2210.08773)

**Doc2Query--（见 Q2）**
- 生成器自身的概率不能识别坏扩展，需要独立的相关性模型。这对用 VLM 自身 logprob 当后验有警示意义 — [Doc2Query--](https://arxiv.org/pdf/2301.03266)

### Inferences
- 对 7B/8B 的 reader（Qwen2.5-VL-7B、Qwen3-VL-8B、LLaVA-OV-7B），Everson 的 BLOOMZ 结果是最接近的负面先验：把候选平铺进 prompt 可能有害。即使对 GPT-3.5 级别的模型，收益也只有 +0.4 到 +1.7 个点，而且只出现在 WER 约 20%（适中）、有示例的条件下。我们的 ambiguous 桶是"caption 层面互相矛盾"，不确定性比 WCN 平均每词 1.1 个候选大得多，更接近 Everson 说的高噪声区，即 1-best 反而更好的区域。
- 因此，如果要"保留不确定性"，更可行的形式是在 reader 之外做聚合或选项打分（Q3 方案 2、3），只给 reader 一个经过后验加权的结论，或者给出带显式频率的极短候选。PSPL 和 Doc2Query-- 都指向"剪枝和过滤后少而精"，而不是"全部保留"。
- HyPoradise 的"补出列表外 token"，在语音中依赖强语言先验（句子语法约束很强）。对视觉属性（衬衫颜色）没有这种先验，等价于猜测；这与我们观察到的查询时 re-look 编造证据一致。
- 总体评估：这一领域给出的是可靠的机制解释（oracle 覆盖决定上限、后验过滤必要、reader 容量决定能否利用候选、生成器自评不可靠），以及一个便宜的诊断实验（K 次采样的 union-oracle）。它没有给出能直接解决 absent 桶的训练无关方法；absent 桶在语音里的对应问题（OOV）最终只能靠回到原始信号解决。

### Gaps
- 没有找到把多候选或不确定性文本交给 7B 级 VLM 做长时序多选 QA 的研究。
- Everson 只有两个数据集、WER 约 8–20%，没有高 WER 区的完整曲线。
- HyPoradise 中 ICL 设置的具体 WER 数字与模型规模没有抽取。
