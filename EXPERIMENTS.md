# streamkv 实验记录（流式视频关键帧选择）

> 本文件记录每组实验的目的、做法和结论，是唯一的权威来源。数字用 run_id 或 `rx table -p streamkv -t <tag>` 引用。
> 自动生成的运行总表在 `/data/REN/runs/streamkv/RUNS.md`。

## 约定

- 课题（2026-10-24 定）：在 SimpleStream（只看最近 N 帧）之上做**免训练、即插即用**的关键帧选择，目标是 2 区期刊。期刊更看重完整性，不要求超过需要训练的 SOTA（如 SelectStream）。
- 数据（2026-10-24 用户确认）：不下载额外数据集，只用本地的 OVO-Bench 和 StreamingBench-RT（StreamingBench 本地只有 Real-Time 部分的视频）。
- 主指标：OVO-Bench 实时（RT）和回溯（BT）两类的任务宏平均，avg = (BT + RT) / 2，和 SimpleStream 论文一致；StreamingBench-RT 为第二个 benchmark。
- 打分：SimpleStream 的 `extract_mcq_answer`（只认 A–D）。
- 默认 backbone：Qwen2.5-VL-7B-Instruct，1 fps，最近 4 帧（r4）。
- 参考数字：SimpleStream 论文 Qwen2.5-VL-7B r4 为 RT 78.4 / BT 51.9 / avg 65.13；我们 10-12 的复现为 77.73 / 51.78 / 64.76（`results/ovo_qwen25vl_recent4`）。
- 噪声：推理是贪心解码，结果确定；比较时看同一批题目上的翻转数（+赢/−输）。
- 环境：容器 `ren_stream`，venv `envs/simplestream-qwen25`，通过 `rx submit -c ren_stream` 提交；代码在本仓库 `kfs/`，SimpleStream 在 `third_party/SimpleStream`（分支 ren/dev）。

---

## 2026-10-24 方案与 pilot（tag: `kfs-pilot`）

**背景**：
- SimpleStream 论文里，用 CLIP 检索 5 帧历史帧拼进输入（V-RAG），EPM +7.1、ASI +6.1，但 HLD −12.4，实时类各任务掉 2~9，总分 −2.3。作者把这叫「感知-记忆权衡」，并建议「近帧优先，历史帧按需调用」。
- 同类免训练工作 WRWS（arXiv 2609.37345）在固定 4 帧预算内按检索熵混合近帧和检索帧，Qwen2.5-VL-7B 上只比 SimpleStream 高 +0.25。

**方法设想**：
1. 在线关键帧库：与问题无关，写入时决定，容量有上限；
2. 提问时选帧：相关度 + 多样性 + 时间覆盖；
3. 按需开关：先只用最近 4 帧作答，选项置信度低时才补历史关键帧；
4. 时间角色提示：给历史帧标时间戳，并标出「当前画面」。

**pilot 要回答的问题**：
1. 补历史帧本身能带来多少回溯类增益，实时类掉多少；
2. 标时间戳（tagged）能不能减轻实时类掉分；
3. 理想开关的上限有多大，置信度开关能拿到多少；
4. 去冗余（MMR）是否有用：冒烟测试里 CLIP top-4 会扎堆，例如 224–230 秒连续 4 帧。

**是否继续的判断标准**：置信度开关下 avg ≥ +1.5，且 RT 掉分 < 0.5、BT ≥ +3，就继续做完整方法；理想开关的上限 < +2 就换思路。

**设置**：代码 `streamkv@443ba29`（`kfs/eval_ovo_kfs.py`）。k=0、ss 布局与 SimpleStream r4 逐题一致（冒烟测试 `261024-011104-smoke_r4`，10 题的答案和视觉 token 数完全相同）。

| run | 配置 |
|---|---|
| r4_ss | 最近 4 帧（基线，记录置信度） |
| r8_ss | 最近 8 帧（同帧数对照） |
| r4_blocks | 最近 4 帧，每帧单独一个视觉块 |
| uni4_ss / uni4_tag | + 均匀 4 帧历史，不标时间 / 标时间 |
| clip4_ss / clip4_tag | + CLIP-L top-4 历史，不标时间 / 标时间 |
| mmr4_tag | + CLIP-MMR(λ=0.5) 4 帧历史，标时间 |

**运行**：`rx table -p streamkv -t kfs-pilot`；分析用 `python3 kfs/analyze_pilot.py --variants ...`

**结论**：（待填）
