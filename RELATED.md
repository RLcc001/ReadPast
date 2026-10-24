# 基于 / 对比 SimpleStream 的相关工作调研（2026-10-24）

> 来源：Semantic Scholar 上引用 SimpleStream（arXiv 2604.02317）的 36 篇论文（截至 2026-10-24），加网页检索。数字从各论文 arXiv HTML 版摘取，未逐一核对 PDF。
> 统一口径：OVO = (RT + BT) / 2，不含 Forward；论文自报的 "overall" 口径不一，下面的数是按 RT、BT 重新算的。
> 参照：SimpleStream r4 论文值 Qwen2.5-VL-7B 65.13（RT 78.4 / BT 51.9），Qwen3-VL-8B 67.70（81.4 / 54.0）；我们复现的 Qwen2.5 为 64.75，HLD 40.3。

## A. 免训练，在近帧窗口之外按需取历史（和我们最接近）

| 方法 | arXiv | 机制 | 骨干 | OVO (RT/BT) | 相对 SimpleStream | HLD |
|---|---|---|---|---|---|---|
| WRWS | 2609.37345 | SigLIP 检索 + 相似度熵决定检索还是近帧，K 帧预算 | Qwen2.5-7B / Qwen3-8B 等 | 63.86 / 65.37 | +0.04 / +0.32（同表重跑的 SimpleStream） | 未给可靠数 |
| ShallowStream | 2609.02780 | 浅层 KV 建索引；纯文本 few-shot 门（A/B logit 差，阈值在独立集上标定一次）；浅层注意力投票 + max-min 多样性选帧 | Qwen3-8B / LLaVA-OV-7B | 69.5 / 62.2 | **+3.1 / +1.9**（作者重跑的 SimpleStream 66.4 / 60.3） | 49.5（SS 44.1） |
| FOLIO | 2607.13298 | 写入端用 VLM 每 8 秒写实体中心的文本记忆 + 关键帧证据缓存 + 近帧缓冲；查询时混合检索 | Qwen2.5-7B / Qwen3-8B | 67.1（77.0/57.2）/ 75.6（82.0/69.1） | 未直接比较；按论文值约 +2.0 / +7.9 | 66.7 / 84.6 |
| StreamScout（免训练版） | 2609.00291 | 文本时间线 + 逐级升级：近 4 帧 → 均匀回看 8 帧 → 检索 12 帧，模型自己决定是否升级 | Qwen2.5-7B / Qwen3-8B / LLaVA-OV-2 | 62.7 / 68.4 | Qwen2.5 低于 SS；Qwen3 比自身只看近帧 +3.7 | 36.6 / 63.4 |
| FlashBack | 2610.01192 | 冻结 VLM 先判断要不要历史（Native / Recall），Recall 走临时 Side-KV | StreamingVLM-7B / Mage-VL-4B | 57.9 / 65.6 | 相对自身 −1.2 / +0.75 | 下降 |
| SAVEMem | 2605.07897 | 三级记忆 + 伪问题库定显著性 + 近帧锚点门控 | Qwen2.5-7B | 62.7 | 低于 SS（且门控阈值按 RT/BT 类别分别设，用了类别标签） | 37.6 |
| LatentStream | 2609.04131 | 分层记忆 + 测试时用熵奖励优化潜在记忆 token（不改权重） | Qwen2.5-7B | 64.3 | 略低于 SS | 68.3 |
| ReMem | 2610.05940 | 注意力压缩上下文 + 检索 1 个片段 | Qwen2.5-7B / Qwen3-8B | 51.4 / 60.0 | 远低于 SS | 29.6 / 43.4 |
| ObjectStream | 2607.28312 | 从 VLM 特征诱导潜在物体并跨帧追踪作为记忆锚点 | Qwen2.5-7B | 61.4 | 低于 SS | 未报 |
| DeltaStreamer | 2610.19849 | 智能体：记录帧间变化 + 规划 + 回放历史帧（SAM2/Whisper 工具） | Qwen3.6-27B | 70.0 | 未同骨干比较 | 69.9 |
| FrameBlend | 2610.29229 | 查询时选择性重算 KV（ReKV/HERMES/LiveVLM 上的插件） | LLaVA-OV-7B / Qwen2.5-7B | 58.0（HERMES+） | 未在 SS 上测 | 未报 |

## B. 需要训练

| 方法 | arXiv | 训练什么 | 骨干 | OVO (RT/BT) | 相对 SimpleStream | HLD |
|---|---|---|---|---|---|---|
| SelectStream | 2606.16353 | 冻结 VLM，训练潜在记忆图（写入/整合/检索模块） | Qwen2.5-7B / Qwen3-8B | 70.95 / 72.48 | +5.8 / +4.8 | 未报 |
| SPARE | 2610.14671 | 训练"近帧够不够"的门 + 检索 + LoRA 学会弃答 | Qwen3-8B | 77.78 | +10.2 | **90.3**（SS 52.7） |
| StreamScout-S / -R | 2609.00291 | LoRA 蒸馏升级策略 / 再 GRPO | Qwen2.5-7B / Qwen3-8B | 71.6 / 76.2（-R） | 大幅领先 | 89–98 |
| StreamTTT | 2608.13416 | 每层加测试时训练快权重分支存长程历史 | Qwen3-VL-4B | 69.0 | +2.9（SS-4B 66.06） | 60.8 |
| StreamOPD | 2608.16320 | 只做后训练（on-policy 蒸馏），推理仍是近 4 帧 | Qwen3.5-4B | 69.3（ST-CueGate） | 相对自身近帧 +9.6 | 45.7（主指标排除 HLD） |
| MaLoW | 2610.22070 | 超网络把历史写成 LoRA 权重 | Qwen3-VL-4B 等 | 69.3 | 同骨干表中优于近帧窗口 | 61.8 |
| OneStreamer | 2610.01762 | SFT 学生成带时间的字幕记忆 + 近 16 帧 | Qwen3-VL-4B | RT 81.4 | 优于自身近 16 帧 | 99.5 |
| SVMemAgent | 2609.18540 | RL 学与问题无关的在线选帧 | — | — | — | — |

## C. 效率方向（不以精度为主）

CodecSight（2604.06036，编解码信号裁剪 patch + KV 刷新）、FrameMogging（2605.03351，避免重复计算）、DeltaS（2609.27470，混合架构里用线性注意力状态变化做 KV 淘汰）、Mage-VL（2607.24904，codec-native 流式基础模型）。

## D. 新 benchmark（引用 SimpleStream 作基线）

OmniInteract、OVO-S-Bench、Streaming Interventions、StreamArena、APM-Bench、FastBench、EgoProbe、RoboChrono、VSI-Super-Wild。

## 对我们的启示

1. **免训练"检索原始帧 + 门控"这条路已经很挤，而且在 Qwen2.5-VL-7B 上收益接近零**：WRWS +0.04，我们 +0.5（交叉验证）。在 Qwen2.5 上，SAVEMem、LatentStream、StreamScout、ObjectStream、ReMem 都低于 SimpleStream。
2. **免训练里明确超过 SimpleStream 的只有两类做法**：
   - **ShallowStream**：用模型内部信号检索（浅层注意力），门控只看问题文本、阈值离线标定一次，只在 Qwen3-8B / LLaVA-OV 上报告。
   - **FOLIO / StreamScout**：历史以**文本记忆**（实体记录、字幕时间线）形式提供，视觉只保留近帧。文本记忆不稀释对当前画面的感知（OneStreamer 的消融也是这个结论）。FOLIO 是唯一在 Qwen2.5-7B 上明显高于 SimpleStream 的免训练方法（RT/BT 均值约 67.1），但没有和 SimpleStream 直接比较。
3. **"何时取历史"的门控是现在最热的点**：WRWS、ShallowStream、FlashBack、SAVEMem、StreamScout、SPARE 都在做。我们的创新点 2（置信度仲裁）和它们重叠很大。
4. **HLD 是公认的难点**：BT 涨得多的方法主要涨在 HLD，靠训练弃答（SPARE 90、OneStreamer 99.5、StreamScout-R 98），或靠结构化记忆判断"出现过没有"（FOLIO 66.7 / 84.6）。StreamOPD 的主指标直接排除 HLD，并说明 SimpleStream 也把 HLD 单独报告。
5. **低分辨率历史帧**（我们的创新点 1）在这批工作里没人做，但单独效果小（4 帧时从 −2.9 改善到 −1.7）。
