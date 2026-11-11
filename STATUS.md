# ReadPast 当前状态（一页纸；改动方法、口径或决定时同步更新）

最后更新：2026-11-11。历史与每轮细节见 `EXPERIMENTS.md`（按时间追加的日志），本页只写“现在是什么”。

## 文件在哪
| 要找 | 位置 |
|---|---|
| 当前所有数字（主表、官方提示、逐子任务、门的消融） | `RESULTS.md`（自动生成）、`results/main_tables.json`；重算：`python3 kfs/make_tables.py`（宿主机 python，几秒；锚点不一致会报错退出） |
| 主表用的是哪些 run | `kfs/data/canonical_runs.json`（74 个 run ID；`make_tables.py` 只读这里） |
| 环境（容器、venv 包版本、权重 revision、数据 hash） | `envs/locks/README.md` + `envs/locks/<venv>.txt` |
| 每轮的计划、判据、结果 | `EXPERIMENTS.md` |
| 相关工作、对手的论文值、代码可得性 | `RELATED.md`（代码状态表在末尾） |
| 论文 | `paper/draft.md`（md 草稿 v0）、`paper/figs/`（图 1–3 脚本）；`paper/main.tex` + `sections/` 仍是 IEEEtran（TCSVT）框架，需改成 ICME 2027 的 IEEE 会议模板 |
| 调研报告 | `reports/` |
| 原始 run | `/data/REN/runs/streamkv/<yymmdd-HHMMSS-name>/out/`（rx 管理） |

## 投稿
**IEEE ICME 2027**（厦门，2027-07-13–17），论文截止 **2026-12-30**（来自 IEEE SPS 会议列表；时区和页数上限官网 2027.ieeeicme.org 尚未核实，历届约 6 页）。11-11 由 ACL 2027 改投；ARR 1 月一轮（01-04）与 ICME 审稿期重叠，不能双投，不再投 ARR。（更早：11-05 由 TCSVT 改投 ACL。）

## 方法（ReadPast = "see the present, read the past"）
1. **写**：同一骨干每 16 秒对 8 帧写一段字幕（`max_new_tokens=96`），按时间顺序组成文本时间线，放在当前画面之前。
2. **看**：SimpleStream 的近 4 帧（r4）不变。
3. **门 (a)**：只看问题文字的 past/present 判别（`kfs/route.py` 提示 P0），读时间线当且仅当 p_past ≥ p_cf；p_cf = 空输入 {"N/A", "", "[MASK]"} 上 p_past 的平均（免标签、每个骨干算一次）。
4. 作答提示不属于方法：所有我们跑的方法共用“条件弃答指令”（D-HSM 原文，只在选项含弃答选项时加；论文实验设置里一句话 + 附录原文，不消融）。

## 评测口径
- OVO-Bench 1468 题（后向 631 + 实时 837），StreamingBench 实时 2500 题。不评 OVO Forward（第二十一轮只做过校准小实验）。
- 主表列：SB Real-Time｜OVO Real-Time｜OVO Backward（含 / 不含 HLD）｜OVO Avg（含 / 不含 HLD）。标准 Avg = (RT+BT)/2；不含 HLD = (RT + (EPM+ASI)/2)/2。
- 主表基线 = 同一提示的 SimpleStream；官方提示的 SimpleStream 作参照行。只报点数涨幅，正文不需要显著性。
- 骨干：Qwen2.5-VL-7B、Qwen3-VL-8B（开发用）；Qwen3-VL-4B（留出验证）；LLaVA-OV-7B（只报告）。
- 对比方法只选有公开代码的：D-HSM 用公开代码同协议重跑（它默认每题加弃答指令）；ShallowStream 用论文值（REN 11-03：不重跑）。FOLIO、WRWS、StreamScout、FlashBack 等无代码，不进表。

## 现在的主结果（共用提示，相对同一提示 SimpleStream；完整见 `RESULTS.md` 表 1）
| | Qwen2.5 | Qwen3-8B | Qwen3-4B | LLaVA-OV |
|---|---|---|---|---|
| OVO Avg | 70.78 (+1.91) | 73.34 (+1.47) | 72.60 (+1.12) | 61.60 (+1.28) |
| OVO Avg 不含 HLD | 68.44 (+0.82) | 71.19 (+2.94) | 69.36 (+1.30) | 65.95 (+1.01) |
| SB Real-Time | 79.56 (−0.20) | 81.64 (+1.36) | 81.40 (+1.48) | 74.92 (+0.28) |

已知弱点：Qwen2.5 不含 HLD 只 +0.82；SB 低于 D-HSM 重跑（Qwen2.5 −0.95、Qwen3 −0.27）；Qwen3 不含 HLD 低于 ShallowStream 论文值（71.19 对 71.65）；准确率上门 ≈ 总是读，门写成“只读 30–56%、RT 不掉”而不是涨分；LLaVA RT −0.67。

## 运行环境
- rx 提交：`rx submit -n NAME -c ren_stream_all -m GB -- bash -c 'source envs/<venv>/bin/activate && python kfs/...'`；rx `allowed_gpus` = [0, 1]（11-12 REN 指定；之前是 [2, 3]），未经 REN 同意不改。
- venv（`envs/`）：simplestream-qwen25、simplestream-qwen3（含 Conv3d 修补）、dhsm、hermes-*。
- 数据：`/data/REN/datasets/streaming/{ovo_bench,streamingbench}`（SB 只有实时部分）。字幕缓存：`cache/captions_{q25,q3,q4,lov}_s16[_sb]`。
- run 名对应（`kfs/analyze_round9.py` 的 `BACKBONES`）：近 4 帧 `r4_ss` / `r4_q3` / `r4_q4` / `r4_lov`；时间线 `cap16[_q3|_q4|_lov]`；共用提示的 187 题 `r4_cinstr*` / `cap16_cinstr*`；门 `route_*cal_p0`（OVO）、`route_*cal_sb_p0`（SB）；SB 全流程 `sb_casc*`；D-HSM `dhsm_mem_abstain*`、`dhsm_sb_*`。
- git 提交需 `git -c user.name=REN -c user.email=ren@lmd.ist.hokudai.ac.jp commit`。

## 管理规范（11-06 起）
1. **正式 run 登记**：主表的输入只认 `kfs/data/canonical_runs.json`。换 run（如修字幕截断后重跑）= 改清单 + 在下面决定记录写一行；`make_tables.py` 发现同名更新的 run 会打印 WARNING，不会自动换。
2. **数字只由脚本生成**：新的正式数字加进 `kfs/make_tables.py`（必要时加锚点），不手抄进 STATUS / 论文；论文表格待换 ACL 模板时由它直接输出 LaTeX。
3. **每轮结束提交一次 git**：提交信息写轮次与 rx tag（如 `Round 22 (tag r22): ...`），这样 run 的 `meta.json` 里记录的 commit 能对上。
4. **环境有变动就重新导出** `envs/locks/`（命令见其 README），并在决定记录写一行。

## 决定记录（新的在下面追加）
- 10-24 方向：免训练、在 SimpleStream 上加插件；只用 OVO + SB 实时，不下载其他数据集。
- 11-02 主指标 = 含 HLD 的标准 OVO 平均（“别人都加，我也加”）；只报点数涨幅。门 (a) 定为主方法组件；时间线放在当前画面之前（第十轮）。
- 11-03 只对比有代码的方法；ShallowStream 不重跑，用论文值；R1（答案端证据检验）移出方法。
- 11-05 条件弃答指令 = 所有方法共用的作答提示，不算方法组件、不消融；主表基线改为同一提示的 SimpleStream。字幕间隔保持 16 秒（8/32 秒不敏感）。改投 ACL 2027。
- 11-06 主表改为含 / 不含 HLD 两种口径并列（仍不写 HLD 伪增益的分析）。Forward 校准（第二十一轮）未过判据，最多写附录。不在 MVBench / MLVU / Video-MME 上评测。实验管理：正式 run 清单、每轮提交 git、环境冻结（见上“管理规范”）。
- 11-11 改投 ICME 2027（截止 2026-12-30），不投 ARR 1 月一轮（不能双投）。补实验的建议（只补写入延迟修正；D-HSM 同一提示重跑、字幕截断修复、MEMO 补跑都不做）待 REN 确认。

## 待办 / 等 REN 决定
- 第二十二轮（写入延迟修正，`EXPERIMENTS.md` 末尾）：方案待 REN 确认；包括主表是否改报修正后的数字。
- ICME 零 GPU 工作：对比表补 OASIS / HERMES / FluxMem / MEMO 论文值（逐项核对原文，二手转引的标注）；12 月中、截止前各查一次无代码论文（尤其 FOLIO）是否放出代码。
- 论文：改 ICME 模板（约 6 页）；`paper/draft.md` 主表换成上面的列格式；引言“三个问题”的故事线、图 2 样式待 REN 确认。
- 字幕连贯性：E0 修截断（96 token 截断率 Qwen2.5 19%、Qwen3-8B 39%、4B 52%、LLaVA 81%）/ E1 离线去重 / E2 锚点+残差，等 REN。
- D-HSM 用同一条件提示重跑：有了不含 HLD 的列后优先级低。
- ReadPast 写完后的下一个项目（训练模块 / agentic 回看）：REN 说之前不启动。
