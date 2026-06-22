# devlog_rebuttal — dingli 的开发日志与实验产物

本目录整个被 `.gitignore` 忽略，属于本地草稿/实验区，不进版本库。
**给接手人**：先读本文件 → `PROJECT_OVERVIEW.md`（项目全貌）→ `DEVLOG.md`（按时间倒序的开发日志）。
所有「权威」结果都在 `overnight_2026-06-17/`；带 `_archive` 前缀的目录是旧模型快照，可忽略/删除。

---

## ⚠️ 一条最重要的背景

2026-06-17 文本 decode 周期模型从 SCALE-Sim 的 OS 数据流改成**解析 memory-bound roofline**
（`bagel_sim.py`/`janus_sim.py` 的 `_text_gemm_cycles`，详见 `docs/REBUTTAL_FRAMEWORK.md` §3.1 路径 C）。
**这条线把所有「旧模型」数据作废了**——凡是 06-17 之前跑的、含文本阶段的结果都不能用。
同日还把能量系数改成 **DDR4 20 pJ/bit + DRAM 背景 500mW + 计算阵列静态 150mW + per-hw w4a16**
（ours 0.47 / figna·disagg 0.57 / axcore 0.50）。所以判断一份数据能不能用，先看它是不是新模型。

---

## 顶层导航文档

- `PROJECT_OVERVIEW.md` — 项目全貌（架构、运行方式、两级配置、能耗模型、当前状态）。一次性看懂用。
- `DEVLOG.md` — 开发日志，倒序，最新在最上面。
- `docs/REBUTTAL_FRAMEWORK.md` — **方法学权威文档**：每种硬件怎么建模、能量系数、各 rebuttal 需求的口径。
  改了建模就更新它。`.pdf` 是渲染版（用 `tools/md2pdf.py` 生成）。

---

## 当前权威结果：`overnight_2026-06-17/`

新模型 + 最新系数下的全部对外结果都在这里。

- `RESULTS.md` — **主结果**。⚠️ `compose_RESULTS.py` 只重生成 **§A/§B/§C**（整文件覆盖），**§D 和 §S 是手工追加的**——重跑 `compose_RESULTS.py` 会把它们冲掉，记得重新贴回。各段来源：
  - §A0 SA/HCA/ARGUS 摘要、§A1 加速比/能效比（含 Geomean）、§A2 功率/功率比 — `compose_RESULTS.py` 读 `summary_baseA.csv`
  - §B 技术点 ablation（加速比+能效比）— `compose_RESULTS.py` 读 `summary_full.csv` + `summary_supp.csv`
  - §C 阶段×算子时间分解（HCA 归一）— `compose_RESULTS.py` 读 `/tmp/timetable.json`。⚠️ **生产该 json 的脚本未进仓库**（曾在临时 worktree，已丢，同 need6 ctxlen 的情况）；要重刷 §C 需重建一个生成 `/tmp/timetable.json` 的脚本（按阶段×算子分解 cycle，HCA 归一）。
  - §D ARGUS 能量分解（Compute/SRAM/DRAM/Static）— **手工维护**，无生成脚本；数字取自 `--energy` 跑出的 `ENERGY BREAKDOWN` 段。
  - §S SenseNova（支线）— 手工添加，权威副本见 `SENSENOVA.md`。
- `BREAKDOWN.md` — 算子级 cycle 占比（Bagel，新模型）。
- `SENSENOVA.md` — 支线模型 SenseNova 的**权威结果副本**（未接进 compose 流水线；RESULTS.md §S 是它的手工摘录）。
- 运行脚本（都从仓库根目录跑，串行，共享 `topologies/.../layer.csv` 不能并行）：
  - `run_base.py` → `summary_base.csv`/`.json`：§A 的 ARGUS(all_on) + 4 个 DSA 基线（figna/flightvgm/sdma）原始跑。`run_full.py` 的 ablation 在它的 all_on 行基础上展开。**这是 HCA 归一前的早期 §A 跑**，HCA-归一表用下面的 `run_baseA.py`。
  - `run_baseA.py` → `summary_baseA.csv`：§A 大表（HCA 归一，6 硬件 × 5 任务，base 不进表）。compose_RESULTS 的 §A 数据源。
  - `run_full.py` / `run_supp.py` → `summary_full.csv` / `summary_supp.csv`：全矩阵 + ablation 补充（§B 数据源）。
  - `compose_RESULTS.py` → 重生成 `RESULTS.md` 的 §A/§B/§C（见上方警告）。
- `genA/`、`gen_configs/` — 上面脚本临时生成的 per-run JSON + log（HCA/AxCore/SA/ablation 变体），可删可重生。

> §A 大表的旧版散件 `compose_baseA.py`（→ `baseA_table.csv` + `RESULTS_baseA.md`）是中间产物，已被 `compose_RESULTS.py` 取代，留作参考（`RESULTS_baseA.md` 已不在目录里，重跑 compose_baseA 会再生成）。

---

## 专题目录

### `img_attn_array_compare/` — 图像阶段 attention：ARGUS vs 统一 32×64 FP16/FP8/FP4 阵列
- `compare_img_attn.py` — **in-process** 跑 Bagel GenEdit（不污染 `./results`），只取图像 diffusion 阶段
  attention 算子（`img/attn` = attn_qk + attn_sfmxv），对比 ARGUS(32 半宽+T1/T3 稀疏+fp16) 与统一 64 宽
  全稠密阵列在三种精度下的计算时间/动态能耗。用法：`python devlog_rebuttal/img_attn_array_compare/compare_img_attn.py`。
- `IMG_ATTN_ARRAY_COMPARE.md` — 结果表 + 解读（新模型口径；只含动态能耗 MAC+SRAM+DRAM，不含 idle/阵列静态）。
  结论：attention 上 ARGUS 优势主要来自稀疏（少算少搬）+ 速度，非精度。**独立支线分析，不在 RESULTS.md 里。**

### `need9_logs/` — Need-9 分离式（disaggregated）基线
- `aggregate_splits.py` — 读各 split 的 `results/bagel/results_disagg_*.log`，出 Group A/B 双表 + `split_sweep.csv`。
- 文档在 `docs/NEED9_DISAGG.md`（新模型，§7 为准；§4 标记过时）。

### `overnight_2026-06-15/need6/` — Need-6 上下文长度 / batch / 技术点扫描
> 该 06-15 目录现在**只剩 need6**（其余旧模型矩阵已归档到 `_archive_oldmodel_pre0617/`）。
> need6 脚本把输出路径硬编码成 `overnight_2026-06-15/need6/`，故原地保留。
- `run_need6_batch_sweep.py` — batch-size 扫描（Bagel GenEdit，全硬件）。
- `run_need6_argus_ablation.py` / `run_argus_technique_sweep.py` — ARGUS 技术点 × ctx/batch 扫描。
- `run_need6_ctxlen_sweep.py` — 上下文长度扫描（ours all_on，9 点；2026-06-18 重建，原脚本曾在临时 worktree）。
- `gen_need6_summary.py` — 从 fresh JSON 重生成 `need6_newmodel_summary.md`；`compose_need6_ablation_md.py` 刷新 `need6_context_length_sweep.md` 的加速比块。
- `need6_*_results.{csv,json}` — 各扫描结果（脚本产物，**2026-06-18 全部重跑、新模型 + 最终能量系数**）。
- 分析文档（两份，均 2026-06-18 全新模型）：`need6_newmodel_summary.md`（**权威condensed总结：§1-3 GenEdit 技术点 + §4 MM-Vet，看这个**）、`need6_context_length_sweep.md`（methodology + profiling 参数 + per-hw/per-ctx 明细表 + ablation 块）。

---

## `docs/` — 文档与参考
- `REBUTTAL_FRAMEWORK.md` / `.pdf` — 方法学权威文档（见上）。
- `NEED9_DISAGG.md` — Need-9 分离式基线分析。
- `RERUN_PLAN.md` — 2026-06-17 文本模型改造后的重跑计划（任务状态见文档内，现已全部完成）。
- `micro2026-paper2908.pdf` — 投稿论文（参考用）。

## `tools/` — 辅助工具
- `md2pdf.py` — markdown → GitHub 风格 PDF。用法：`python devlog_rebuttal/tools/md2pdf.py <src.md> [dst.pdf]`。
- `gh-pdf.css` — md2pdf 的样式表。

## `_archive_oldmodel_pre0617/` — 旧模型归档（可删）
2026-06-17 文本模型改造前的所有结果，已作废，仅留作历史对比：
- `overnight_2026-06-13/` — 更早一代的全套结果。
- `overnight_2026-06-15_oldmatrix/` — 06-15 那批旧模型全矩阵（RESULTS/BREAKDOWN/summary/logs/driver 脚本）。
  其驱动脚本 `driver.py`/`make_breakdown.py` 已被 06-17 的 `run_full.py`/`run_supp.py` 取代。
- `need6_logs_oldmodel/`、`need9_logs_oldmodel/` — 旧 need-6/need-9 日志快照。
