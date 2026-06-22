# 重跑计划（2026-06-17 文本模型改造后）

> **状态（2026-06-18）：全部重跑完成 ✅。** 权威结果在 `overnight_2026-06-17/`
> （`RESULTS.md` §A–§D + `BREAKDOWN.md`）、need-9 在 `need9_logs/`、need-6 在
> `overnight_2026-06-15/need6/`（新模型）。旧模型矩阵已归档到 `_archive_oldmodel_pre0617/`。
> 本文档保留作「当时怎么重跑的」记录。

## 为什么要重跑
文本 decode 周期模型从 SCALE-Sim OS 改为解析 memory-bound roofline
（`REBUTTAL_FRAMEWORK.md` §3.1 路径 C / `bagel_sim.py`+`janus_sim.py` 的 `_text_gemm_cycles`）。
文本周期缩 ~14×、变 memory-bound、带宽敏感、权重精度成主导。**所有含文本阶段的旧数全部作废**。
能量账本与图像阶段（路径 A/B）不变。

## 数据现状盘点（devlog_rebuttal/）

| 路径 | 内容 | 模型版本 | 处置 |
|---|---|---|---|
| `docs/REBUTTAL_FRAMEWORK.md` `docs/NEED9_DISAGG.md` | 方法学文档 | ✅ 已更新到新模型 | 保留 |
| `need9_logs/split_sweep.csv` `aggregate_splits.py` | need-9 5档split×满/半带宽×4利用率 | ✅ **新模型（今日重跑）** | 保留 |
| `overnight_2026-06-15/{summary,breakdown,RESULTS,BREAKDOWN,logs,*.py}` | 全硬件×任务×ablation 结果 + 旧 driver | ❌ 旧模型 | ✅ 已归档到 `_archive_oldmodel_pre0617/overnight_2026-06-15_oldmatrix/`，由 06-17 的 `run_full/run_supp/run_baseA` 取代 |
| `overnight_2026-06-15/need6/` | need-6 ctx/batch/technique 扫描 | ✅ **新模型（2026-06-17 重跑）** | 保留（脚本路径硬编码故原地） |
| `overnight_2026-06-17/` | §A–§D 大表 + breakdown + sensenova + 全 summary CSV | ✅ **新模型（权威）** | 保留 |
| `_archive_oldmodel_pre0617/overnight_2026-06-13/` | 更早的全套结果 | ❌ 旧模型（隔两代） | 已归档（可删） |
| `_archive_oldmodel_pre0617/need9_logs_oldmodel/` | 旧 need-9 快照日志 | ❌ 旧模型 | 已归档（可删） |

## 重跑任务

### 1. need-9（split 扫描 + 带宽组）— ✅ 已完成
今日已在新模型上重跑全部 11 配置，结果在 `need9_logs/split_sweep.csv`、文档 `NEED9_DISAGG.md` §7。
驱动：`python devlog_rebuttal/need9_logs/aggregate_splits.py`（读各 split 的 `results/bagel/results_disagg_*.log`）。
> 备注：§7 当前只含 disagg(5档)+ARGUS。若论文要 figna/sdma/base 并列，需在新模型上补跑这三者 GenEdit。

### 2. overnight 全矩阵（result + breakdown）— ✅ 已完成
重跑产物在 `overnight_2026-06-17/`：`run_full.py`/`run_supp.py` → `summary_full.csv`/`summary_supp.csv`，
`run_baseA.py` → `summary_baseA.csv`，`compose_RESULTS.py` → `RESULTS.md`（§A–§D），`BREAKDOWN.md` 重生成。
下面是当时的（旧）重跑步骤记录，现已被 06-17 的脚本取代：
矩阵见 `overnight_2026-06-15/manifest.json`：bagel × {ours,figna,flightvgm,sdma,base} × {GenEdit,GenEval,MM}
+ ours 的 ablation（alloff/onlyT1/onlyT2/onlyT3/noT3/allonpure）+ janus × {ours,figna,flightvgm,sdma,base} × {GenEval,MM}。

驱动脚本 `overnight_2026-06-15/driver.py` 串行跑 manifest（共享 layer.csv 不能并行）、解析 log 出
`summary.{csv,json}`；`make_breakdown.py` 出 `breakdown.csv` + 4桶版。脚本调用 `run_bagel.py`/`run_janus.py`，
**自动走新模型，无需改脚本**。

重跑步骤（输出到新目录 `overnight_2026-06-17/`，保留 06-15 作历史对比）：
```bash
source env.sh   # 或 conda activate zhb-scalesim-v3
BASE=/tmp/argus_rerun_0617
mkdir -p $BASE && cp devlog_rebuttal/overnight_2026-06-15/manifest.json $BASE/
# 改 driver.py 顶部 BASE 指向 $BASE（或复制一份改路径），然后：
python devlog_rebuttal/overnight_2026-06-15/driver.py          # → summary.{csv,json}
python devlog_rebuttal/overnight_2026-06-15/make_breakdown.py  # → breakdown.csv / 4bucket
mkdir -p devlog_rebuttal/overnight_2026-06-17
cp $BASE/{summary.*,breakdown*.csv} devlog_rebuttal/overnight_2026-06-17/
# 然后据新数重写 RESULTS.md / BREAKDOWN.md（旧模型结论如 U形/排名会变，见 NEED9 §7 的同类翻转）
```
预期变化：所有 MM 数缩 ~14×；GenEval/GenEdit 文本部分缩、总数小幅降；文本阶段 int4(figna) 反超
ours；ablation 阶梯里 T1/T2（图像技术）相对占比上升（因文本变小）。

### 3. need-6（ctx/batch/technique 扫描）— ✅ 已完成
2026-06-18 在新模型 + 最终能量系数（DDR4/idle500/static150/per-hw w4a16）下**全部重跑**：
ctxlen(9) + batch(5hw×7) + argus_ablation(24) + technique(ctx32+batch32)。ctxlen 的生产脚本已重建
（`run_need6_ctxlen_sweep.py`）。结果 `need6_*_results.{csv,json}`；权威总结 `need6_newmodel_summary.md`
（`gen_need6_summary.py` 生成）；加速比块由 `compose_need6_ablation_md.py` 刷新。脚本均可复用。

## 重跑后要同步更新的文档
- `overnight_2026-06-17/RESULTS.md` `BREAKDOWN.md`（新数+新结论）
- `REBUTTAL_FRAMEWORK.md` §7 需求表状态、§8 待办19（标记已重跑的部分）
- need-6 的 `need6_*.md`（若重跑）
