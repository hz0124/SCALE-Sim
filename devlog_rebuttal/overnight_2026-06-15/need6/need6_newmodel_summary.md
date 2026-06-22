# Need-6 扫描汇总（新文本 memory-bound 模型 + 最终能量系数，2026-06-18 重跑）

> 模型：BAGEL / GenEdit（T1/T2 是图像阶段技术，纯文本对它们 no-op，不扫）。
> 数据：`need6_technique_{ctx,batch}_sweep_results.csv`、`need6_batch_sweep_results.csv`、
> `need6_ctxlen_sweep_results.csv`（全部新模型 + DDR4 20pJ/bit + idle 500mW + 阵列静态 150mW + per-hw w4a16）。
> §1/§2 是 cycle 比（不随能量系数变）；§3 的能量/req 反映最终能量系数。本文件由
> `gen_need6_summary.py` 从 JSON 重生成。

## 1. 技术点贡献 vs 上下文长度（GenEdit，batch=1）

### 隔离加速比 only_Tx / all_off（cycles）

| Context | only_T1 | only_T2 | only_T3 | all_on |
|---|---|---|---|---|
| 2566 | 1.07 | 1.14 | 1.63 | 2.35 |
| 3211 | 1.09 | 1.15 | 1.61 | 2.39 |
| 7609 | 1.19 | 1.13 | 1.50 | 2.50 |
| 12018 | 1.28 | 1.11 | 1.42 | 2.54 |

### 边际(留一)加速比 noTx / all_on（cycles）—— 越大说明该技术越关键

| Context | noT1→T1 | noT2→T2 | noT3→T3 |
|---|---|---|---|
| 2566 | 1.15 | 1.14 | 1.77 |
| 3211 | 1.19 | 1.15 | 1.75 |
| 7609 | 1.40 | 1.14 | 1.69 |
| 12018 | 1.55 | 1.12 | 1.63 |

## 2. 技术点贡献 vs batch size（GenEdit）

### 隔离加速比 only_Tx / all_off（cycles）

| Batch | only_T1 | only_T2 | only_T3 | all_on |
|---|---|---|---|---|
| 1 | 1.09 | 1.26 | 1.61 | 2.65 |
| 4 | 1.09 | 1.27 | 1.61 | 2.66 |
| 16 | 1.09 | 1.27 | 1.60 | 2.67 |
| 64 | 1.09 | 1.27 | 1.60 | 2.66 |

### 边际(留一)加速比 noTx / all_on（cycles）—— 越大说明该技术越关键

| Batch | noT1→T1 | noT2→T2 | noT3→T3 |
|---|---|---|---|
| 1 | 1.22 | 1.27 | 1.73 |
| 4 | 1.22 | 1.28 | 1.73 |
| 16 | 1.22 | 1.28 | 1.73 |
| 64 | 1.22 | 1.28 | 1.72 |

## 3. Batch 吞吐 / 能效扫描（GenEdit，ours all_on）

> 新模型显式把「权重流式跨 M 摊薄」建进去 → decode 批处理收益由此而来。

| batch | 总周期(e9) | 加速比(vs all_off) | 吞吐(req/s) | 吞吐相对 b=1 | 能量/req(e3 mJ) | 能效相对 b=1 |
|---|---|---|---|---|---|---|
| 1 | 611.9 | 2.65 | 8.17e-04 | 1.00 | 2443.0 | 1.00 |
| 2 | 1201.8 | — | 8.32e-04 | 1.02 | 2316.1 | 1.05 |
| 4 | 2381.8 | 2.66 | 8.40e-04 | 1.03 | 2252.6 | 1.08 |
| 8 | 4741.7 | — | 8.44e-04 | 1.03 | 2220.9 | 1.10 |
| 16 | 9462.9 | 2.67 | 8.45e-04 | 1.03 | 2205.1 | 1.11 |
| 32 | 18925.5 | — | 8.45e-04 | 1.03 | 2198.1 | 1.11 |
| 64 | 37850.9 | 2.66 | 8.45e-04 | 1.03 | 2194.6 | 1.11 |

> 加速比(vs all_off) = cycles(GEdit/all_off) / cycles(ours all_on)，源自 `need6_argus_ablation_results.json`
> 的 GEdit/all_off batch 轴（仅跑了 batch 1/4/16/64，其余标 —）。GenEdit 由 array-saturated 图像阶段主导，
> 故加速比随 batch 几乎恒定 ~2.65×（对比 §4 MM-Vet 纯文本随 batch 跌到 1.0×）。

## 4. MM-Vet（纯文本 MM 任务，Bagel）all_on vs all_off

> §1–§3 是 GenEdit（图生成全流程）；MM-Vet 是纯文本 decode（无图像阶段），T1/T2 对它 no-op，
> all_on 的收益只来自 **T3 精度 + quant_proj**（投影/FFN 走 W4A16，对齐 figna）。数据源
> `need6_argus_ablation_results.json`（MMVet/all_on、MMVet/all_off）。加速比 = cycles_off/cycles_on。

### 加速比 / 能效比 vs 上下文长度（batch=1）

| Context | all_on 周期(e9) | all_on 能量(mJ) | 加速比(off/on) | 能效比(E_off/E_on) |
|---|---|---|---|---|
| 2566 | 15.7 | 183,662 | 3.23 | 3.22 |
| 3211 | 16.7 | 194,526 | 3.10 | 3.09 |
| 7609 | 23.0 | 268,598 | 2.52 | 2.51 |
| 12018 | 29.4 | 342,856 | 2.19 | 2.19 |

### 加速比 / 能效比 vs batch size（base context）

| Batch | all_on 周期(e9) | all_on 能量(mJ) | 加速比(off/on) | 能效比(E_off/E_on) |
|---|---|---|---|---|
| 1 | 16.7 | 195,031 | 3.10 | 3.09 |
| 4 | 31.8 | 372,603 | 2.10 | 2.10 |
| 16 | 103.5 | 1,097,720 | 1.23 | 1.38 |
| 64 | 414.0 | 4,030,010 | 1.00 | 1.11 |

## 注记
- 隔离=只开一个技术 vs 全关；留一=关一个 vs 全开。T1/T2 是图像阶段技术，短上下文工作点贡献小、长上下文放大。
- batch 扫描：投影/FFN 权重跨 batch 复用（stream 摊薄）→ 吞吐随 batch 升；attention 不可批（每请求独立 KV）→ 收益递减。
- 能量/req 含 DRAM 背景(idle 500mW)+阵列静态(150mW)运行时间项；能效相对 b=1 = (E/req @b=1)/(E/req @b)。
- **MM-Vet 随 batch 加速比从 3.10×→1.00×**（GenEdit 几乎不变）：纯文本 decode 小 batch 是 memory-bound
  （W4A16 权重带宽优势显著），大 batch 转 compute-bound → 精度不再省 cycle，all_on/all_off 收敛。
