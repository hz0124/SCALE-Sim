# ARGUS 隔夜跑数结果 (2026-06-13, 50/50 ok)

口径见 `RUN_CHECKLIST.md` §3。利用率 = ΣMAC/(2048×cycles)，分母含空闲子阵列。
w4a16 MAC 系数 = 0.5 pJ（占位，对 figna 能效敏感）。原始数据：本目录 `summary.csv`/`logs/`。

## A. 基础对比（5 硬件 × 模型 × 任务）

加速比/能效比相对 **base**。

| 模型/任务 | 硬件 | 加速比 | 能效比 | util% | EDP |
|---|---|---|---|---|---|
| **bagel/GenEdit** | ours | **1.22×** | **1.86×** | 66.1 | 2.30e9 ✅ |
| | base | 1.00 | 1.00 | 76.3 | 5.22e9 |
| | sdma | 1.01 | 1.01 | 76.4 | 5.10e9 |
| | flightvgm | 1.05 | 1.05 | 75.4 | 4.74e9 |
| | figna | 0.59 | 1.94 | 44.9 | 4.58e9 |
| **bagel/GenEval** | ours | **1.08×** | **1.56×** | 68.1 | 1.29e9 ✅ |
| | base | 1.00 | 1.00 | 72.9 | 2.16e9 |
| | figna | 0.62 | 2.13 | 45.3 | 1.63e9 |
| **bagel/MM** | ours | 1.42× | 1.50× | 0.5 | 1.61e7 |
| | base | 1.00 | 1.00 | 0.4 | 3.44e7 |
| | figna | **2.22×** | **2.63×** | 0.8 | 5.88e6 |
| **janus/GenEval** | ours | **0.80×** ⚠ | 1.30× | 47.7 | 4.43e7 |
| | base | 1.00 | 1.00 | 64.7 | 4.58e7 |
| | figna | 0.64 | 1.93 | 41.7 | 3.69e7 |
| **janus/MM** | ours | **0.78×** ⚠ | 1.10× | 0.3 | 1.83e6 |
| | base | 1.00 | 1.00 | 0.4 | 1.56e6 |
| | figna | 1.38× | 2.14 | 0.5 | 5.31e5 |

（sdma/flightvgm 在 GenEval/MM 与 base 几乎一致，省略；见 summary.csv）

## B. ARGUS 技术点消融（ours，相对 all_off 基线）

| 模型/任务 | 配置 | 加速比 | 能效比 | util% |
|---|---|---|---|---|
| **bagel/GenEdit** | all_off | 1.00 | 1.00 | 42.5 |
| | only_T1 | 1.08 | 1.09 | 42.7 |
| | only_T2 | 1.24 | 1.25 | 40.9 |
| | **only_T3** | **1.59** | 1.38 | **67.6** |
| | noT3(T1+T2) | 1.37 | 1.40 | 41.1 |
| | **all_on** | **2.20** | **1.96** | 66.1 |
| **bagel/GenEval** | all_off→all_on | 1.00→**1.89** | →1.64 | 41.5→68.1 |
| | only_T3 | 1.67 | 1.42 | 69.2 |
| **bagel/MM** | only_T3 = all_on | **1.45** | 1.51 | 0.5 |
| | only_T1/T2/noT3 | 1.00 | 1.00 | 0.4 |
| **janus/GenEval** | all_off→all_on | 1.00→**1.49** | →1.39 | 34.6→47.7 |
| | only_T3 | 1.38 | 1.27 | 47.7 |
| **janus/MM** | only_T3 = all_on | **1.26** | 1.29 | 0.3 |

## C. 关键结论

1. **T3 是单技术最大贡献，且唯一拉升利用率**：bagel GenEdit 单开 T3 = 1.59×（T1=1.08、T2=1.24），
   利用率 42.5%→67.6%；T1/T2 不动利用率（仍 ~41%）。**直接支撑需求 8（T3 提升利用率）**。
   全开 2.20× > 各技术单独之和 → 技术点叠加复合。
2. **MM（纯文本）只有 T3 起作用**：T1（attn 稀疏）/T2（FFN reuse）在纯 decode 无图像阶段不生效；
   只有 T3 的 FFN 权重带宽（BW16）有用 → only_T3 = all_on = 1.45×（bagel）。
3. **⚠ Janus 上 ours 比 base 慢（0.80×/0.78×）**：见 §D，**需讨论**。
4. **figna = "慢但极省"，且 MM 反超**：图像任务 figna 慢（0.59-0.64×，半阵列利用率 ~45%），
   但能效高（1.9-2.6×，全 int4）；MM（纯文本、M=1 带宽瓶颈）figna 的 int4 4× 权重带宽让它
   **最快**（bagel 2.22×、janus 1.38×）。能效比受 w4a16 系数强烈影响（0.5 占位）。

## D. ⚠ 需讨论：Janus 上 ARGUS(ours) 净慢于 base

- 现象：janus GenEval ours 0.80× base，janus MM 0.78× base。消融看技术**有效**
  （all_off→all_on = 1.49× GenEval / 1.26× MM），但 ours 的**硬件基线**（all_off）本身就比
  base 慢（GenEval all_off 1.78e11 vs base 0.95e11 ≈ 慢 1.87×），技术补不回来。
- 根因：Janus 维度小（dim 2048、MHA num_head_kv=16）、GenEval KV 极小（kv_init=48），
  workload 由**投影/FFN 主导、attention 占比极低**。ours 的 qkv/omap 走 FP-FP **半阵列
  （width 32）**，图像阶段 M 大时比 base（width 64）慢 2×；T1（attn 稀疏）几乎无用武之地。
- 对比 Bagel：dim 3584、GQA、KV 大，图像 FFN（ours width 64 = base）+ T1/T2 足以盖过半阵列
  惩罚 → ours 1.08-1.22× 领先。
- 这是**真实的架构权衡显现**，不是 bug：ARGUS 的双阵列在小模型/小 KV 的投影主导 workload 上
  吃亏。可能的处置（待与你讨论）：
  (a) 叙事上 Janus 强调能效（ours 仍 1.10-1.30× 能效）而非延迟；
  (b) 核对 Janus 图像阶段投影是否真该半阵列（若 Janus 权重也量化可用满阵列）；
  (c) 确认 Janus GenEval 的 gen_image_step/分支配置是否合理（影响图像阶段占比）。
