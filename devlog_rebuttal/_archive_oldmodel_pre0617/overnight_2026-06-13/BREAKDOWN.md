# 算子时间 breakdown (需求3, 2026-06-13)

每个 run 的 log 有 `CYCLE BREAKDOWN` 段（`OPBREAKDOWN <label> <cycles> <pct>`），
长表汇总在 `breakdown.csv`。分桶之和严格 = total_cycles（同 multiplicity 累加）。
6 类算子 × text/img 两阶段。

## A. 基础对比 — bagel/GenEdit 各算子占比（% of end-to-end）

| 硬件 | qkv | attn | omap | ffn_up | ffn_down | text% | img% |
|---|---|---|---|---|---|---|---|
| ours | 15.0 | 10.8 | 11.0 | 34.1 | 29.2 | 14.1 | 85.9 |
| base | 7.3 | 13.5 | 5.0 | 47.6 | 26.6 | 16.3 | 83.7 |
| sdma | 7.4 | 12.5 | 5.1 | 48.1 | 26.9 | 16.5 | 83.5 |
| flightvgm | 7.6 | 14.1 | 5.3 | 45.2 | 27.8 | 17.1 | 82.9 |
| figna | 6.3 | 13.3 | 4.9 | 50.1 | 25.3 | 4.3 | 95.7 |

- **FFN 绝对主导**：所有硬件 ffn_up+ffn_down ≈ 63–75%。
- **ours 的 qkv/omap 占比偏高（15%/11% vs base 7%/5%）**：因为 ours 用半阵列（width 32）算这俩、
  绝对周期不变但 FFN 被 T2/T3 砍小了 → qkv/omap 占比被动抬高。这正是 ours 相对 base 的"短板算子"
  （半阵列、无技术加成）。
- figna img% 95.7%：文本阶段 int4 极快（带宽 4×），图像占比被抬到最高。

## B. 技术点 ↔ 算子归因（消融，bagel/GenEdit 绝对周期 e9）

| config | qkv | attn | omap | ffn_up | ffn_down | TOTAL |
|---|---|---|---|---|---|---|
| all_off | 117.6 | 217.5 | 86.1 | 854.2 | 453.7 | 1729.1 |
| only_T1 | 117.6 | **84.7** | 86.1 | 854.2 | 453.7 | 1596.3 |
| only_T2 | 117.6 | 217.5 | 86.1 | **521.8** | 453.7 | 1396.7 |
| only_T3 | 117.6 | 217.5 | 86.1 | **435.5** | **229.5** | 1086.2 |
| noT3 | 117.6 | 84.7 | 86.1 | 521.8 | 453.7 | 1263.9 |
| all_on | 117.6 | **84.7** | 86.1 | **267.9** | **229.5** | 785.8 |

**干净的算子归因**（每个技术精确砍特定算子）：
- **T1（稀疏 attention）→ 只砍 attn**：217.5 → 84.7（2.6×）。qkv/omap/ffn 完全不动。
- **T2（FFN reuse）→ 只砍 ffn_up**：854.2 → 521.8（M 缩减）。ffn_down 不动
  （已知保守项：reuse 只缩 ffn_up，核对项10）。
- **T3（阵列编排）→ 砍 ffn_up + ffn_down**：width-64 双阵列把两个 FFN 都减半
  （ffn_down 453.7 → 229.5；ffn_up 与 T2 叠加后 521.8 → 267.9）。
- **qkv（117.6）/omap（86.1）跨所有配置恒定** —— 没有任何技术触及（fp16 半阵列投影）。
  这是 ours 的固定成本，也是 Janus 小 KV 下 ours 输给 base 的根源（投影主导时半阵列吃亏）。

→ all_on 把 FFN 从 1308（854+454）压到 498（268+230）、attn 从 218 压到 85，
  qkv/omap 原封不动 → 总 1729 → 786（2.2×）。

## C. 其他 model/task

完整逐算子数据见 `breakdown.csv`（461 行，含全部 50 run × 7 算子）。
- **MM（纯文本）**：无图像阶段，attn 占比极低（M=1 decode），FFN 主导；只 T3 的 FFN 带宽起作用。
- **Janus**：结构同 Bagel，但 dim 小、图像 qkv 占比更高 → 半阵列投影惩罚更明显（呼应 RESULTS.md §D）。
