# 算子级 breakdown（Bagel，新模型，% of end-to-end）

> 每个算子周期占端到端总周期的百分比（取 log 最后一块 CYCLE BREAKDOWN）。

## Bagel GenEdit — 各硬件算子占比(%)

| 算子 | ours | figna | flightvgm | sdma |
|---|---|---|---|---|
| text/qkv | 0.14 | 0.01 | 0.01 | 0.01 |
| text/attn | 0.82 | 0.32 | 0.66 | 0.63 |
| text/omap | 0.10 | 0.01 | 0.01 | 0.01 |
| text/ffn_up | 2.23 | 0.05 | 0.06 | 0.05 |
| text/ffn_down | 1.11 | 0.03 | 0.03 | 0.03 |
| img/qkv | 8.30 | 6.26 | 7.38 | 7.08 |
| img/attn | 9.83 | 12.29 | 13.90 | 12.01 |
| img/omap | 6.22 | 4.81 | 4.96 | 4.76 |
| img/ffn_up | 38.36 | 50.82 | 46.82 | 50.29 |
| img/ffn_down | 32.88 | 25.41 | 26.19 | 25.14 |

## ours — 跨任务算子占比(%)

| 算子 | GenEdit | GenEval | MM |
|---|---|---|---|
| text/qkv | 0.14 | 0.18 | 1.60 |
| text/attn | 0.82 | 0.08 | 77.44 |
| text/omap | 0.10 | 0.14 | 1.24 |
| text/ffn_up | 2.23 | 2.95 | 13.14 |
| text/ffn_down | 1.11 | 1.48 | 6.57 |
| img/qkv | 8.30 | 7.34 | 0.00 |
| img/attn | 9.83 | 6.32 | 0.00 |
| img/omap | 6.22 | 5.50 | 0.00 |
| img/ffn_up | 38.36 | 46.93 | 0.00 |
| img/ffn_down | 32.88 | 29.08 | 0.00 |

## 注记
- text/* 现在占比很小（新 memory-bound 模型下文本阶段缩 ~14×）；端到端由图像阶段 img/ffn_up+ffn_down 主导。
- MM 为纯文本任务（无 img/*）。