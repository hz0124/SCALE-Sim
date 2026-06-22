# 需求 9：分离式 (disaggregated) 加速器基线

**目的**：造一个反聚合基线——专用文本加速器 (FIGNA 式) + 专用视觉加速器 (S-DMA 式)，两段
**串行**执行，**享 T1+T2 但缺 T3**，用 latency / 利用率 / 能效反证 ARGUS T3（阶段自适应统一阵列）的价值。
**范围**：仅 Bagel GenEdit。**结论先说**：ARGUS 在 latency / 利用率 / 能效三维全面胜过任何固定 split 的 disagg。

---

## 1. 为什么需要扫多档 split

GenEdit 两阶段的纯 MAC 工作量（array 无关）：

| 阶段 | MAC | 占比 | 性质 |
|---|---|---|---|
| 文本 decode | 1.65e12 | **0.155%** | M=1，memory-bound |
| 图像 diffusion | 1.06e15 | **99.845%** | M=1378，compute-bound |

算力比 ≈ **1:643**。一块固定面积的芯片要劈成 text-accel + vision-accel 两份（高 32、宽合计 64 =
2048 lane），无论怎么劈都喂不饱两个相差 643× 的阶段——这本身就是 T3（统一阵列按阶段自适应）的论据。
我们扫 5 档劈分看 trade-off：

| split | W_text | text lanes | W_vision | vision lanes |
|---|---|---|---|---|
| w2 | 2 | 64 | 62 | 1984 |
| w4 | 4 | 128 | 60 | 1920 |
| w8 | 8 | 256 | 56 | 1792 |
| w16 | 16 | 512 | 48 | 1536 |
| w32 | 32 | 1024 | 32 | 1024 |

---

## 2. 建模

`disagg` 是一个**编排器**（非第三套硬件模型）：文本段复用 figna 口径、图像段复用 sdma 口径，各跑自己的
窄/宽阵列，串行相加。**能力给定**：享 T1（稀疏 cross-attn）+ T2（FFN reuse），唯独无 T3 → 自变量只剩 T3。

- **文本段（text-accel，FIGNA 式 W4A16，32×W_text）**：投影/FFN int4 权重 × fp16 激活，注意力 fp16。
  KV、带宽都只用 text-accel 自己那份。
- **图像段（vision-accel，S-DMA 式 fp16，32×W_vision）**：单块 fp16 阵列（无 T3 双子阵列）；
  T1 稀疏 cross-attn 在单阵列上、**不做** T3 的双阵列 /2 负载均衡（那个阵列共享正是 T3）；T2 reuse 照旧。
- **静态功耗（2026-06-17 起：全 hw 统一，非 disagg 专属）**：`array_static_mw × 运行时间` 给
  **所有硬件**计全芯片(2048-lane)漏电+时钟能量底（`coefficients.py` 默认 150mW 占位）。disagg 的
  劣势体现为它**更慢→静态多付**（不再用旧的"仅 idle 半芯片"特例）；ARGUS 最快→静态最少。
  详见 `REBUTTAL_FRAMEWORK.md` §6.5。
- **利用率（4 个，全芯片 2048 口径）**：text/image × compute/mem。
  `compute = stage_MAC/(2048·stage_cycles)`；`mem = (stage_DRAM_bytes/stage_cycles)/PEAK_BPC`。
  分阶段 MAC/字节由能量账本在文本→图像边界 `snapshot()` 取差。

### ⚠️ 文本段周期 = 解析 memory-bound 模型（2026-06-17 关键修正）

文本 decode 是 M=1，物理上 memory-bound（每 token 把整层权重从 DRAM 流一遍、算得极少）。但**旧的
SCALE-Sim OS 仿真把它误建成 compute-bound**：OS 把 M 映射到阵列 32 行 → M=1 只用 1/32 阵列
（util ~3%），被欠利用放大 ~37× 的计算时间盖死了访存，导致**砍带宽完全无效**（3 探针 + os/ws/is 全验证，
stall 恒 0）。根因：M=1 欠利用把"等效计算屋顶"砸到访存屋顶之下，瓶颈成了阵列几何这"第三堵墙"。

修正：文本投影/FFN/注意力改解析 roofline（`bagel_sim.py`/`janus_sim.py` 的 `_text_gemm_cycles`）：

```
cycles = max( 理想填满计算 = M·N·K / text_array_macs ,  操作数流式 = N·K·operand_bytes / BW_bpc )
  text_array_macs = disagg:text-accel lanes / ours:FP-FP 32×32 / 其它:array_h×array_w
  operand_bytes   = 投影/FFN 权重精度(int4=.5/int8=1/fp16=2) ; 注意力 KV(fp16=2)
  BW_bpc          = peak_dram_gbps×1e9/500MHz  (满32GB/s→64; 半带宽组 peak=16→32 B/cyc)
```
效果：文本变 memory-bound（text_mem≈100%）、砍半带宽文本精确 2×、权重精度成主导（int4 快于 fp16）。
> 此修正改了**所有 Bagel+Janus 硬件**的文本周期（含 MM 全部、各任务文本部分），详见
> `REBUTTAL_FRAMEWORK.md` §3.1 路径 C 与 `RERUN_PLAN.md`。

---

## 3. 结果（Bagel GenEdit，新模型）

聚合脚本 `need9_logs/aggregate_splits.py` → `need9_logs/split_sweep.csv`。

**Group A — 满带宽（每 core 活动时独占 32GB/s）**

| 配置 | text lanes | 总周期(e9) | text_cmp% | text_mem% | img_cmp% | img_mem% | 聚合util% | 能量(e6 mJ,含静态+DDR4) |
|---|---|---|---|---|---|---|---|---|
| **ARGUS T3-full** | 统一 2048 | **611.8** | 3.0 | 100 | 88.7 | 6.0 | **85.0** | 2.443 |
| disagg w2  | 64   | 657.3  | 2.9 | 59  | 86.2 | 11.5 | 82.6 | 3.486 |
| disagg w4  | 128  | 663.8  | 4.9 | 100 | 83.8 | 11.1 | 81.8 | 3.495 |
| disagg w8  | 256  | 705.8  | 4.9 | 100 | 78.7 | 10.5 | 76.9 | 3.551 |
| disagg w16 | 512  | 807.4  | 4.9 | 100 | 68.6 | 9.1  | 67.2 | 3.686 |
| disagg w32 | 1024 | 1178.6 | 4.9 | 100 | 46.7 | 6.2  | 46.1 | 2.506 |

**Group B — 半带宽（单管道静态劈成两条 16GB/s 半通道）**：同 5 档，总周期较 A +0.8~2.5%
（w2 657→662 / w4 664→680 / w8 706→722 / w16 807→824 / w32 1179→1195），text_mem 全部 ≈100%。

### 结论
1. **ARGUS 全胜**：611.8e9 最快、util 85% 最高、**能量最低(1.462e6 mJ vs 最佳 disagg w2 的 2.229e6，省 1.52×)**。
   胜势来自图像阶段（T3 双阵列）+ 跑得快→静态能量最少（慢的 disagg 静态多付）。
2. **没有固定 split 能两头兼顾**：util 随 W_text 单调降（文本越宽 → 抢走 vision lane → img_cmp 从 86.2%
   跌到 46.7%）；最接近 ARGUS 的 w2 也差一截。两阶段瓶颈相反（文本 memory-bound、图像 compute-bound）。
3. **带宽劈半真实可见**：文本段 memory-bound，半带宽 → 文本段精确 2×；但文本只占总数 ~4%（图像主导），
   故总数仅 +0.8~2.5%。

---

## 4. 怎么跑

```bash
# 满带宽 5 档（w4=config_disagg.json, w8=config_disagg_b.json, 其余 config_disagg_w{2,16,32}.json）
python run_bagel.py --hw disagg --task GenEdit --config ./topologies/bagel/config_disagg_w2.json --energy
# ... w4/w8/w16/w32 同理；半带宽用 config_disagg_w{2,4,8,16,32}_bwhalf.json
python run_bagel.py --hw ours --task GenEdit --config ./topologies/bagel/config_ours.json --energy   # ARGUS 参照
python devlog_rebuttal/need9_logs/aggregate_splits.py    # 出 Group A/B 双表 + split_sweep.csv
```

---

## 附录：实现文件 & 历史注记

**改动文件**：`bagel_sim.py`（disagg 编排 + 文本 memory-bound 模型 + 4 利用率/DETAIL UTIL 日志）、`accountant.py`（全 hw 静态功耗 `compute_array_static_pJ`）、
`janus_sim.py`（文本模型同步）、`energy_accounting/{coefficients,accountant}.py`（`array_static_mw`/`snapshot`）、
`configs/bagel/disagg_*.cfg`（4 text × {int4,fp16} + 5 vision）、`topologies/bagel/config_disagg*.json`
（满/半带宽各 5）、`run_bagel.py`（`--hw disagg`）。**对非-disagg/满带宽零影响**（默认值守卫 + hardware_type 守卫）。

**历史**：本需求最初（旧 SCALE-Sim OS 文本模型）只跑 w4/w8，结论是"sweet spot 在 w8、util 73/58/49%"。
那套数字已随文本模型修正作废（旧模型把文本误建成 compute-bound）。当前以上表为准。
