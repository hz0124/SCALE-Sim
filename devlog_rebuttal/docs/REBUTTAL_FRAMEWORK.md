# ARGUS 评估框架方法论（Rebuttal 讨论文档）
---

## 1. 框架总览

整个评估栈分三层：

```
┌─────────────────────────────────────────────────────────────┐
│  MLLM Decode Harness（本仓库新增, simulation_core/）          │
│  按"模型结构 × 任务 × 硬件"组合出 decode 全过程的算子序列，    │
│  对每个代表性算子调用下层仿真一次，再解析放大成端到端周期/能量  │
├─────────────────────────────────────────────────────────────┤
│  单 GEMM 评估器：SCALE-Sim v3 逐周期仿真（图像文本侧投影）+   │
│  解析 roofline（文本 decode，§3.1 路径 C）+ 解析折叠（图像）  │
├─────────────────────────────────────────────────────────────┤
│  能量账本 EnergyAccountant（本仓库新增, Phase F）             │
│  MAC/SRAM/DRAM 访问次数 × pJ 系数（RTL/Accelergy 系数可换）   │
└─────────────────────────────────────────────────────────────┘
```

核心设计决策：**不对整个 MLLM 做 cycle-accurate 仿真（单次 decode 含数万个大 GEMM，不可行），
只精确仿真少数代表性单层 GEMM，再用结构对称性（层间相同、token 间相似、head 间相同）把结果
解析放大。** 所有硬件（ours 与全部 baseline）走完全相同的放大路径，放大误差在对比中系统性
抵消——我们报告的是**相对加速比/能效比**，不是绝对时延的标定值。

一次运行的流程（以 Bagel 为例，Janus 同构）：

```
run_bagel.py --hw <硬件> --task <任务> --config <workload JSON>
  → 读 JSON：任务参数 + 该硬件的 cfg 指向与算法参数
  → 文本 decode 阶段：各算子用解析 roofline（memory-bound aware，§3.1 路径 C）算一次复用全程；attention 按窗口抽样
  → 图像生成阶段（生成类任务）：构造一个代表扩散步，按 CFG 分支 × 步数放大
  → 输出端到端周期（500 MHz 换算秒）＋能量报告
```

---

## 2. 模型与任务抽象（workload 怎么建）

### 2.1 模型

decoder 主干抽象为一组结构参数：

| 参数 | Bagel | Janus | SenseNova（待做，需求 2） | 含义 |
|---|---|---|---|---|
| `num_layer` | 28 | 24 | 待算法侧提供 | decoder 层数 |
| `dim` | 3584 | 2048 | 〃 | hidden dim |
| `head_dim` | 128 | 128 | 〃 | 每 head 维度 |
| `num_head_q` | 28 | 16 | 〃 | Q heads |
| `num_head_kv` | 4（GQA） | 16（MHA） | 〃 | KV heads |
| `upshape` | 18944 | 5632 | 〃 | FFN 中间维度 |

每层 decode 拆成 6 类 GEMM 算子（也是 breakdown 的分桶粒度）：

1. **qkv** — Q/K/V 投影：`(M × dim) · (dim × (dim + 2·n_kv·head_dim))`
2. **attn_qk** — Q·Kᵀ：每 head `(M × head_dim) · (head_dim × kv_len)`
3. **attn_sfmxv** — Softmax(·)·V：每 head `(M × kv_len) · (kv_len × head_dim)`
4. **omap** — 输出投影：`(M × dim) · (dim × dim)`
5. **ffn_up** — up/gate 投影：`(M × dim) · (dim × upshape)`（gate+up 计 2 次）
6. **ffn_down** — down 投影：`(M × upshape) · (upshape × dim)`

softmax/LayerNorm/激活等非 GEMM 算子不建模（统一口径，见 §6 与 §8 讨论项 2）。

### 2.2 任务参数

任务由 workload JSON 描述，取值来自真实模型 trace / 数据集统计：

| 参数 | 含义 | 来源 |
|---|---|---|
| `kv_cache_init` | prefill 结束时 KV cache 长度（GenEdit=3254） | 数据集平均 prompt 长度 |
| `image_len` / `text_len` | prompt 中图像/文本 token 数 | 同上 |
| `gen_text_len` | 文本 decode 步数（229） | 数据集平均输出长度 |
| `gen_image_step` | 扩散步数（50） | 模型推理配置 |

### 2.3 推理流程（任务 × 模型）

**文本 decode 阶段（所有任务相同）**：prefill 不仿真（`kv_cache_init` 作初始条件），
逐 token decode（M=1），KV 每步 +1，attention 按 `sample_rate=20` 分组抽样。

**图像生成阶段**：每个扩散步 = 对图像 token 做一次完整 decoder 前向，步间 KV 不增长，
故只建一个代表步 × 步数；CFG（classifier-free guidance）的每个分支是一组独立 GEMM：

| 任务 | 模型 | CFG 分支 | 每分支的 M | 每分支的 KV |
|---|---|---|---|---|
| MM（理解） | Bagel / Janus | 无图像阶段 | — | — |
| GenEval（文生图） | Bagel | full_cache / without_text | 1378 | 全 KV / 扣 text_len |
| GenEval（文生图） | Janus | full_cache / without_text | 全序列 / 仅图像 token | 全 KV / 仅图像 token |
| GenEdit（编辑） | Bagel | full_cache / without_img / without_text | 1378 | 全 KV / 扣 image_len / 扣 text_len |

Janus 不支持 GenEdit（与论文一致）。Bagel 图像 attention 双向、Janus 因果；harness 不区分
mask 形状，均按给定 kv_len 的稠密 GEMM 建模（声明为近似）。

**扩展维度（需求 6，机制已支持/待做）**：更长上下文 = 纯 JSON 参数扫描（已支持，等数据）；
batch size = 文本 decode 的 M 从 1 改 batch（待做，小改动：投影/FFN 权重摊销，attention
跨请求不可 batch、周期 × batch，能量 multiplicity 同步乘）。

---

## 3. 周期（latency）建模

### 3.1 三步法：分解 → 仿真 → 放大

**第一步（分解）**：利用结构对称性，每个"形状唯一"的 GEMM 只评估一次：
28/24 层相同 → ×`num_layer`；各 Q-head 相同 → ×`num_head_q`；
kv 无关算子（qkv/omap/ffn）全程形状不变 → 仿真一次复用。

**第二步（仿真/解析）**，三条路径：

- **路径 A：SCALE-Sim cycle-accurate 仿真**。单层 MNK topology，跑 OS dataflow 完整仿真
  （阵列折叠、SRAM 双缓冲、DRAM stall）。*tile 缩放*：N 大的 GEMM 只仿真 N=64 列条带，
  总周期 = 条带 ×(N/64)，OS 下严格线性（实测误差 0.73%）。**2026-06-17 起仅用于图像阶段
  文本侧投影**（`qkv_q_strip`/`qkv_kv_strip`，27 分钟→15 秒的 tile 放大，2026-06-12）；
  文本 decode 已迁出（见路径 C）。注意 `verbose=False` 会让 `run_scale` 返回 0 周期，
  故 `run_sim_once` 保持 `verbose=True`。
- **路径 B：解析折叠模型（compute-bound）**。图像生成内循环（M=1378，随分支/扩散步重复
  数千次）用闭式：`cycles = ⌈M/array_h⌉ × ⌈N/array_w⌉ × (K + array_h + array_w − 2)` +
  prefetch/drain 带宽项。M 大、阵列填满 → compute-bound，与 SCALE-Sim 无 stall 一致。
- **路径 C：解析 roofline（memory-bound aware）——文本 decode，2026-06-17 新增**。
  文本 decode 是 M=1。SCALE-Sim 的 OS 把 M 映射到阵列 32 行 → M=1 只用 1/32 阵列
  （overall util ~3%），被欠利用放大 ~37× 的计算时间盖死所有访存 → stall 恒 0、**任何带宽
  旋钮（片外 `Bandwidth` / 片上 `FilterSRAMBankBandwidth`）都不影响周期**（3 探针 + os/ws/is
  全验证；即便权重 896KB ≫ 256KB SRAM 仍 stall=0）。这与物理矛盾：M=1 每 token 要把整层
  权重从 DRAM 流一遍、计算量极小，本应 memory-bound（AI 在 roofline 拐点之下）。根因是
  M=1 的 ~37× 阵列欠利用把**等效计算屋顶砸到访存屋顶之下**，瓶颈成了"阵列几何"这第三堵墙；
  SCALE-Sim 为大 M 的 CONV/GEMM 设计、表达不了"M=1 填满阵列 + 权重/KV 带宽受限"。
  故文本每个 GEMM 改为闭式：
  `cycles = max(理想填满计算 = M·N·K / text_array_macs, 操作数流式 = N·K·operand_bytes / BW_bpc)`
  - `operand_bytes`：投影/FFN 取权重精度字节（int4=0.5/int8=1/fp16=2），attention 取 KV（fp16=2）；
  - `BW_bpc = peak_dram_gbps×1e9 / 500MHz`（满 32GB/s→64 B/cyc；disagg 半带宽组 `peak_dram_gbps=16`→32）；
  - `text_array_macs`：disagg = `pe_total − h·array_width`（text-accel）/ ours = `h·array_width_fp`（FP-FP）/ 其它 = `h·array_width`；
  - stream 与 M 无关（权重/KV 加载一次、跨 M 复用）→ **batch 摊薄，正解释 decode 批处理收益**。
  `bagel_sim.py` + `janus_sim.py` 同步（helper `_text_gemm_cycles`/`_text_array_macs`/`_text_bw_bpc`）。
  **影响**：文本现 memory-bound（text_mem≈100%），砍半带宽 → 文本阶段精确 2×；权重精度成主导
  → int4（figna/disagg）文本快于 fp16（ours）。文本周期较旧 OS 值缩 ~14×（disagg w8 243e9→16.7e9）；
  **所有 Bagel+Janus 文本数随之改变**（MM 全部、GenEval/GenEdit 文本部分；旧不变量
  `ours MM=127,169,309,589` 故意作废，现 ~16.6e9），需重算。能量账本与图像阶段（路径 A/B）不变。

**第三步（放大）**：文本 attention 随 kv_len 增长，按 `sample_rate=20` 分组、组内取平均
kv_len 仿真一次 × 组内步数（attention 周期对 kv_len 分段线性，组内平均即组和）；
每步周期 = (qkv + attn×heads + omap + ffn) × num_layer；图像阶段再 × 分支数相关项 × 步数。

### 3.2 输出指标

- **端到端周期 / 秒**：统一 500 MHz（所有 DSA 同频，与论文 §7.1.4 对齐口径一致）。
- **算子 breakdown（待做，需求 3，只出 Bagel）**：框架内部本就按 §2.1 的 6 类算子逐项累加，
  breakdown 只需把累加分桶记下来输出；ours 可再叠"高精阵列/低精阵列"一维。
  前置：先修文本侧 ffn_up ×2 口径（§8 核对项）。
- **硬件利用率（待做，需求 8/9 公共依赖）**：
  `utilization = Σ有效MAC / (PE_total × total_cycles)`，分子用能量账本已有的 MAC 计数，
  分母按各阵列物理 lane 数 × 周期累加，可按 AR/扩散阶段分桶（对照论文 Fig.14 的 75.36%）。
  实现是现有计数器的一个除法，无新仿真机制（分母口径见 §8 讨论项 4）。

---

## 4. 硬件建模

### 4.1 两级配置与精度约定

每个"硬件 × 任务"由两级配置完全确定：

1. **SCALE-Sim `.cfg`**（`configs/bagel/`）——物理硬件：阵列尺寸、SRAM、DRAM 带宽、
   dataflow（全部 OS）。
2. **workload JSON**（`topologies/{bagel,janus}/config_<hw>[_<task>].json`）——算法行为：
   稀疏率、分支 reuse 比例、cfg 槽位指向、解析模型镜像参数、任务参数。

**精度约定**：SCALE-Sim 不感知位宽，低精度的硬件收益（同面积更多 lane、同带宽更多元素、
同容量更多权重）统一折算成 cfg 等效参数，**一个精度档 = 一份 cfg**；harness 按算子类型把
子仿真路由到对应档的 cfg。注意"精度档"描述的是收益组合而非计算单元：例如 ours 的文本
FFN 档 = fp16 计算（FP-FP 阵列）+ INT8 权重存储（仅带宽/容量收益），不存在 INT8 计算
单元。精度对模型质量的影响属于算法侧评估（论文 Table 1/2），不进仿真器。

### 4.2 各硬件配置一览（Bagel，GenEdit 取值）

现有硬件：

| 硬件 | 阵列（cfg） | SRAM（I/F/O） | DRAM BW | 精度路由（文本投影 / 文本 attn / 图像侧） | 稀疏 / reuse 参数 | 复现机制 |
|---|---|---|---|---|---|---|
| **HCA**（=ARGUS all-off，§A 归一基准） | 同 ours 物理阵列 | 同 ours | 8 | ours 但五开关+`quant_proj` 全关：文本/图像投影·FFN 全 fp16、单路全 KV attention、无 reuse、图像宽 32 | `*_sim`=0、attn 单路全 KV | 异构计算阵列基线（T1/T2/T3 全关）；`--hw ours` + alloff 配置 |
| **base** | 32×64 fp16 | 64/64/64 KB | 8 | 全 fp16 单 cfg | 无 | 朴素脉动阵列下界 |
| **ours (ARGUS)** | FP-FP 32×32 + FP-INT4 32×32 双阵列。cfg：fp16 档 32×32；文本 FFN 档（`ours_ffn_w8.cfg`）32×32+BW16；int4 档 32×32, WBuf 128KB, WBW 32 | 32/32/32（fp16 档；FFN 档 WBuf 64 = 64KB×1B） | 8（文本 FFN 档 16） | 文本全 fp16、仅 FFN 权重 INT8 存储（FFN 档；纯文本 MM 则 AR-FFN 改走 W4A16 对齐 figna，见 §4.4 T3 行）/ 文本 attn fp16（KV 不量化） / 图像 qkv·omap 解析宽 64（`quant_proj` 开，INT4/W4A16；关时退回 32 仅 FP-FP）、FFN 宽 64、attention 每路 32 | `sparsity_cross_attn=0.38`、`low_precise_self_attn=0.49`、`image_only_sim=0.71`、`text_only_sim=0.28` | T1+T2+T3 全开，开关化（§4.4/§4.5） |
| **figna** | **双阵列**：两个 32×32 = 2048 物理 lane（与 ours/base 同面积）。int4 档 WBuf 128KB 物理、权重带宽 `FilterSRAMBankBandwidth=32`（int4 4× 元素带宽）。W4A16 限制 → **每算子只用一个子阵列（有效 1024 lane）、另一子阵列空闲，全程利用率 ~50%**（2026-06-12 确认） | 128KB/buffer（双阵列共享池） | ofmap 8、ifmap 8、filter 32（三 operand 独立，§6 第11条） | 投影/FFN int4（W4A16：MAC 按 int4、权重 0.5B、激活 2B）/ attn fp16 / 图像侧同样 int4 权重 | 无 | FIGNA 的 W4A16 FP-INT 单元（权重低精、激活高精） |
| **axcore** | 与 figna 完全相同（双阵列、W4A16、单子阵列 ~50% util） | 同 figna | 同 figna | 与 figna 完全相同 | 无 | FIGNA 同款 W4A16，**仅 w4a16 MAC 系数不同**（AxCore 0.50 vs FIGNA 0.57，见 §5）；`--hw axcore` |
| **flightvgm** | 32×64 fp16 | 64/64/64 | 8 | 单 cfg | `image_only_sim=0.136`、`text_only_sim=0.092`（FFN 激活稀疏，实测 profiling） | FFN 稀疏跳过 |
| **sdma** | 32×64 fp16 | 64/64/64 | 8 | 单 cfg | `sparsity=0.9`（KV 保留比） | KV 压缩 |

计划新增（rebuttal 需求，全部复用上表的机制，括号内为状态）：

| 硬件 | 配置方式 | 用途 | 状态 |
|---|---|---|---|
| **sa** | 一份 fp16 cfg（阵列/SRAM 对齐 RTL 设计规格）+ 稀疏参数全中性 JSON；机制同 base，差别仅在档位（base 是 2048-MAC 系统级，sa 是 core 级与 RTL 同规格） | 需求 1：core 级 latency+energy，与 FIGNA/AxCore 同口径 | （待做：cfg+JSON；能量系数见 §5） |
| **ours_noT3** | ours JSON 开关组合：`t3_soft=false`，其余全开（§4.5） | 需求 7：完整 ARGUS 减 T3（leave-one-out） | （待做：一份新 JSON，依赖开关化实现） |
| **ours_T3only** | ours JSON 开关组合：仅 `t3_soft=true`，`t1_soft`/`t1_hard`/`t2_soft` 全关 | 需求 8：只有 T3 的利用率 | （待做：一份新 JSON；原 `attn_mode: uniform_fp` 开关需求由 `t1_soft=false` 覆盖） |
| **disagg** | text 加速器（FIGNA 配置）跑 AR 阶段 + vision 加速器（S-DMA 配置）跑扩散阶段，两段串行相加，另一半芯片 idle 计入利用率分母 | 需求 9：分离式架构利用率，反证 T3 调度价值 | （待讨论后做：面积分配/能力给定/流水反论，见 §8） |

> 面积公平性说明：所有设计对齐 **2048 lane / 128KB-per-buffer SRAM / 500 MHz / 32 GB/s**
> （论文 §7.1.4 口径）。
> - **base/flightvgm/sdma**：完整 2048 个 fp16 MAC 的单一阵列（32×64）。
> - **ours**：32×32 FP-FP（1024 fp16 MAC）+ 32×32 FP-INT4（1024 lane）双阵列。**不存在
>   FP-INT8 计算单元**：INT8 只是权重存储格式，CRU 解量化回 BF16 后在 FP-FP 阵列上算。
>   文本/AR 阶段只 FP-FP 工作（有效 1024、FP-INT4 空闲）；图像/扩散 FFN 双阵列合用
>   （有效 2048）。
> - **figna**：同样双阵列 2048 lane，但 W4A16 限制下**每算子只用一个子阵列（有效 1024）**，
>   全程利用率 ~50%。**这是反证 T3 的关键**：只有 ours 图像 FFN 能用满 2048，figna 卡 50%。
>
> 三者总片上 SRAM 统一 128KB/buffer：ours/figna 的两个子阵列**共享一个 128KB SRAM 池**
> （单算子可用满 128KB，2026-06-12 决定），故单子阵列仿真的 cfg 也填 128KB 物理
> （元素数 = 128KB / 该 operand 字节宽）。**利用率分母一律用 2048 物理 lane × 周期**
> （含空闲子阵列），figna/ours-文本的 50% 才成立（§3.2，对照论文 Fig.14）。
> int4 档不加宽（W4A16 吞吐受 FP 激活侧限制），收益在权重 buffer（4× 元素）与权重带宽
> （`FilterSRAMBankBandwidth=32`，4× 元素/周期）。

cfg 槽位 → 算子的固定路由（精度路由的机制；括号内为 ours 的取值）：

| 槽位 | 文本 decode 驱动 | 图像生成驱动 |
|---|---|---|
| `config_comm0` | qkv / omap（ours: `quant_proj` 开→`ours_proj_int4.cfg` INT4/W4A16 合并宽 64；关→fp16 cfg） | 文本侧 qkv（ours: 同左） |
| `config_comm1` | attn_qk / attn_sfmxv（ours: fp16 cfg） | （不用） |
| `config_ffn_text`（新增槽位） | ffn_up / ffn_down（ours: `ours_ffn_w8.cfg` = INT8 存储；**纯文本 MM 例外：走 `config_int4`=`ours_int4.cfg`，AR-FFN 在 FP-INT4 阵列跑 W4A16 对齐 figna，门控 `t3_soft and task=='MM'`，2026-06-17**；`t3_soft` 关时退回 fp16 cfg；其他硬件 = 其原投影 cfg，行为不变） | （不用） |
| `config_comp0` | （不用） | 图像侧能量精度标签 |
| `config_comp1` | 预留 | 预留 |

`ours_int8.cfg`（32×64、BW16）**废弃**：它假设了一个不存在的 FP-INT8 计算阵列。

两点实现说明：

- 周期计算有三条路径（§3.1）：文本 decode 用解析 roofline（路径 C，2026-06-17；阵列宽度/带宽
  取 JSON 的 `array_*`、`pe_total`、`peak_dram_gbps`，不再调 SCALE-Sim）；图像阶段文本侧投影
  仍调 SCALE-Sim（路径 A，参数从 `.cfg` 读）；图像内循环用解析折叠（路径 B，参数取 JSON
  `array_*`/`*bufsz`/`*bw`）。解析路径所需 JSON 字段是把 cfg 数值抄一份，两处需人工保持一致
  （历史实现选择）。
- 能量计费需要知道每个子运行的精度（pJ/MAC 与字节数随精度变）。baseline 直接取该子运行
  所用 cfg 的文件名后缀（`*_int4.cfg` → 按 int4 计费）；ours 因精度按操作数拆分（MAC /
  权重 / 激活可不同，见 §5），改为调用点显式传入，不再从文件名嗅探。

### 4.3 稀疏 / 量化 / reuse 如何进入周期模型

所有"算法优化"统一折算为**有效问题尺寸缩小或等效参数**，再走与 baseline 完全相同的仿真路径：

- **KV 稀疏**（ours cross-attn 块跳过、sdma KV 压缩）→ 缩短 attention 的有效 `kv_len`。
  块/token 级稀疏下剩余计算确实是更小的稠密 GEMM，转化忠实；mask 预测与 gather 开销不计
  （论文实测 0.39%，对所有带稀疏硬件统一豁免）。
- **混合精度 attention**（ours）→ 拆低精路（保留 cross-attn KV + 低精 self-attn KV）与
  高精路（其余 self-attn KV），两路在双阵列并行，attention 周期 = `max(低精路, 高精路)`
  （建模并行执行与负载不均）。
- **条件分支 FFN reuse**（T2）→ `image_only_sim`/`text_only_sim` = 条件分支中与 normal
  分支激活相似度超阈值（0.90）、可直接复用 FFN 结果的 token 比例，缩小该分支 ffn_up 的
  有效 M。flightvgm 复用同两个旋钮但填其激活稀疏率（语义不同，口径分开 profile）。
  三分支任务（GenEdit）的 `full_cache` 分支也享受 FFN reuse，按 `text_only_sim` 缩减
  （与 without_img 分支同口径，2026-06-12 决定）；对所有带 T2 的硬件（ours、flightvgm）
  一致生效，base/sdma/figna 因 `*_sim=0` 无影响。
- **低精度计算/存储** → 路由到对应精度 cfg（§4.1）。

逐层/逐 head 的稀疏度差异用任务级均值代表：周期对有效尺寸线性、总周期为逐层求和，
均值与逐层取值的总和严格相等（例外见 §6 第 2 条）。

### 4.4 ARGUS 三个技术点的建模落点

| 技术点 | 论文机制 | 模型落点 | 未建模（声明可忽略） |
|---|---|---|---|
| **T1** 区域结构化稀疏 attention（HSD） | cross-attn 块跳过+低精；self-attn 高精、masked 部分降精；HSD 派发到双阵列 | `sparsity_cross_attn` + `low_precise_self_attn` 拆两路 kv_len，双路并行取 `max` | mask 预测与派发开销（0.39%） |
| **T2** 相似度感知 FFN reuse（SAU） | 条件分支中相似 token 跳过 FFN，SAU 片上累加 | `image_only_sim`/`text_only_sim` 缩小条件分支 ffn_up 的 M | SAU 累加开销；ffn_down 未随 reuse 缩减（偏保守，§8 核对项） |
| **T3** 阶段自适应阵列编排 | AR 阶段：FFN 权重 INT8 存储、CRU 解量化回 BF16 再算（带宽减半）；扩散阶段：FFN 权重拆 INT4+BF16 两份，双阵列同算一个算子 | AR 阶段：仅 FFN 槽位指 `ours_ffn_w8.cfg`（32×32、`Bandwidth: 16` = 权重带宽 ×2，计算仍在 FP-FP），attention 全 fp16 无收益（KV 不量化），qkv/omap **不从 T3 获益**（T3 只动 FFN），但其 INT4 量化由独立开关 `quant_proj` 负责（开则 qkv/omap 走 W4A16 合并宽 64，用到 FP-INT4 阵列，见 §4.2/§5）；`quant_proj` 关、且非纯文本 MM 时，AR 阶段 FP-INT4 阵列才真正空闲（**例外：纯文本 MM 没有扩散阶段来占用 FP-INT4 阵列，故 AR-FFN 改在其上跑 W4A16，槽位指 `ours_int4.cfg`（32×32、WBW32、INT4 权重），与 figna 对齐；门控 `t3_soft and task=='MM'`，2026-06-17。投影/attention 不变**）；扩散阶段：解析模型 FFN 用 `array_width=64`（INT4+BF16 双阵列）、qkv/omap 用 `array_width_proj`（`quant_proj` 开 = `array_width=64`，INT4 权重 W4A16；关 = `array_width_fp=32`，仅 FP-FP）、attention 每路 `array_width_half=32` | 阶段切换过程（论文称 negligible）；INT4/BF16 拆分比例动态调整（用静态等效宽度代替） |

### 4.5 ARGUS 技术五开关（取代 ours_balence 变体）

`ours_balence` 硬件类型**已决定删除**（2026-06-12）：它只是 ours + T1 的硬件负载均衡
调度，不是独立设计；而两阶段编排（T3）本来就融合在 ours 内。ARGUS 技术拆为五个 JSON
开关（ours 默认全开），ablation / leave-one-out 全部用开关组合表达：

| 开关 | 机制 | 现状 | 关闭后行为 |
|---|---|---|---|
| `t1_soft` | 区域结构化稀疏 attention：cross-attn 块跳过 + 低精 self-attn 分类（`sparsity_cross_attn`/`low_precise_self_attn` 拆两路 kv） | **已有**（仅图像阶段；文本阶段无 KV 稀疏，按设计） | 单路、全精、全 KV（原计划的 `attn_mode: uniform_fp`，需求 8 并入此开关） |
| `t1_hard` | HSD 双阵列派发 + 负载均衡 | **部分**：双路并行取 `max()` 已在 ours；负载均衡（两路平均）来自原 ours_balenced 死代码 | 关 = `max(两路)`（只派发、不均衡）；开 = 两路平均（理想均衡） |
| `t2_soft` | 相似度感知条件分支 FFN reuse（`image_only_sim`/`text_only_sim` 缩条件分支 ffn_up 的有效 M） | **已有**（ffn_down 未随 reuse 缩减，§8 核对项 10） | `*_sim` 视为 0 |
| `t2_hard` | SAU 片上累加 | **缺失**：reuse 被当作免费，SAU 开销未建模；功率缺口走 §8 讨论项 3 的背景功率方案 | 预留占位开关（当前无行为差异），待 SAU 开销模型定型后补实现 |
| `t3_soft` | 阶段自适应编排（§4.4 T3 行） | **已有**（2026-06-12 修正后口径：INT8 仅权重存储、无 FP-INT8 阵列） | 文本 FFN 槽位退回 fp16 cfg（BW 8）；图像解析 FFN `array_width` 64→32；能量按 fp16 / 2B |

### 4.6 Baseline 公平性原则

- 所有硬件对齐 2048 MAC / 500 MHz / 同 DRAM 带宽与类型；
- baseline 稀疏率/压缩率取其论文声明或其方法在我们模型上的实测 profiling，不做有利截断；
  稀疏收益按理想尺寸缩小给出（上界），即对 baseline 同样慷慨；
- baseline 不具备的能力完全不给，而不是给弱化版；
- 电路级单元设计差异（如 FIGNA 与 AxCore 的 FP-INT 乘法器）不影响吞吐建模（同为每周期
  每 PE 一次 MAC），通过能量系数注入区分（§5）。

---

## 5. 能量建模（Phase F）

能量 = Σ(事件数 × 每事件能量)：

| 事件 | 计数来源 | 默认系数（28nm 文献基线） |
|---|---|---|
| MAC | `M×N×K`（按 fp16/int8/int4/w4a16 分桶） | 1.5 / 0.2 / 0.1 pJ（Horowitz ISSCC'14）；**w4a16 按硬件区分**：ARGUS 0.47 / FIGNA·disagg 0.57 / **AxCore 0.50**（见下） |
| SRAM 读写 | 闭式几何估计（fold 粒度），字节按精度折算 | 0.05 pJ/B（Sze et al. 2017） |
| DRAM 读写 | 每张量一次（M·K、K·N、M·N），M=1 decode 下精确 | **DDR4 20 pJ/bit**（2026-06-17 改；HBM2 3.9 可选，`dram_type` 切换） |
| DRAM 背景 | 总周期 × idle 功率（运行时间项→罚慢设计） | **500 mW**（DDR4 子系统；2026-06-17 100→500） |
| 计算阵列静态 | 总周期 × 静态功率（全 2048-lane 漏电+时钟，所有 hw） | 150 mW（占位，§6.5） |

不用 SCALE-Sim 访问报告的原因：harness 跑的是 N=64 tile 子问题，其 DRAM 计数被固定
prefetch buffer 填充主导，放大会系统性高估；故计数用真实 M/N/K 闭式估计，**放大系数与
周期累加严格同一个 multiplicity**（层数 × 步数 × head 数 × 分支数），能量和周期描述同一次推理。

**精度按操作数拆分（2026-06-12 修正）**：MAC 计费精度与权重/激活字节数解耦——当低精度
只是存储格式时，MAC 仍按计算单元的精度计费。ours 各算子的计费口径：

| 算子 | MAC | 权重字节 | 激活字节 |
|---|---|---|---|
| 文本 qkv / omap（`quant_proj`） | w4a16 | 0.5B（INT4） | 2B |
| 文本 attention | fp16 | 2B | 2B |
| 文本 ffn_up / ffn_down（图像任务） | fp16（CRU 解量化后在 FP-FP 上算） | 1B（INT8 存储） | 2B |
| 文本 ffn_up / ffn_down（**纯文本 MM**，2026-06-17） | **w4a16**（FP-INT4 阵列，对齐 figna） | 0.5B（INT4） | 2B |
| 图像 qkv / omap（`quant_proj`） | w4a16 | 0.5B（INT4） | 2B |
| 图像 attention | fp16 | 2B | 2B |
| 图像 ffn_up / ffn_down | 50% fp16 + 50% **w4a16**（双阵列各 32 lane 静态拆分） | 0.5B | 2B |

figna（W4A16）同口径拆分：投影/FFN 两阶段均 MAC **w4a16**、权重 0.5B、激活 2B；attention 全 fp16。

**w4a16 MAC 系数（int4 权重 × fp16/bf16 激活）**：FIGNA/AxCore 的 FP-INT 乘法器与 ARGUS
扩散阶段的 INT4 阵列都属此类——**不是纯 int4×int4**，16-bit 激活路径主导能量，介于 int4
（0.1）与 fp16（1.5）之间。默认 **0.5 pJ 为占位值**，待 RTL 后仿校准（§5 版本 A），可经
JSON `energy_coefficients.mac_pj.w4a16` 覆盖。**按硬件区分（2026-06-17）**：
**ARGUS = 0.47 pJ / FIGNA·disagg = 0.57 pJ / AxCore = 0.50 pJ**（ARGUS 的 INT4+BF16 双阵列
W4A16 单元最省，FIGNA 的 FP-INT 乘法器最费，AxCore 居中；`coefficients.py` `DEFAULT_COEFFICIENTS`
按 hw 注入）。**AxCore 与 FIGNA 在 harness 里行为完全相同（W4A16 投影/FFN、attn fp16、单子阵列），
唯一差别就是这个 w4a16 系数。** sdma/flightvgm/base 不发 w4a16 MAC，其值无效。

此系数是能效结论的**最敏感旋钮**（figna 90% MAC 是 w4a16）。Bagel GenEdit 实测 **ARGUS-vs-FIGNA
能效比** 随 w4a16（此处用同值对比看敏感性）变化：

| w4a16 (pJ, 同值) | 0.5 | 1.0 | 1.25 | 1.5(=fp16) |
|---|---|---|---|---|
| ARGUS-vs-FIGNA 能效 | 1.32 | 1.48 | 1.54 | 1.60 |

物理上 w4a16 = int4 权重 × **fp16(16-bit) 激活**，能量由 16-bit 激活读取+fp 累加主导、int4 只省乘法器
一侧 → 介于 int4(0.1) 与 fp16(1.5) 之间。采纳的 0.47/0.57 为占位/RTL-informed，**真实值待 RTL
后仿标定（§5 版本 A）**。

系数定位：默认 pJ 表五种硬件共用——**硬件间能量差异来自访问次数**（稀疏少算、低精少搬、
KV 缩短）。电路级差异通过 JSON `energy_coefficients` 按硬件覆盖注入。

**（待做，需求 1 已定方案）SA 与 core 级对比出两版系数，共用同一计数**：
- 版本 A（主口径）：RTL 后仿功耗折算每 MAC / 每 SRAM 访问 pJ，经 `energy_coefficients`
  覆盖；与 RTL 对比时只取 Compute+SRAM 桶（RTL 边界不含 DRAM）。
- 版本 B：同一份计数，系数换 Accelergy(28nm) 输出。
- 报告两版及比值。对外口径："系数来自 RTL 实测，计数来自仿真器，Accelergy 作第三方
  一致性 sanity check，不作数字来源。"

---

## 6. 已知近似与适用边界（对外主动声明）

1. **线性放大**：tile 缩放、kv_len 抽样、层/头复用假设周期对尺寸线性；OS dataflow 稠密
   GEMM 下结构性成立，误差仅 ceil 离散化（<1 fold/算子）。
2. **稀疏度取任务级均值**：线性路径下与逐层取值严格等价；唯一例外是 ours attention 的
   `max(两路)`（凸性 ⇒ 轻微低估 ours 周期，方向对我们有利，需声明；可逐层 JSON 扫描做
   敏感性）。
3. **非 GEMM 算子不计**：softmax/LayerNorm/激活不在周期与能量内。辩护：FLOP 占比 <1%；
   ARGUS 架构上由 CRU 处理并与阵列计算重叠；所有模拟硬件同口径。已知风险：GPU 实测时间
   含这些算子（对比时我们略占便宜）；CRU 功率（论文 Table 3：32.67mW/7.8%）未计入能量。
4. **算子串行**：层内算子顺序累加，不建模流水/重叠；decode 依赖链长，影响小，全硬件一致。
5. **能量 = 动态 + 计算阵列静态底（2026-06-17 起）**：除动态(MAC/SRAM/DRAM)外，按
   `array_static_mw × 运行时间` 给**所有硬件**计一个全芯片(2048-lane)静态(漏电+时钟树)能量底
   （`coefficients.py` 默认 150mW 占位；`accountant.compute_array_static_pJ`）。**这修了此前
   "动态下界"对慢设计的系统性偏袒**：FIGNA 跑 2.6× 长却不付阵列漏电，能效被高估（ARGUS-vs-FIGNA
   能效 1.14×→1.32×）。`array_static_mw` 是占位值（漏电+时钟无法从论文 Table-3 总功率分离，需 RTL，
   见 §5 版本 A）；设 0 即回到纯动态下界口径。DRAM 背景功率(`dram_idle_mw`=**500mW**, DDR4 子系统)同理按时间计——它和阵列静态都是"运行时间项",**慢的设计(如 FIGNA 跑 2.6×)多付**,是比每比特读写能量更有效的能效杠杆。
6. **DRAM 无溢出重取**：每张量读一次；M=1 精确，大 M 下若权重 buffer 不足会低估
   （最坏 ~2×，可加保守因子）。
7. **prefill 不仿真**：`kv_cache_init` 作初始条件（decode-throughput 论文通行做法）。
8. **T3 静态化**：只建模调度的稳态等效，不建模切换过程。
9. **mask 形状不区分**：causal/双向均按稠密 kv_len GEMM。
10. **单实例运行**：共享临时 topology 文件，同一时刻只能跑一个 harness 实例。
11. **三 operand DRAM 带宽独立（非近似）**：USER 模式下 SCALE-Sim 把 ifmap/filter/ofmap
    的 backing（DRAM）带宽分别取自 `[layout] IfmapSRAMBankBandwidth` /
    `[layout] FilterSRAMBankBandwidth` / `[architecture_presets] Bandwidth`（顶层
    `Bandwidth` 只驱动 ofmap）（`single_layer_sim.py:267-270`）。带宽单位是 元素/周期、
    `word_size` 写死 1 字节（bitwidth-blind），故低精度的权重带宽收益直接写进
    `FilterSRAMBankBandwidth` 的数值：所有精度物理权重带宽对齐 16 字节/周期——
    fp16 = 8 elem×2B、int8 = 16 elem×1B、int4 = 32 elem×0.5B，公平且精确（不是近似）。
    ours 文本 FFN 档只调 `FilterSRAMBankBandwidth=16`（仅权重 ×2），ifmap/ofmap 保持 8。
12. **图像 FFN MAC 50/50 静态拆分**：INT4+BF16 双阵列按各 32 lane 静态对半计费 MAC
    能量，不建模拆分比例的动态调整（与 §4.4 T3 的静态等效宽度同源）。
13. **文本 decode 用解析 roofline（2026-06-17，§3.1 路径 C）**：M=1 decode 不走 SCALE-Sim
    （其 OS 映射 M=1 只用 1/32 阵列、util ~3%，盖死访存使任何带宽都不生效——见路径 C），
    改 `cycles = max(理想填满计算, 权重/KV 流式 / BW)`。两点已知简化（统一施加、影响 <1%）：
    (a) **GQA KV 复用**：attention KV 流式按 `num_head_q` 计、未按共享的 `num_head_kv`
    （Bagel GQA 28:4 → memory-bound 下 KV 多算 ~7×；能量账本同口径；Janus heads 相等无此问题）；
    (b) 理想填满计算忽略 fill/drain（略乐观，但文本 memory-bound 时 compute 项基本不触发）。
    注：此模型令 §6.6 的"大 M 权重 buffer 不足低估"对文本不再适用（文本已显式按权重带宽建模）。

---

## 7. Rebuttal 需求 ↔ 框架映射（索引）

详细内容已并入正文，此表只做导航：

| 需求 | 内容 | 框架落点 | 状态 |
|---|---|---|---|
| 1 | plain SA vs FIGNA/AxCore，latency+energy | §4.2 计划表 `sa`；§5 两版系数（A: RTL 主口径，B: Accelergy） | 待做 |
| 2 | SenseNova 模型 | §2.1 参数表；需算法侧提供结构/workload/稀疏 profiling 三类数据 | 待数据 |
| 3 | 算子 breakdown（只出 Bagel+3 bench） | §3.2 输出指标 | 待做（先修 ffn_up ×2） |
| 4 | 技术点 ablation | **已定单开口径**：基线 = ours 五开关全关（§4.5，等价于文本全 fp16、attention 单路全 KV、`*_sim=0`、图像 width 32）；在基线上分别**只开 T1**（`t1_soft`+`t1_hard`）/ **只开 T2**（`t2_soft`）/ **只开 T3**（`t3_soft`，即 `ours_T3only`，与需求 8 共用） | 待开关化实现后出配置 |
| 5 | GPU 功耗对比 | GPU = 稳态采样均值 × 时间；ARGUS 侧用含 DRAM 动态+idle 的总能量、RTL 校准系数 | 待讨论（§8） |
| 6 | 更长上下文 / batch | §2.3 扩展维度 | 上下文已支持待数据；batch 待做 |
| 7 | ARGUS−T3 baseline（保 T1/T2） | §4.2 计划表 `ours_noT3` | 待做（纯 JSON） |
| 8 | ARGUS+T3 only，测利用率 | §4.2 计划表 `ours_T3only`；§3.2 利用率指标 | 待做（原 `attn_mode` 开关由 §4.5 `t1_soft=false` 覆盖） |
| 9 | 分离式 text+vision 加速器，测利用率 | §4.2 计划表 `disagg`；§3.2 利用率指标；含 5 档 split 扫描（w2/4/8/16/32）+ 满/半带宽两组 + 分阶段 4 利用率（text/image × compute/mem） | ✅ 已做，见 `NEED9_DISAGG.md` §6-7 |

---

## 8. 待讨论 / 待确认清单

**代码核对项（先确认/修复再出数）**

7. `run_model()` 只在 task∈{GenEdit, GenImage} 进图像阶段，`--task GenEval` 会跳过图像
   生成——核对历史 GenEval 数据怎么跑的，并把条件修为含 GenEval。
8. 文本阶段 ffn_up 未 ×2（gate+up），图像阶段 ×2——Bagel 是 gated FFN，文本侧应同口径；
   影响 breakdown（需求 3）准确性，建议直接修。
10. T2 reuse 只缩 ffn_up 未缩 ffn_down——被 reuse 的 token 理应跳过整个 FFN；当前偏保守
    （低估 T2 收益），确认是否有意；若改，需求 4/7/8 一起重跑。

**外部依赖**

14. SenseNova（需求 2）数据清单：模型结构参数；每数据集 workload 统计（prompt 长度拆
    text/image、生成长度、扩散步数、CFG 分支数）；每数据集稀疏/相似度 profiling
    （`sparsity_cross_attn`、`low_precise_self_attn`、ours 的 `image_only_sim`/
    `text_only_sim`、flightvgm 口径的激活稀疏率）。

**新增待办（2026-06-13，dingli review 隔夜数据后）**

18. **Breakdown 输出 text/image 分阶段、qkvo 合并版**：在现有逐算子 breakdown 基础上，
    出一份按 text/image 两阶段分开、且 qkv+omap 合并为 qkvo 的精简表（5→4 桶：
    qkvo / attn / ffn_up / ffn_down）。纯重聚合现有 `breakdown.csv`，无需重跑。

**新增待办（2026-06-17，文本 roofline 模型改造后）**

19. **所有含文本的旧数作废、需重跑**（§3.1 路径 C / §6.13 影响面）：文本 decode 周期模型从
    SCALE-Sim OS 改为解析 memory-bound roofline，文本周期缩 ~14×。受影响：**MM 任务全部**
    （纯文本，旧不变量 `ours MM=127,169,309,589` 故意作废，现 ~16.6e9）、**GenEval/GenEdit 的
    文本部分及总周期**、需求 3/4/6 中文本相关数。Bagel 已改并验证；Janus 已同步移植（`janus_sim.py`，
    `_text_gemm_cycles` 等，冒烟通过）。能量账本与图像阶段不变。need-9 的 Bagel GenEdit 已在新
    模型上重跑（`NEED9_DISAGG.md` §7）；其余 hw/task 待逐一重跑（dingli 自行处理）。
20. **（可选）GQA KV 复用修正**：§6.13(a)，attention KV 流式改按 `num_head_kv` 计而非
    `num_head_q`（需同步能量账本口径）；影响 <1%，预先就存在，暂保留。