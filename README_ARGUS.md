# ARGUS-SCALE-Sim — 项目总览（唯一上手文档）

> 接手第一份、也是唯一要通读的文档。运行指令单独看 [`RUN.md`](RUN.md)。
> 上游 SCALE-Sim 的通用介绍在 `README.md`；Ramulator / 稀疏等原生功能见 `README_ramulator.md` / `README_Sparsity.md`。

---

## 0. 一分钟看懂

- 本仓库是 **SCALE-Sim v3**（上游开源的脉动阵列 cycle-accurate 加速器仿真器）的一个 **fork**。
- Fork 目的：给论文 **ARGUS**（*Enabling Efficient MLLM Inference through Hierarchical Compression and
  Algorithm-Hardware-Datatype Co-design*，投稿 DAC 2026；当前主线 **MICRO 2026 rebuttal**）提供实验数据。
  ARGUS 是一个面向**统一多模态大模型（Unified MLLM）解码阶段**的加速器。
- 做法：用 SCALE-Sim 当底层「算一层 GEMM 要多少 cycle」的引擎，上层自建一套 **MLLM 解码吞吐 harness**
  （`simulation_core/` + `run_*.py`），把一整个 MLLM 解码轨迹拆成成百上千个小 GEMM 子仿真，
  按层数 / head 数 / KV 长度 / 采样步长加权求和，得到端到端 cycle，再换算吞吐 / 能耗。
- **产出**：端到端解码 cycle → 吞吐（加速比）+ 能效比 + 可选能量分解，写进 `.log`
  （每个 workload JSON 的 `result_path` 指向，通常 `results/{bagel,janus}/*.log`）。
- **报告的是相对加速比 / 能效比，不是绝对时延**——所有硬件走完全相同的放大路径，放大误差在对比中系统抵消。

**▶ 最小可跑示例**（读完这份就能跑出第一个 log；完整命令表见 [`RUN.md`](RUN.md)）：
```bash
source env.sh                                                                   # 激活 conda 环境 zhb-scalesim-v3
python run_bagel.py --hw ours --task GenEdit --config topologies/bagel/config_ours.json
# 结果写到该 JSON 的 result_path 指向的 .log；加 --energy 追加能量分解
```

---

## 1. 术语表（读下文前先扫一眼）

**模型**
| 术语 | 含义 |
|---|---|
| MLLM | Unified Multimodal LLM，统一多模态大模型（同一模型既理解图像又生成图像） |
| **Bagel** / **Janus** | 两个被评测的开源 MLLM（decoder 主干结构不同） |
| SenseNova | 支线模型，基于 Janus 结构 + 新文本模型 |
| prefill / decode | prefill=一次性处理 prompt；decode=逐 token 自回归生成（M=1，本项目主要建模这一段） |
| GQA / MHA | 分组 / 多头注意力；GQA 的 KV heads 少于 Q heads（Bagel q28/kv4，Janus MHA q16/kv16） |

**任务**
| 术语 | 含义 |
|---|---|
| **GenEdit** | 图像编辑（有文本 decode + 图像扩散两阶段） |
| **GenEval**（= GenImage） | 图像生成（同上两阶段） |
| **MM** | 多模态理解（纯文本 decode，无图像扩散阶段） |

**硬件（`--hw` 取值）**
| 术语 | 含义 |
|---|---|
| **ours** = **ARGUS** | 本文提出的加速器：FP-FP 32×32 + FP-INT4 32×32 **双阵列**，含下面 T1/T2/T3 三技术 |
| **HCA** | = `ours` 把三技术**全关**（异构阵列基线）。**§A 结果表的归一基准（=1.00）** |
| **SA** = **base** | 纯脉动阵列（single systolic array）基线 |
| **FIGNA** (`figna`) | baseline 加速器，W4A16（int4 权重 × fp16 激活） |
| **AxCore** (`axcore`) | FIGNA 同款设计，仅 w4a16 能量系数更低（0.50 vs 0.57）；周期/利用率与 FIGNA 完全一致 |
| **FlightVGM** (`flightvgm`) | baseline 加速器 |
| **S-DMA** (`sdma`) | baseline 加速器，靠 KV 稀疏 |
| **disagg** | 分离式（disaggregated）基线：text 加速器（FIGNA 配置）跑自回归阶段 + vision 加速器（S-DMA 配置）跑扩散阶段，两段串行、另一半芯片 idle。用来**反证 ARGUS 的 T3 统一调度价值**（need-9） |

**ARGUS 三个技术点（`ours` 的核心，可逐个开关做 ablation）**
> HSD / SAU / CRU 是 ARGUS 论文里的硬件模块名。每个技术有 `*_soft` / `*_hard` 两级开关：
> **soft = 算法效果**（稀疏/reuse 带来的省算省搬）；**hard = 硬件派发/累加机制**（双阵列均衡、片上累加等）。

| 术语 | 论文机制 | 模型落点（JSON 开关） |
|---|---|---|
| **T1** | 区域结构化稀疏 attention（HSD 单元）：cross-attn 块跳过 + 低精 self-attn，派发到双阵列 | `t1_soft`（稀疏本身）/ `t1_hard`（双阵列负载均衡）；缩小有效 KV `sparsity_cross_attn` / `low_precise_self_attn` |
| **T2** | 相似度感知 FFN reuse（SAU 单元）：相似 token 跳过 FFN | `t2_soft`（reuse 效果）/ `t2_hard`（SAU 片上累加，当前占位未建模）；缩小 ffn_up 有效 M（`image_only_sim` / `text_only_sim`） |
| **T3** | 阶段自适应阵列编排：自回归阶段 FFN 权重 INT8 存储、扩散阶段 FFN 拆 INT4+BF16 双阵列同算 | `t3_soft` |
| all_off / all_on | 五个开关全关（=HCA）/ 全开（=完整 ARGUS） | ablation 里 `onlyT1`/`onlyT2`/`onlyT3`/`noT3` 是单开或留一 |

**其它**
| 术语 | 含义 |
|---|---|
| **w4a16** | int4 权重 × fp16 激活的 MAC；能量系数**按硬件区分**（ours 0.47 / figna·disagg 0.57 / axcore 0.50），是能效结论**最敏感的旋钮** |
| roofline / memory-bound | M=1 的 decode 是访存瓶颈；周期取 `max(计算, 访存)`（见 §3） |
| **need-N** | rebuttal 中 reviewer 编号的需求/问题。常见：**need-2** SenseNova；**need-3** 算子 breakdown；**need-6** 上下文长度/batch 扫描；**need-7** ARGUS 去掉 T3；**need-8** 只开 T3 的利用率；**need-9** 分离式（disagg）基线利用率 |

---

## 2. ⚠️ 一条最重要的背景：2026-06-17 分水岭

**2026-06-17 文本 decode 周期模型从 SCALE-Sim 的 OS 数据流改成解析 memory-bound roofline**
（`bagel_sim.py` / `janus_sim.py` 的 `_text_gemm_cycles`）：

```
cycles = max( M·N·K / text_array_macs,  N·K·operand_bytes / BW )
```

起因：SCALE-Sim 的 OS 把 M=1 的 decode 误建成 compute-bound（M 只映射到阵列 32 行、util ~3%），
计算时间被欠利用放大 ~37× 盖死所有访存，任何带宽旋钮都不影响周期。物理上 M=1 应是 memory-bound。

**这条线把所有「旧模型」数据作废**——凡 06-17 之前跑的、含文本阶段的结果都不能用。
同日还把能量系数统一为 **DDR4 20 pJ/bit + DRAM 背景 500 mW + 计算阵列静态 150 mW + per-hw w4a16**。
判断一份数据能不能用，先看它是不是新模型。

---

## 3. 代码结构地图

```
SCALE-Sim/
├── run_bagel.py / run_janus.py        # 入口壳（argparse: --hw --task --config --energy --validate）
├── run_sensenova.py                   # 支线 SenseNova 入口
├── simulation_core/                   # ★ 本 fork 的核心 harness（调用 scalesim，不是它的一部分）
│   ├── bagel_sim.py                   #   Bagel_sim：编排一整个解码轨迹
│   ├── janus_sim.py                   #   Janus_sim：同上
│   ├── sensenova_sim.py               #   SenseNova_sim：Janus 结构 + 新文本模型
│   └── energy_accounting/             #   能耗核算（Phase F，--energy 才启用）
│       ├── coefficients.py            #     pJ 系数表（MAC/SRAM/DRAM，28nm baseline）
│       ├── accountant.py              #     EnergyAccountant 累加器 + 报表
│       ├── extractors.py              #     从 SCALE-Sim 报表抽 access count
│       └── validate.py                #     --validate：用 accelergy CLI 交叉校验
├── scalesim/                          # 上游 cycle-accurate 引擎
│   ├── scale_sim.py / simulator.py / single_layer_sim.py
│   ├── compute/                       #   三种数据流：systolic_compute_{ws,os,is}.py + operand_matrix + sparsity
│   └── memory/                        #   双缓冲 scratchpad + read/write buffer/port
├── configs/bagel/*.cfg                # 硬件规格（ArrayHeight/Width, *SramSzkB, Bandwidth, Dataflow…）
├── topologies/{bagel,janus,sensenova}/*.json   # 仿真流程配置（KV、gen_len、稀疏率、阵列尺寸、指向 .cfg）
├── topologies/<family>/*.csv          # 上游风格工作负载（CONV 默认；GEMM_mnk/ 需 -i gemm）
├── rundir-accelergy/                  # accelergy/CACTI/Aladdin 交叉校验运行目录
├── results/{bagel,janus}/             # harness 输出 log
└── devlog_rebuttal/                   # rebuttal 权威结果 + 历史数据（gitignore，见 §7）
```

---

## 4. harness 内部流程（`bagel_sim.py` 为例，Janus / SenseNova 同构）

`Bagel_sim` 的方法链：

1. **`read_from_json(cfg_path)`** — 读工作负载 + 硬件 cfg 路径。**关键 dispatch**：
   - `ours`/`figna`：按精度分多个 cfg（`config_fp16`/`config_int4`）→ 字段 `config_comp0/1`、`config_comm0/1`、
     `config_ffn_text`；`ours` 另读 5 个技术开关 `t1_soft`/`t1_hard`/`t2_soft`/`t2_hard`/`t3_soft`（默认全 true）+ `array_width_fp`。
   - `base`/`flightvgm`/`sdma`：单个 `config` 复用所有槽位。
   - **新增硬件类型就改这里的 if/elif 链**（bagel 和 janus 都要改），并加对应 `configs/bagel/<hw>*.cfg` + `topologies/.../config_<hw>*.json`。
2. **`build_topologies(kv_len, is_gen_text, part)`** — 给每个子仿真写一份单层 GEMM topology 到
   `topologies/bagel/layer.csv`。`part` 选投影：`qkv / attn_qk / attn_sfmxv / omap / ffn_up / ffn_down`。
3. **`run_sim_once(...)`** — 调上游 scalesim（GEMM 模式，layout `layouts/GEMM_mnk/vit_l_KM_KN.csv`）返回单层 cycle。
   **2026-06-17 起仅图像阶段的文本侧投影用它**；文本 decode 已迁到 4b。
4. **`run_sim_once_comp(...)`** — 解析式替代：用 array 尺寸 + tile + op 尺寸算 fold cycle，不调 SCALE-Sim（快）。图像内循环用它。
   - **4b. `_text_gemm_cycles(...)`（2026-06-17 新增）** — 文本 decode 解析 memory-bound roofline，见 §2。
5. **`run_gen_text()` / `run_gen_image()`** — 组合每步 cycle，按层数/head/采样步长/KV 长度放大求和。
   稀疏建模 = 缩小有效 KV 长度：`ours` 用 `sparsity_cross_attn`+`low_precise_self_attn`（T1）；`sdma` 用 `sparsity_kv`；baseline 用全量 KV。
6. **`run_model()`** — 先文本生成，再（GenEdit/GenImage）图像生成，写 `result_path`。

**时钟**：harness 默认 **500 MHz**（`_append_to_log` 里把 cycle 换算成秒）。
方法学权威细节（每类算子怎么建模、各 need 的口径）见 `devlog_rebuttal/docs/REBUTTAL_FRAMEWORK.md`。

---

## 5. 两级配置系统（容易混）

| 层级 | 文件 | 管什么 |
|---|---|---|
| 仿真流程 | `topologies/{bagel,janus,sensenova}/config_<hw>[_<task>].json` | KV 大小、gen_text_len、gen_image_step、稀疏率、阵列尺寸、buffer、采样率、指向 .cfg |
| 硬件规格 | `configs/bagel/<hw>*.cfg` | ArrayHeight/Width、*SramSzkB、Bandwidth、Dataflow、可选 [layout]/[sparsity] |
| 上游工作负载 | `topologies/<family>/*.csv` | CONV（默认）或 MNK（`-i gemm`） |

**流程 JSON 示例**（`config_ours.json` 节选）：
```jsonc
{
  "kv_cache_init": 3254,          // prefill 结束时 KV 长度（数据集平均 prompt）
  "gen_text_len": 229,            // 文本 decode 步数
  "config_fp16": "./configs/bagel/ours_fp16.cfg",
  "config_ffn_text": "./configs/bagel/ours_ffn_w8.cfg",   // ARGUS 文本 FFN 槽位（T3）
  "sparsity_cross_attn": 0.3819,  // T1：cross-attn 有效 KV 比例
  "result_path": "./results/bagel/results_ours_GEdit.log"
}
```

**硬件 cfg 示例**（`ours_fp16.cfg` 节选）：
```
[architecture_presets]
ArrayHeight:   32
ArrayWidth:    32
IfmapSramSzkB: 32
Bandwidth:     8
```

**精度不在 cfg 里**：所有 `.cfg` 无数据类型字段，精度纯靠文件名后缀嗅探
（`precision_from_config_path`：`ours_int4.cfg`→int4）。各精度 cfg 差异在架构（int4 filter SRAM 128kB / int8 BW16）。

---

## 6. Phase F：能耗核算（`--energy`）

**默认不启用**，加 `--energy` 才算（约 1.5× 慢）。原理：harness 跑每个子仿真时累加 MAC ops + SRAM access +
DRAM access，乘以 per-component pJ 系数，在 `results.log` 追加 `ENERGY BREAKDOWN` 段。

**能量 = 动态(MAC/SRAM/DRAM) + DRAM 背景 idle + 计算阵列静态**。后两项是「运行时间项」，慢的设计
（如 FIGNA 跑 2.6×）多付，是跨硬件能效对比的有效杠杆。

系数表（`coefficients.py`，28nm CMOS baseline，来源见文件内注释）：

| 量 | 默认值 | 来源 |
|---|---|---|
| MAC fp16 / int8 / int4 | 1.5 / 0.2 / 0.1 pJ | Horowitz ISSCC 2014（int4 外推） |
| SRAM 读写 | 0.05 pJ/byte | Sze et al. Proc IEEE 2017 |
| DRAM 读写 | **DDR4 20 pJ/bit**（可切 HBM2 3.9 / HBM3 3.0 / LPDDR4 7.0） | JEDEC / 数据手册 |
| DRAM 背景 idle | **500 mW** | DDR4 子系统估计 |
| 计算阵列静态 `array_static_mw` | **150 mW**（全 2048-lane 漏电+时钟，所有 hw 都付） | 占位，待 RTL 标定 |
| **per-hw w4a16** | ours 0.47 / figna·disagg 0.57 / axcore 0.50 | int4 权重×fp16 激活，最敏感旋钮 |

- **精度自动识别**：从 `.cfg` 文件名后缀嗅探，自动选对应 pJ。
- **workload JSON 覆盖**：可加 `dram_type`（HBM2/HBM3/LPDDR4/DDR4）或 `energy_coefficients`
  （嵌套覆盖 `mac_pj`/`sram_*_pj_per_byte`/`dram_pj_per_bit`/`dram_idle_mw`/`freq_hz`）。
- **`--validate`**（需配 `--energy`）：挑能耗 top-3 子仿真，用 accelergy+CACTI+Aladdin 重算，追加 `CROSS-VALIDATION` 段。
  ⚠️ **ratio 不该是 1.0x**：账本是动态访问下界、accelergy 是含 leak/idle 上界（还因 tile-down 采样把 leak 按体积放大），
  **10x–100x 正常**，真实硅在两者之间。需 accelergy + cacti/aladdin 插件装在当前 conda env。

---

## 7. rebuttal 结果与历史数据（`devlog_rebuttal/`，gitignore）

整个 `devlog_rebuttal/` 被 `.gitignore` 忽略，是本地结果 / 历史数据区，不进版本库。
**新 clone 不会带这个目录**——要么从维护者（dingli）处拷贝，要么用 [`RUN.md`](RUN.md) §3 的重跑脚本重新生成。

- **权威结果**：`overnight_2026-06-17/`（新模型 + 最终系数下的全部对外结果）
  - `RESULTS.md` — 主结果表（§A 加速比/能效比/功率、§B 技术点 ablation、§C 阶段×算子时间、§D 能量分解、§S SenseNova）
  - `BREAKDOWN.md` / `SENSENOVA.md` / `summary_*.csv`
- **专题**：`need9_logs/`（need-9 分离式基线，配 `docs/NEED9_DISAGG.md`）、
  `overnight_2026-06-15/need6/`（need-6 上下文长度/batch/技术点扫描）、
  `img_attn_array_compare/`（图像 attention ARGUS vs 统一阵列）
- **方法学**：`docs/REBUTTAL_FRAMEWORK.md`（每种硬件怎么建模、能量系数、各 need 口径——改了建模就更新它）
- **归档**：`_archive_oldmodel_pre0617/` 是 06-17 之前的作废旧数据，仅留历史对比，可删

### 头条结果（HCA 归一；恒等式 能效比=加速比×功率比）

§A0 摘要（SA / HCA / ARGUS，Geomean = 跨 5 任务几何平均）：

| 硬件 | GEdit加速比 | GEdit能效比 | MM加速比 | MM能效比 | J-GEval加速比 | J-MM能效比 | **Geomean加速比** | **Geomean能效比** |
|---|---|---|---|---|---|---|---|---|
| SA | 1.99 | 1.25 | 1.00 | 1.00 | 1.88 | 1.00 | **1.48** | **1.12** |
| HCA | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** |
| **ARGUS** | 2.65 | 2.42 | 3.10 | 3.09 | 2.02 | 2.95 | **2.55** | **2.44** |

全硬件 Geomean：FIGNA 1.59 / 2.02、AxCore 1.59 / 2.05、FlightVGM 1.48 / 1.12、S-DMA 1.46 / 1.12、**ARGUS 2.55 / 2.44**。
> 完整 §A2 功率、§B ablation、§C 时间分解、§D 能量分解、§S SenseNova 见 `devlog_rebuttal/overnight_2026-06-17/RESULTS.md`。

---

## 8. 测试与回归

没有 pytest，用 shell 对拍金标准 trace：
- `bash test/general/scripts/diff_calc.sh`（还有 `diff_user_{is,os,ws}.sh`）
- `bash test/sparsity/scripts/function_test.sh`

⚠️ **坑**：这些 diff 脚本会用 `sed` 就地改 `configs/scale.cfg` 和 `scalesim/scale.py` 再跑，跑完记得 revert，别误提交。

---

## 9. 已知建模简化 / 坑（接手必读）

有意的简化或不对称，不是 bug，但影响数值解读，改之前想清楚：

- **GQA KV 重复计**：文本注意力 `attn_cycles = attn_single_cycle * num_head_q`，KV 按 query head 数各流一遍，
  没按 GQA 的 kv-head 共享。会**高估文本 KV 访存**（Bagel 28/4=7×、SenseNova 32/8=4×）。
- **单 precision 套三张量**：能量账本 `word_bytes` 由单一 precision 推出后同时套 ifmap/filter/ofmap，低估激活/输出字节。
  DRAM 占能量 99%、但激活只占其中一小块，影响有限。
- **janus/sensenova 无分阶段利用率日志**：STAGE UTIL / DETAIL UTIL 只有 bagel `run_model` 打；利用率分析只在 Bagel GenEdit 做。
- **janus 无 `batch_size`**：need-6 batch 扫描只跑 Bagel GenEdit；给 janus 跑 batch 会静默用 M=1。
- **FlightVGM / FIGNA 走全密 KV**：harness 稀疏只有「缩 KV 注意力长度」一种，只有 `ours`/`sdma` 吃；
  FlightVGM 论文的激活/视觉稀疏类型不同、现机制表达不了 → 对比里被当全密。reviewer 可戳。
- **T2 只缩 ffn_up 未缩 ffn_down**：被 reuse 的 token 理应跳整个 FFN，当前偏保守、低估 T2 收益。
- **日志是文件路径不是目录**：每个 JSON 的 `result_path` 指向一个 `.log` 文件；`run_model` 跑前 `os.remove` 它。
  同一 `result_path` 被多 run 复用是「后跑覆盖」，并行跑会互相踩——driver 都串行跑、跑前手动删 log。

---

## 10. 环境与分支

```bash
source env.sh          # 激活 conda 环境 zhb-scalesim-v3 + HF 镜像（不是通用 venv）
pip install -e .       # 要改 scalesim/* 时装 editable
```

当前分支 `argus-rebuttal`；PR 目标 `main`；git 署名 `dl0321`。
本项目基于上游 [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim)（通用介绍见 `README.md`）。
