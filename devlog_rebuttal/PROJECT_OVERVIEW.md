# ARGUS-SCALE-Sim 项目全貌速读

> 维护者：dingli ｜ 首次整理：2026-06-09
> 本文档是「一次性把项目看懂」的速读版。日常进度记录在同目录 `DEVLOG.md`。

---

## 1. 这个项目是什么

- 它是 **SCALE-Sim v3**（上游开源的「脉动阵列 / systolic-array」cycle-accurate 加速器仿真器）的一个 **fork**。
- Fork 的目的：为 **ARGUS** 这篇论文提供实验数据（投稿见 `README_MLLM.md`；当前主线是 **MICRO 2026 rebuttal**，论文 PDF `docs/micro2026-paper2908.pdf`）。
- ARGUS 是一个面向 **统一多模态大模型（Unified MLLM）解码阶段** 的加速器。本仓库在上游仿真器之上，搭了一套 **MLLM 解码吞吐量评测 harness**，用来评测：
  - 两个模型：**Bagel**、**Janus**；
  - 多个任务：`GenEdit`（图像编辑）、`GenEval`(=`GenImage`，图像生成)、`MM`（图像理解 / 多模态）；
  - 多套硬件：`ours`(=ARGUS)、`figna`(FIGNA)、`axcore`(AxCore=FIGNA 同款 W4A16、仅 w4a16 系数更低)、`flightvgm`、`sdma`(S-DMA)、`base`(=SA 纯脉动阵列)、`disagg`(need-9 分离式基线编排器)。另有「软件配方」基线 **HCA** = `ours` 全技术关 (all_off 异构阵列)，用作 §A 归一基准。
  - 支线模型 **SenseNova**（`simulation_core/sensenova_sim.py`，基于 Janus 结构 + 新文本模型），见 `overnight_2026-06-17/SENSENOVA.md`。

**一句话**：用 SCALE-Sim 当底层「算一层 GEMM 要多少 cycle」的引擎，上层 harness 把一整个 MLLM 解码轨迹拆成成百上千个小 GEMM 子仿真，按层数 / head 数 / KV 长度 / 采样步长加权求和，得到端到端 cycle，再换算成吞吐/能耗。

---

## 2. 两条运行路径（都会用到）

### 路径 A：上游单工作负载（底层引擎）
```bash
python scalesim/scale.py -c configs/scale.cfg -t topologies/conv_nets/Resnet18.csv -p ./results
python scalesim/scale.py -c configs/scale.cfg -t topologies/GEMM_mnk/vit_s.csv   -p ./results -i gemm   # -i gemm 走 MNK
```
跑「1 个 config + 1 个 topology」，输出 `COMPUTE_REPORT.csv` / `BANDWIDTH_REPORT.csv` / `DETAILED_ACCESS_REPORT.csv` + 逐层 SRAM/DRAM trace。

### 路径 B：MLLM harness（本 fork 的主战场）
```bash
python run_bagel.py --hw ours      --task GenEdit --config ./topologies/bagel/config_ours.json
python run_janus.py --hw flightvgm --task MM      --config ./topologies/janus/config_flightvgm_janus_MM.json
```
`run_*.py` 只是 argparse 壳，真正逻辑在 `simulation_core/{bagel,janus}_sim.py`。结果写到 JSON 里 `result_path` 指向的 `results/.../*.log`。

---

## 3. 代码结构地图

```
SCALE-Sim/
├── run_bagel.py / run_janus.py        # 入口壳（argparse: --hw --task --config --energy --validate）
├── simulation_core/                   # ★ 本 fork 的核心 harness（不是上游 scalesim 的一部分，是调用它）
│   ├── bagel_sim.py   (870 行)        # Bagel_sim：编排一整个解码轨迹
│   ├── janus_sim.py   (740 行)        # Janus_sim：同上
│   └── energy_accounting/             # ★ 当前分支新增的「能耗核算」(Phase F)
│       ├── coefficients.py            #   pJ 系数表（MAC/SRAM/DRAM，28nm baseline）
│       ├── accountant.py              #   EnergyAccountant 累加器 + 报表生成
│       ├── extractors.py              #   从 SCALE-Sim 报表里抽 access count
│       └── validate.py                #   --validate：用 accelergy CLI 交叉校验
├── scalesim/                          # 上游 cycle-accurate 引擎
│   ├── scale_sim.py                   #   public façade（scalesim 类）
│   ├── simulator.py                   #   每个 topology 行 → 一个 single_layer_sim
│   ├── single_layer_sim.py
│   ├── compute/                       #   三种数据流：systolic_compute_{ws,os,is}.py + operand_matrix + sparsity
│   └── memory/                        #   双缓冲 scratchpad + read/write buffer/port
├── configs/bagel/*.cfg                # 硬件规格（ArrayHeight/Width, *SramSzkB, Bandwidth, Dataflow…）
│                                      #   ours/figna 有按精度分的 *_fp16/_int8/_int4.cfg
├── topologies/{bagel,janus}/*.json    # 仿真流程配置（KV、gen_len、稀疏率、阵列尺寸、指向 .cfg）
├── topologies/<family>/*.csv          # 上游风格工作负载（CONV 默认；GEMM_mnk/ 需 -i gemm）
├── rundir-accelergy/                  # accelergy/CACTI/Aladdin 交叉校验运行目录
└── results/{bagel,janus}/             # harness 输出 log
```

---

## 4. harness 内部流程（bagel_sim.py 为例）

`Bagel_sim` 的方法链（行号见 `grep` 结果）：
1. `read_from_json(cfg_path)` (L188) — 读工作负载 + 硬件 cfg 路径。**关键 dispatch**：
   - `ours`/`figna`：按精度分多个 cfg（`config_fp16/int4/int8`）→ 字段 `config_comp0/1`、`config_comm0/1`；
   - `base`/`flightvgm`/`sdma`：一个 `config` 复用四个槽位。
   - **新增硬件类型就改这里的 if/elif 链**（bagel 和 janus 都要改）。
2. `build_topologies(kv_len, is_gen_text, part)` (L264) — 给每个子仿真写一份单层 GEMM topology 到 `topologies/bagel/layer.csv`。`part` 选投影：`qkv / attn_qk / attn_sfmxv / omap / ffn_up / ffn_down`。
3. `run_sim_once(...)` (L354) — 调上游 scalesim（GEMM 模式，layout `layouts/GEMM_mnk/vit_l_KM_KN.csv`），返回该单层 cycle。**2026-06-17 起仅图像阶段文本侧投影用它**；文本 decode 已迁出（见 4b）。
4. `run_sim_once_comp(...)` (L505) — **解析式**替代：直接用 array_height/width + tile + op 尺寸算 fold cycle，不调 SCALE-Sim（快很多）。图像生成内循环用它。
4b. `_text_gemm_cycles(...)`（2026-06-17 新增，bagel+janus）— **文本 decode 解析 memory-bound roofline**：`cycles = max(M·N·K/text_array_macs, N·K·operand_bytes/BW)`。取代了文本阶段的 `run_sim_once`（OS 把 M=1 误建成 compute-bound、盖死带宽）。详见 `docs/REBUTTAL_FRAMEWORK.md` §3.1 路径 C / `docs/RERUN_PLAN.md`。
5. `run_gen_text()` (L406) / `run_gen_image()` (L596) — 组合每步 cycle，按层数/head/采样步长/KV 长度放大求和。
   - 稀疏性建模 = 缩小有效 KV 长度：`ours` 用 `sparsity_cross_attn` + `low_precise_self_attn`；`sdma` 用 `sparsity_kv`；baseline 用全量 KV。
6. `run_model()` (L789) — 先文本生成，再（GenEdit/GenImage）图像生成，写 `result_path`。

**时钟**：harness 默认 **500 MHz**（`_append_to_log` 里把 cycle 换算成秒）。要改时钟改那里。

---

## 5. 两级配置系统（容易混）

| 层级 | 文件 | 管什么 |
|---|---|---|
| 仿真流程 | `topologies/{bagel,janus}/config_<hw>[_<task>].json` | KV 大小、gen_text_len、gen_image_step、稀疏率、阵列尺寸、buffer 大小/带宽、采样率、指向 .cfg |
| 硬件规格 | `configs/bagel/<hw>*.cfg` | ArrayHeight/Width、*SramSzkB、Bandwidth、Dataflow、可选 [layout]/[sparsity] |
| 上游工作负载 | `topologies/<family>/*.csv` | CONV（默认）或 MNK（`-i gemm`） |

---

## 6. Phase F：能耗核算（当前分支 `feat/argus-energy-accounting` 的工作）

这是当前分支最近 10 个 commit（F1–F8）做的事。**默认不启用**，加 `--energy` 才算（约 1.5x 慢）。

- 原理：harness 跑每个子仿真时，累加 MAC ops + SRAM access + DRAM access，乘以 per-component pJ 系数，最后在 `results.log` 追加 `ENERGY BREAKDOWN` 段。
- 系数表（`coefficients.py`，28nm CMOS baseline）：MAC fp16=1.5pJ / int8=0.2pJ / int4=0.1pJ；SRAM 0.05pJ/byte；DRAM 读写 **默认 DDR4 20 pJ/bit**（`dram_type` 可切 HBM2 3.9 / LPDDR4 7.0）；**DRAM 背景 idle 500mW**；**计算阵列静态 `array_static_mw`=150mW**（全 2048-lane 漏电+时钟，按运行时间计，所有 hw 都付 → 罚慢设计）。
  - **能量 = 动态(MAC/SRAM/DRAM) + DRAM 背景 + 阵列静态**。后两项是「运行时间项」，慢的设计（如 FIGNA 跑 2.6×）多付，是跨硬件能效对比的有效杠杆。
  - **per-hw w4a16 系数（2026-06-17）**：int4 权重 × fp16 激活的 MAC，按硬件区分——`ours`(ARGUS) 0.47 / `figna`·`disagg` 0.57 / `axcore` 0.50（其余 hw 不发 w4a16）。在 `coefficients.py` 字典构造后用 per-hw 覆盖实现；workload JSON 也可用 `dram_type` / `energy_coefficients` 进一步覆盖。这是能效结论**最敏感的旋钮**（FIGNA 90% MAC 是 w4a16）。详见 `docs/REBUTTAL_FRAMEWORK.md` §5。
- 精度自动识别：从 `.cfg` 文件名后缀（`*_fp16/_int8/_int4.cfg`）嗅探，自动选对应 pJ。
- `--validate`（需配合 `--energy`）：挑能耗 top-3 的子仿真，用 accelergy + CACTI + Aladdin 重算交叉校验，追加 `CROSS-VALIDATION` 段。
  - ⚠️ **ratio 不该是 1.0x**：EnergyAccountant 是「动态访问能耗下界」（无 leak/idle）；accelergy 是「含 leak/idle 的上界」，又因 tile-down 采样把 leak 按体积还原放大，所以 10x–100x 是正常的。真实硅在两者之间。

---

## 7. 测试与回归

- 没有 pytest，用 shell 对拍金标准 trace：
  - `bash test/general/scripts/diff_calc.sh`（还有 `diff_user_{is,os,ws}.sh`）
  - `bash test/sparsity/scripts/function_test.sh`
- ⚠️ **坑**：这些 diff 脚本会用 `sed` 就地改 `configs/scale.cfg` 和 `scalesim/scale.py` 再跑，跑完记得 revert，别误提交。

---

## 8. 环境

```bash
source env.sh          # 激活 conda 环境 zhb-scalesim-v3 + HF 镜像（不是通用 venv）
pip install -e .       # 要改 scalesim/* 时装 editable
```

---

## 9. 分支情况

- 当前：`feat/argus-energy-accounting`（能耗核算，F1–F8 已做）。
- 其它本地分支：`dev_zhb_order`（前人 zhb 的主开发线，CLAUDE.md 里写 PR 目标是 main）、`dev_micro26_rebuttal`、`feat/accelergy-merge`。
- PR 目标：`main`。git 署名已设为 `dl0321`。

---

## 10. 已知建模简化 / 坑（接手必读）

这些是有意的简化或不对称，不是 bug，但影响数值解读，改之前先想清楚：

- **GQA KV 重复计**：文本注意力 `attn_cycles = attn_single_cycle * num_head_q`，KV 按 query head
  数各流一遍，没按 GQA 的 kv-head 共享（应 ×`num_head_kv`）。bagel/janus/sensenova 三者一致。
  会**高估文本 KV 访存**（Bagel 28/4=7×、SenseNova 32/8=4×）。要严肃 KV 带宽数值时需确认。
- **单 precision 套三张量**：能量账本 `word_bytes` 由单一 precision 推出后同时套 ifmap/filter/ofmap，
  低估激活/输出字节（`bagel_sim.py` 能量段）。DRAM 占能量 99%、但激活只占其中一小块，影响有限。
- **janus / sensenova 无分阶段利用率日志**：STAGE UTIL / DETAIL UTIL（Fig14 那套 4 利用率）只有 bagel
  `run_model` 打。need-9/利用率分析目前只在 Bagel GenEdit 做。
- **janus 无 `batch_size`**：need-6 batch 扫描只跑 Bagel GenEdit；给 janus 跑 batch 会静默用 M=1。
- **FlightVGM / FIGNA 走全密 KV**：harness 稀疏只有「缩 KV 注意力长度」一种，只有 `ours`/`sdma` 吃；
  FlightVGM 论文的激活/视觉稀疏类型不同、现机制表达不了 → 对比里 FlightVGM 被当全密。reviewer 可戳。
- **日志是文件路径不是目录**：每个 JSON 的 `result_path` 指向一个 `.log` 文件；`run_model` 跑前会
  `os.remove` 它（2026-06-18 修好，之前是空操作导致 append 污染）。同一 `result_path` 被多个 run 复用时
  仍是「后跑覆盖」，并行跑会互相踩——driver 都串行跑、且跑前手动删 log。
