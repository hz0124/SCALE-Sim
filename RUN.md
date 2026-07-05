# RUN.md — 运行指令速查

> 项目背景与架构见 [`README_ARGUS.md`](README_ARGUS.md)。本文件只讲「怎么跑」。
> 所有命令都从**仓库根目录**执行。
> 术语（`need-N` / `HCA` / `SA` / `T1–T3` / `disagg` / `w4a16` 等）一律见 `README_ARGUS.md` §1 术语表。

---

## 0. 环境（每次开新 shell 先跑）

```bash
source env.sh          # 激活 conda 环境 zhb-scalesim-v3 + HF 镜像
pip install -e .       # 仅当你要改 scalesim/* 时装 editable（改 harness 不用）
```

---

## 1. MLLM harness —— 本 fork 的主入口

```bash
python run_bagel.py    --hw <HW> --task <TASK> --config <workload.json> [--energy] [--validate] [--batch N]
python run_janus.py    --hw <HW> --task <TASK> --config <workload.json> [--energy] [--validate]
python run_sensenova.py --hw <HW> --task <TASK> --config <workload.json> [--energy] [--validate]
```

**参数取值**

| 参数 | Bagel | Janus | SenseNova |
|---|---|---|---|
| `--hw` | ours / sdma / flightvgm / figna / axcore / base / **disagg** | ours / sdma / flightvgm / figna / axcore / base | ours / sdma / flightvgm / figna / axcore / base |
| `--task` | GenEdit / GenEval / MM | GenEval / MM | GenEval / MM / GenEdit |
| `--config` | `topologies/bagel/*.json` | `topologies/janus/*.json` | `topologies/sensenova/*.json` |
| `--batch N` | ✅（仅 bagel） | ✗（静默 M=1） | ✗ |
| `--energy` | 累加 MAC/SRAM/DRAM 能耗，追加 `ENERGY BREAKDOWN`（约 1.5× 慢） | 同 | 同 |
| `--validate` | 需配 `--energy`；accelergy 交叉校验，追加 `CROSS-VALIDATION` | 同 | 同 |

> **`--task` 是什么**：`GenEdit`=图像编辑、`GenEval`(=GenImage)=图像生成、`MM`=多模态理解（纯文本，无图像扩散阶段）。
> **`--hw` 是什么**：`ours`=ARGUS（本文加速器）；`figna`/`axcore`/`flightvgm`/`sdma`=对比 baseline 加速器；
> `base`=SA 纯脉动阵列；`disagg`=分离式基线；`HCA`（非 `--hw` 值）=ours 全技术关的归一基准。各自细节见术语表。
>
> `--hw` 与 `--config` 必须匹配：JSON 里的 cfg 指向要对上硬件。命名约定 `config_<hw>[_<task>].json`。

**config 命名对照（挑常用）**

| 硬件 | Bagel GenEdit | Bagel GenEval | Bagel MM | Janus GenEval | Janus MM |
|---|---|---|---|---|---|
| ours | `config_ours.json` | `config_ours_GenEval.json` | `config_ours_MM.json` | `config_ours_janus_GenEval.json` | `config_ours_janus_MM.json` |
| figna | `config_figna.json` | `config_figna_GenEval.json` | `config_figna_MM.json` | `config_figna_janus_GenEval.json` | `config_figna_janus_MM.json` |
| flightvgm | `config_flightvgm.json` | `config_flightvgm_GenEval.json` | `config_flightvgm_MM.json` | `config_flightvgm_janus_GenEval.json` | `config_flightvgm_janus_MM.json` |
| sdma | `config_sdma.json` | `config_sdma_GenEval.json` | `config_sdma_MM.json` | `config_sdma_janus_GenEval.json` | `config_sdma_janus_MM.json` |

> `axcore` 复用 figna 的 JSON（仅能量系数不同）。`disagg`（分离式基线）用 `config_disagg*.json`，仅 Bagel。
> SenseNova 用 `topologies/sensenova/config_{ours,HCA}_{GenEdit,GenEval,MM}.json`。
> ⚠️ **`HCA` 不是合法 `--hw`**——它是 ours 全技术关的归一基准，JSON 里用的是 `ours_*.cfg`，
> 所以跑 HCA 配置时 `--hw` 仍填 `ours`（见下方示范）。

**典型命令**

```bash
# ARGUS（ours）跑 Bagel 图像编辑
python run_bagel.py --hw ours --task GenEdit --config topologies/bagel/config_ours.json

# 带能耗核算 + accelergy 交叉校验
python run_bagel.py --hw ours --task MM --config topologies/bagel/config_ours_MM.json --energy --validate

# FIGNA baseline 跑 Janus 多模态理解
python run_janus.py --hw figna --task MM --config topologies/janus/config_figna_janus_MM.json

# AxCore：复用 figna 的 JSON（只是能量系数不同，无独立 config_axcore*.json）
python run_bagel.py --hw axcore --task GenEdit --config topologies/bagel/config_figna.json

# disagg（分离式基线，仅 Bagel）：config_disagg_w<N>.json 里的 N 是 split 档位（注意这个 w 是切分档位，跟精度 w4a16 无关）
# （现成 plain config 只有 w2/w16/w32，w4/w8 仅 _bwhalf 变体；新模型最优 split=w2，见 devlog_rebuttal/docs/NEED9_DISAGG.md）
python run_bagel.py --hw disagg --task GenEdit --config topologies/bagel/config_disagg_w2.json

# SenseNova 支线（ours）
python run_sensenova.py --hw ours --task GenEval --config topologies/sensenova/config_ours_GenEval.json

# HCA 归一基准：--hw 填 ours，换成 HCA 的 JSON（HCA 本身不是合法 --hw）
python run_sensenova.py --hw ours --task GenEval --config topologies/sensenova/config_HCA_GenEval.json
```

**输出**：结果写到 JSON 里 `result_path` 指向的 `.log`（通常 `results/{bagel,janus}/*.log`），
含端到端 cycle、按 500 MHz 换算的秒、可选 `ENERGY BREAKDOWN` / `CROSS-VALIDATION` 段。
跑完终端会打印 `Simulation completed successfully!` + 总 cycle。
- `result_path` 是**文件路径**不是目录；其父目录（如 `results/bagel/`）需已存在。
- ⚠️ 脚本末尾那句 `... {result_path}/results.log` 的打印是**误导**（result_path 已是文件），以 JSON 里的 `result_path` 为准。

⚠️ **必须串行跑**：所有子仿真共享 `topologies/{bagel,janus}/layer.csv`，并行会互相踩。
同一 `result_path` 被多个 run 复用是「后跑覆盖」。

---

## 2. 上游单工作负载路径（底层引擎，一般不直接用）

```bash
# CONV 拓扑（默认）
python scalesim/scale.py -c configs/scale.cfg -t topologies/conv_nets/Resnet18.csv -p ./results
# GEMM MNK 拓扑要加 -i gemm
python scalesim/scale.py -c configs/scale.cfg -t topologies/GEMM_mnk/vit_s.csv -p ./results -i gemm
```

输出 `<-p>/<run_name>/{COMPUTE,BANDWIDTH,DETAILED_ACCESS}_REPORT.csv` + 逐层 SRAM/DRAM trace
（`run_name` 取自 cfg 的 `[general]` 段）。

---

## 3. 复现 rebuttal 权威结果

> `need-N` = rebuttal 中 reviewer 编号的需求（need-6=上下文/batch 扫描，need-9=分离式基线，详见术语表）。
> `§A–§S` 是 `RESULTS.md` 里的结果小节名。

权威结果在 `devlog_rebuttal/overnight_2026-06-17/`。重跑脚本（**都从仓库根目录跑，串行**）：

```bash
# §A 大表（HCA 归一，6 硬件 × 5 任务）→ summary_baseA.csv
python devlog_rebuttal/overnight_2026-06-17/run_baseA.py

# §B 技术点 ablation 全矩阵 + 补充 → summary_full.csv / summary_supp.csv
python devlog_rebuttal/overnight_2026-06-17/run_full.py
python devlog_rebuttal/overnight_2026-06-17/run_supp.py

# 重新生成 RESULTS.md 的 §A/§B/§C（⚠️ 会冲掉手工维护的 §D/§S，记得从版本里贴回）
python devlog_rebuttal/overnight_2026-06-17/compose_RESULTS.py
```

need-6（上下文/batch/技术点扫描，输出硬编码到 `overnight_2026-06-15/need6/`）：

```bash
python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_ctxlen_sweep.py
python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_batch_sweep.py
python devlog_rebuttal/overnight_2026-06-15/need6/gen_need6_summary.py     # 从 JSON 重生成总结 md
```

need-9（分离式 split 扫描）：

```bash
python devlog_rebuttal/need9_logs/aggregate_splits.py    # 读 results/bagel/results_disagg_*.log → split_sweep.csv
```

图像 attention 专题（in-process，不污染 ./results）：

```bash
python devlog_rebuttal/img_attn_array_compare/compare_img_attn.py
```

---