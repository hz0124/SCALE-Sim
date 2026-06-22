# SenseNova 结果（支线，2026-06-18）

新模型 SenseNova 接入 harness（`simulation_core/sensenova_sim.py`，基于当前 Janus 结构 +
2026-06-17 memory-bound 文本模型）。三任务：**MM-Vet**（纯文本理解）、**GenEval**（文生图）、
**GEdit-Bench-EN**（图像编辑）。本轮只跑 **ours（完整 ARGUS）+ HCA（all-off 基线）**。

> ⚠️ 本文件是 SenseNova 数据的**权威副本**。`RESULTS.md` 由 `compose_RESULTS.py` 从 CSV
> 重新生成（整文件覆盖写），手工加进 RESULTS.md 的 SenseNova 段会在下次 compose 时被冲掉。
> 要长期进 RESULTS.md，需把 SenseNova 接进 compose 流水线（见末尾）。

## 模型结构（GQA）

| 参数 | 值 |
|---|---|
| num_layer | 42 |
| dim | 4096 |
| head_dim | 128 |
| num_head_q | 32 |
| num_head_kv | 8（GQA） |
| upshape（intermediate） | 12288（gated SwiGLU → ffn_up ×2） |

## 每任务 workload（源数据 → 配置）

| 任务 | kv_cache_init | gen_text_len | activation(image_input_len) | gen_image_step | 文本阶段 | ViT/cross 稀疏 | Self 稀疏 | FFN reuse |
|---|---|---|---|---|---|---|---|---|
| MM-Vet | 1007（prefill） | 680(ours)/685(HCA)（decode） | — | — | 有 | — | — | — |
| GenEval | 15（文本提示）* | 0 | 4096 | 50 | **无**（flag） | 无（GenEdit 才有） | 无 | 0.7347 |
| GEdit-Bench | 4026（=ViT len）* | 196（text_gen） | 4484 | 50 | 有 | 0.816 裁→`scn=0.184` | 0.604（精度拆分）→`lps=0.604` | 0.8285 |

\* 占位/推断值，见"假设与口径"。映射规则（沿用团队口径）：ViT sparsity=裁掉比例→`sparsity_cross_attn=1−x`；
Self sparsity=低/高精度拆分→`low_precise_self_attn=x`（直填）；FFN reuse rate→`image_only_sim`。

## 结果（@500 MHz）

| 任务 | 硬件 | cycles | 秒 | 能量(mJ) | 平均功率(W) |
|---|---|---|---|---|---|
| MM-Vet | ours | 5.547e10 | 110.9 | 648,077 | 5.84 |
| MM-Vet | HCA | 1.860e11 | 372.0 | 2,164,484 | 5.82 |
| GenEval | ours | 1.704e12 | 3407.1 | 6,212,360 | 1.82 |
| GenEval | HCA | 4.011e12 | 8022.0 | 13,473,691 | 1.68 |
| GEdit-Bench | ours | 3.157e12 | 6314.5 | 12,393,527 | 1.96 |
| GEdit-Bench | HCA | 7.136e12 | 14272.2 | 25,911,461 | 1.82 |

## 加速比 / 能效比（HCA 归一；恒等式 能效比=加速比×功率比）

| 硬件 | S-MM加速比 | S-MM能效比 | S-GEval加速比 | S-GEval能效比 | S-GEdit加速比 | S-GEdit能效比 |
|---|---|---|---|---|---|---|
| HCA | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ARGUS | **3.35** | **3.34** | **2.35** | **2.17** | **2.26** | **2.09** |

（MM 3.35/3.34，GEval 2.35/2.17，GEdit 2.26/2.09。MM 收益最大：纯文本访存受限，ARGUS 投影
+FFN 全 W4A16 对齐 figna，4× 权重带宽。以上 GenEval/GenEdit 为图像 FFN INT4=1/2 默认值。）

## 图像 FFN INT4/FP16 任务量拆分 sweep（精度未定）

图像 T3 的 FFN 任务量当前默认 1/2 INT4 + 1/2 FP16。扫 INT4 份额 f ∈ {1/2, 2/5, 1/3, 1/4, 1/5}
（FP16=1−f）。建模:INT4 份跑 FP-INT4 阵列、FP16 份跑 FP-FP 阵列,各宽 32、并行 → 等效 FFN 宽
= 32/(1−f)（f≤1/2 时 FP16 半为瓶颈）;能量 MAC 按 (1−f) fp16 + f w4a16 计费。只影响图像;
**S-MM 恒 3.35/3.34（无图像阶段）;HCA 不受影响（T3 关）**。机制:`sensenova_sim.py` 新增 JSON
字段 `ffn_int4_frac`（默认 0.5,复现原值）。

| 配置 | S-MM加速比 | S-MM能效比 | S-GEval加速比 | S-GEval能效比 | S-GEdit加速比 | S-GEdit能效比 |
|---|---|---|---|---|---|---|
| HCA | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ARGUS INT4=1/2（默认） | 3.35 | 3.34 | 2.35 | 2.17 | 2.26 | 2.09 |
| ARGUS INT4=2/5 | 3.35 | 3.34 | 2.11 | 2.02 | 2.04 | 1.96 |
| ARGUS INT4=1/3 | 3.35 | 3.34 | 1.97 | 1.93 | 1.92 | 1.89 |
| ARGUS INT4=1/4 | 3.35 | 3.34 | 1.83 | 1.83 | 1.79 | 1.80 |
| ARGUS INT4=1/5 | 3.35 | 3.34 | 1.75 | 1.78 | 1.71 | 1.76 |

INT4 份额越低 → FP16 半阵列越成瓶颈、FFN 越慢、fp16 MAC（1.5pJ）越多 → 加速/能效单调下降（精度↑的代价）。

## 假设与口径（需确认）

1. **GenEval 无文本阶段**（已确认）：`text_gen_finished_flag=True, text_gen_cycles=0`，image
   attention KV = 15+0+4096=4111（与给定 KV 均值一致）。
2. **GenEval 仅 FFN reuse**（已确认）：无 ViT/cross 稀疏、无 Self 稀疏，只 `image_only_sim=0.7347`。
3. **GEdit 2-branch 估计**：当前 image 阶段沿用 Janus 的 2 分支（full_cache/without_text），
   **未建模 GenEdit 的第 3 个 CFG 分支（without_img，丢输入图）** → GEdit image 成本是 ~2 分支
   下界；若 SenseNova GEdit 用 3 路 CFG，请给第 2 个 reuse rate，我补 without_img。
4. **GEdit `t1_hard=false`**：为让 Self 稀疏（0.604）经 dispatch 路径生效（`t1_hard=true` 的
   均衡路径会忽略 `low_precise_self_attn`）。这偏离"ours 默认 t1_hard=true"，需确认口径。
5. **kv_cache_init 占位**：GenEval=15（=4111−4096，文本提示长度）；GEdit=4026（=ViT len，输入图
   作前缀上下文）。GEdit harness full_cache KV=4026+196+4484=8706（报告 KV 均值 8510=ViT+activation，
   不含文本 decode）。确认真实文本提示长度。
6. **text_attn/vae_attn 拆分**占位（2/4094 等），文本侧投影成本极小，影响可忽略。
7. **MM 文本 FFN 走 W4A16**（figna 对齐，2026-06-17 决定）：T3 开 + task=MM → FFN INT4。

## 接进 RESULTS.md 流水线（待办）

`compose_RESULTS.py` 从 `baseA_table.csv` 等读数生成 §A。要让 SenseNova 永久进 RESULTS.md：
给 SenseNova 也产一份同格式 CSV（或扩 `run_baseA.py` 跑 sensenova），再在 compose 里加
SenseNova 列/段。当前先以本文件为准 + RESULTS.md 末尾手工段。
